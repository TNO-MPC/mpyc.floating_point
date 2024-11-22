"""
Module that contains logic for secret-shared floating point numbers.
"""

from __future__ import annotations

import functools
import sys
import types
from math import ceil, floor, log2
from typing import TYPE_CHECKING, Any, TypeVar, cast

from mpyc.finfields import PrimeFieldElement
from mpyc.sectypes import SecInt, SecureInteger, SecureNumber, SecureObject

from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point import basic_functions, extended_coroutine
from tno.mpc.mpyc.floating_point._add import (
    AdditionStrategy,
    CacheAllTwoExponentAddition,
    ComputeTwoExponentAddition,
    IgnoreTwoExponentAddition,
    _add,
)
from tno.mpc.mpyc.floating_point._mpyc_cached_attribute import (
    CachedMPyCAttribute,
    convert_to_stype,
)
from tno.mpc.mpyc.floating_point._mul import _mul
from tno.mpc.mpyc.floating_point._plaintext_helper_functions import (
    calculate_two_to_exponent_mod,
)
from tno.mpc.mpyc.floating_point._sys import SYSTEM_EXPONENT_BITS
from tno.mpc.mpyc.floating_point.vendor.mpyc._divmod import secret_div

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if TYPE_CHECKING:
    from tno.mpc.mpyc.stubs import BaseSecureFloat

    # The BaseSecureFloat class allows mypy to recognize SecureFloatingPoint as a class that is
    # somewhat similar to mpyc.sectypes.SecureFloat. This is required to leverage
    # tno.mpc.mpyc.stubs to its fullest (that module is unaware of
    # tno.mpc.mpyc.floating_point.SecureFloatingPoint).
    SecureNumber_ = BaseSecureFloat
else:
    SecureNumber_ = SecureNumber

SecureObjectT = TypeVar("SecureObjectT", bound=SecureObject)


class SecureFloatingPoint(SecureNumber_):
    """Base class for secure (secret-shared) floating-point numbers."""

    __slots__ = (
        "significand_bit_length",
        "exponent_bit_length",
        "significand_class",
        "exponent_class",
        "two_to_exponent_class",
        "max_concurrent_additions",
        "max_concurrent_multiplications",
    )
    share: tuple[SecureInteger, SecureInteger]
    significand_bit_length: int
    exponent_bit_length: int
    significand_class: type[SecureInteger]
    exponent_class: type[SecureInteger]
    two_to_exponent_class: type[SecureInteger]
    max_concurrent_additions: int
    max_concurrent_multiplications: int
    _is_zero: CachedMPyCAttribute
    _two_to_exponent: CachedMPyCAttribute

    @property
    def significand(self) -> SecureInteger:
        """
        The significand of the secure floating point.

        :return: Secret-shared significand.
        """
        return self.share[0]

    @property
    def exponent(self) -> SecureInteger:
        """
        The exponent of the secure floating point.

        :return: Secret-shared exponent.
        """
        return self.share[1]

    def is_zero(self) -> SecureInteger:
        """
        Get secure indication of equality with zero.

        The result is cached for efficient reuse.

        :return: Secret-shared 1 if the floating point equals zero, secret-shared 0 otherwise.
        """
        return self._is_zero.compute()

    async def has_cached_is_zero(self) -> bool:
        """
        Indicate whether the is_zero value is cached.

        :return: True if the is_zero value is cached.
        """
        return await self._is_zero.is_cached()

    def two_to_exponent(self) -> SecureInteger:
        """
        Get base-2 exponentiation of the exponent of the secure floating point.

        The result is cached for efficient reuse.

        :return: Secret-shared base-2 exponentiation.
        """
        return self._two_to_exponent.compute()

    async def has_cached_two_to_exponent(self) -> bool:
        """
        Indicate whether the two_to_exponent value is cached.

        :return: True if the two_to_exponent value is cached.
        """
        return await self._two_to_exponent.is_cached()

    @classmethod
    def _input(
        cls, secflps: list[SecureFloatingPoint], senders: list[int]
    ) -> list[list[SecureFloatingPoint]]:
        """
        Secret-share a list of SecureFloatingPoints. This method is called internally by mpc.input.

        :param secflps: A secure object containing a local share or Future.
        :param senders: A list of party identifiers representing who will share inputs.
        :return: A list of secure objects representing secret shares, or just one if there is one
            sender.
        """

        @extended_coroutine.mpc_coro_extended
        async def _mpyc_input() -> list[list[SecureFloatingPoint]]:
            """
            Secret-share a list of SecureFloatingPoints.

            :return: A list of secure objects representing secret shares.
            """
            secure_flp_type = type(secflps[0])
            await extended_coroutine.returnType(
                secure_flp_type, len(senders), len(secflps)
            )

            significands_distributed = [
                flp.mpc.input(secflp.significand, senders=senders) for secflp in secflps
            ]
            exponents_distributed = [
                flp.mpc.input(secflp.exponent, senders=senders) for secflp in secflps
            ]
            # contents: significands_distributed = [
            #     [
            #         secflp_1_sender_1_significand,
            #         ...,
            #         secflp_1_sender_len(senders)_significand,
            #     ],
            #     [
            #         secflp_2_sender_1_significand,
            #         ...,
            #         secflp_2_sender_len(senders)_significand,
            #     ],
            #     ...,
            # ]
            secflps_distributed = []

            for secflp, significand_distributed, exponent_distributed in zip(
                secflps, significands_distributed, exponents_distributed
            ):
                # Dynamically get the appropriate values for the CachedMPyCAttributes.
                # This assumes that if secure_value._attr is a CachedMPyCAttribute, then it is
                # passed to the initializer as argument "attr".
                cached_mpyc_attribute_names = [
                    attr
                    for attr in dir(secflp)
                    if isinstance(getattr(secflp, attr), CachedMPyCAttribute)
                ]

                # For every CachedMPyCAttribute and every sender, find out whether or not the attribute is
                # actually cached. If so, store the value for the SecureFloatingPoint initializer.
                cached_mpyc_attributes: dict[str, list[SecureInteger | None]] = {}
                for attr_name in cached_mpyc_attribute_names:
                    attr_values = []
                    for sender in senders:
                        is_attr_cached = await flp.mpc.transfer(
                            (
                                await getattr(secflp, attr_name).is_cached()
                                if flp.mpc.pid == sender
                                else None
                            ),
                            senders=sender,
                        )
                        attr_val: SecureInteger | None = None
                        if is_attr_cached:
                            attr_val = flp.mpc.input(
                                getattr(secflp, attr_name)._value, senders=sender
                            )
                        attr_values.append(attr_val)
                        # contents: attr_values = [
                        #     secflp_sender_1_attr, secflp_sender_2_attr, ...,
                        # ]

                    initializer_arg_name = attr_name.lstrip("_")
                    cached_mpyc_attributes[initializer_arg_name] = attr_values
                    # contents: cached_mpyc_attributes = {
                    #     "attr1": [secflp_sender_1_attr1, secflp_sender_2_attr1, ...],
                    #     "attr2": [secflp_sender_1_attr2, secflp_sender_2_attr2, ...]
                    # }

                secflp_distributed = []
                for i, (sign, exp) in enumerate(
                    zip(significand_distributed, exponent_distributed)
                ):
                    mpyc_cached_attr_kwargs = {
                        attr_name: attr_vals[i]
                        for attr_name, attr_vals in cached_mpyc_attributes.items()
                    }
                    # contents: mpyc_cached_attr_kwargs = {
                    #     "attr1": secflp_sender_i_attr1,
                    #     "attr2": secflp_sender_i_attr2,
                    # }
                    secflp = secure_flp_type(
                        value=(sign, exp), **mpyc_cached_attr_kwargs
                    )
                    secflp_distributed.append(secflp)

                secflps_distributed.append(secflp_distributed)

            # We now have a list of lists, where the length of the outer list is len(secflp) and
            # the length of each inner list is len(senders). mpc._distribute expects the reverse,
            # so we "transpose" the lists
            return list(zip(*secflps_distributed))  # type: ignore[arg-type]

        return _mpyc_input()

    @classmethod
    async def _output(
        cls,
        obj: list[SecureFloatingPoint],
        receivers: int | list[int],
        threshold: int,
    ) -> list[float]:
        """
        Reconstruct a list of secret-shared floating points. This method is called internally by
        mpc.output.

        :param obj: Object to be reconstructed.
        :param receivers: Party IDs of the parties to receive the output.
        :param threshold: The threshold for reconstruction.
        :return: Reconstructed list of floats.
        """
        significands = [elem.significand for elem in obj]
        exponents = [elem.exponent for elem in obj]
        significands_output = await flp.mpc.output(
            significands, receivers=receivers, threshold=threshold
        )
        exponents_output = await flp.mpc.output(
            exponents, receivers=receivers, threshold=threshold
        )

        if significands_output[0] is None:
            # not one of the receivers, so return list of None
            return [None] * len(obj)
        return [
            cls.output_conversion(*pair)
            for pair in zip(significands_output, exponents_output)
        ]

    @classmethod
    def output_conversion(
        cls, sign: PrimeFieldElement | int, exp: PrimeFieldElement | int
    ) -> float:
        """
        Function that is called on the reconstructed finite field elements representing the significand and exponent
        to create interpretable output.

        :param sign: Significand finite field element.
        :param exp: Exponent finite field element.
        :return: A regular float representing the same value.
        :raises ValueError: Provided exponent is too large to convert the inputs into a python float.
        """
        # first cast to Python internal floating point representation +/- 1.abcd * 2 ** e to prevent unnecessary overflow
        if int(sign) == 0:
            return 0.0

        exp = int(exp)
        biased_exponent = exp + cls.significand_bit_length - 1
        if (biased_exponent_length := exp.bit_length()) > SYSTEM_EXPONENT_BITS:
            raise ValueError(
                f"Obtained (IEEE-754) exponent ({biased_exponent}) has {biased_exponent_length}"
                " bits, which is too much for the system to convert. This is probably caused by a"
                " bug in one of the functions that returned this SecureFloatingPoint (or one of"
                " its predecessors)."
            )
        return float(
            int(sign) / 2 ** (cls.significand_bit_length - 1) * 2**biased_exponent
        )

    def __init__(
        self,
        value: (
            None
            | int
            | float
            | tuple[int, int]
            | tuple[PrimeFieldElement, PrimeFieldElement]
            | tuple[SecureInteger, SecureInteger]
        ) = None,
        *,
        two_to_exponent: None | PrimeFieldElement | SecureInteger = None,
        is_zero: None | PrimeFieldElement | SecureInteger = None,
    ) -> None:
        """
        Initialize a secure floating-point number.

        :param value: If value is None, this object is a dummy. The values for the significand and exponent and assumed
            to be set at a later time (e.g., by the input function or under the hood by the extended_coroutine
            decorator. Otherwise, it is assumed to be either an integer, a float or a 2-tuple of values representing the
            significand and exponent.
            If the given values in the tuple are already finite field elements or secure integers, they will be used
            as is.
            Otherwise they are assumed to be regular integers and their representation will be corrected before they
            are converted to finite field elements.
        :param two_to_exponent: Exponentiation base two of the exponent of the provided floating point. This will be computed
            automatically if the provided argument to `value` is a (tuple of) number(s). If not provided, the value will
            be computed on-the-fly only when it is required by certain computations.
        :param is_zero: Flag that signals whether the provided floating point has value zero. This will be set
            automatically if the provided argument to `value` is a (tuple of) number(s). If not provided, the value will
            be computed on-the-fly only when it is required by certain computations.
        :raises TypeError: Provided argument for `value` is not of any of the expected types.
        :raises ValueError: The required exponent to represent the provided float argument to `value` is too large.
        """
        # The comment below ignores no-member errors in pylint. Pylint complains, because members are defined in
        # __slots__ for efficiency, but pylint does not understand that.
        # pylint: disable=no-member

        super().__init__((None, None, None))

        # Ensure that the names of the CachedMPyCAttributes agree with the respective argument
        # names to this initializer prepended with an underscore.
        self._is_zero = CachedMPyCAttribute(
            self.significand_class, lambda: self.significand == 0, runtime=flp.mpc
        )
        self._two_to_exponent = CachedMPyCAttribute(
            self.two_to_exponent_class,
            lambda: basic_functions.exp2_int(
                flp.mpc.convert(self.exponent, self.two_to_exponent_class)
            ),
            runtime=flp.mpc,
        )

        if value is None:
            self.share = (self.significand_class(), self.exponent_class())
            return

        # Turn numerical value into a tuple
        if isinstance(value, (int, float)):
            value = self.find_floating_point_representation(
                value, self.significand_bit_length
            )

        if not isinstance(value, tuple):
            raise TypeError(
                f"Expected input to be None, int, float or tuple, but received {type(value)}"
            )
        if len(value) != 2:
            raise TypeError(f"Expected a tuple of length 2, got length {len(value)}")
        if not any(
            all(isinstance(val, typ) for val in value)
            for typ in (int, PrimeFieldElement, SecureInteger)
        ):
            raise TypeError(
                "Expected a tuple with two ints / field elements / secure integers, but received "
                f"types {type(value[0])} and {type(value[1])}."
            )

        if is_zero is not None and not isinstance(is_zero, type(value[0])):
            raise TypeError(
                "Expected the provided is_zero to be None or of the same type as value, but "
                f"received types {type(is_zero)} and {type(value[0])}."
            )
        if two_to_exponent is not None and not isinstance(
            two_to_exponent,
            (self.two_to_exponent_class, self.two_to_exponent_class.share),  # type: ignore[arg-type]
        ):
            raise TypeError(
                "Expected the provided two_to_exponent to be None or of type "
                f"{type(self.two_to_exponent_class)} or {type(self.two_to_exponent_class.share)}, but "
                f"received type {type(two_to_exponent)}."
            )

        extracted_sign, extracted_exp = value
        if isinstance(value[0], int):
            (
                converted_sign,
                converted_exp,
            ) = self.correct_floating_point_representation(
                extracted_sign, extracted_exp, self.significand_bit_length  # type: ignore[arg-type]
            )
            if (exp_bit_len := converted_exp.bit_length()) > self.exponent_bit_length:
                raise ValueError(
                    f"With significand bit length {self.significand_bit_length}, "
                    f"the exponent needs at least bit length {exp_bit_len}, "
                    f"but the specified bit length for the "
                    f"exponent is {self.exponent_bit_length} for provided value {value}."
                )
            self.share = (
                convert_to_stype(converted_sign, self.significand_class, "significand"),
                convert_to_stype(converted_exp, self.exponent_class, "exponent"),
            )
            self._is_zero.set(converted_sign == 0, attr_name="is_zero")
            self._two_to_exponent.set(
                calculate_two_to_exponent_mod(
                    converted_exp, self.two_to_exponent_class.field.modulus
                ),
                attr_name="two_to_exponent",
            )
        else:
            self.share = (
                convert_to_stype(extracted_sign, self.significand_class, "significand"),
                convert_to_stype(extracted_exp, self.exponent_class, "exponent"),
            )
            self._is_zero.set(is_zero, attr_name="is_zero")
            self._two_to_exponent.set(two_to_exponent, attr_name="two_to_exponent")

    def _coerce(self, other: SecureFloatingPoint | int | float) -> SecureFloatingPoint:
        """
        Convert a SecureFloatingPoint integer or float to a compatible type.

        :param other: secure floating-point number, integer or float to be checked and/or converted.
        :return: A compatible SecureFloatingPoint object.
        :raises TypeError: If the other object is a secure floating-point number that is not compatible
        :raises NotImplementedError: if the other object is not of any of the expected types
        """
        if isinstance(other, SecureFloatingPoint):
            if not isinstance(self, type(other)):
                raise TypeError(
                    "The other secure floating point number is not compatible with this one."
                )
            return other
        if isinstance(other, (int, float)):
            return type(self)(other)
        raise NotImplementedError("Expected an int, float, or SecureFloatingPoint")

    @staticmethod
    def correct_floating_point_representation(
        significand: int, exponent: int, significand_bit_length: int
    ) -> tuple[int, int]:
        """
        Change the significand and exponent value such that the significand is in the range [2^(l-1), 2^l - 1), where
        l is the significand bit length.

        :param significand: Integer representing the significand value.
        :param exponent: Integer representing the exponent value.
        :param significand_bit_length: The significand bit length
        :return: A tuple of integers representing the corrected significand and exponent.
        """
        if significand == 0:
            return 0, 0
        sign_bit_len = significand.bit_length()
        new_significand = significand
        new_exponent = exponent
        if sign_bit_len < significand_bit_length - 1:
            bit_diff = significand_bit_length - sign_bit_len
            new_significand = significand << bit_diff
            new_exponent = exponent - bit_diff
        elif sign_bit_len > significand_bit_length:
            bit_diff = sign_bit_len - significand_bit_length
            new_significand = significand >> bit_diff
            new_exponent = exponent + bit_diff
        return new_significand, new_exponent

    @staticmethod
    def find_floating_point_representation(
        value: float, significand_bit_length: int
    ) -> tuple[int, int]:
        r"""
        Function that represents a value as s * 2**e and returns s and e such that s \in [2^(l-1), 2^l - 1), where
        l is the significand bit length.
        :param value: floating-point number to be converted.
        :param significand_bit_length: bit length of the significand.
        :return: A tuple representing the significand and exponent value.
        """
        if value == 0:
            return 0, 0
        if isinstance(value, int):
            exponent_base = value.bit_length() - 1
        elif isinstance(value, float):
            exponent_base = int(value.hex().rsplit("p")[-1])
        else:
            exponent_base = floor(log2(abs(value)))
        exponent = exponent_base - significand_bit_length + 1
        significand = round(value / 2**exponent)
        return significand, exponent

    def sign(self, other: SecureInteger | None = None) -> SecureInteger:
        """
        Function to securely calculate the sign. If no other value is provided, the input value is this object's
        significand, otherwise the other value is used.

        :param other:
        :return:
        """
        if other is None:
            check = self.significand
        else:
            check = other
        return -2 * flp.mpc.sgn(check, LT=True) + 1

    def __abs__(self) -> SecureFloatingPoint:
        """
        Function that securely computes the absolute value.

        :return: A secure floating-point number that represents the absolute value of the object it is called on.
        """

        @extended_coroutine.mpc_coro_extended
        async def _abs() -> SecureFloatingPoint:
            """
            Coroutine that securely computes the absolute value.

            :return: A secure floating-point number that represents the absolute value of the object it is called on.
            """
            stype = type(self)
            await extended_coroutine.returnType(stype)
            new_significand = self.sign() * self.significand
            new_is_zero = self.is_zero() if await self.has_cached_is_zero() else None
            new_two_to_exponent = (
                self.two_to_exponent()
                if await self.has_cached_two_to_exponent()
                else None
            )
            return stype(
                (new_significand, self.exponent),
                two_to_exponent=new_two_to_exponent,
                is_zero=new_is_zero,
            )

        return _abs()

    def __neg__(self) -> SecureFloatingPoint:
        """
        Function that securely computes the negative value.

        :return: A secure floating-point number that represents the additive inverse of the object it is called on.
        """

        @extended_coroutine.mpc_coro_extended
        async def neg() -> SecureFloatingPoint:
            """
            Coroutine that securely computes the negative value.

            :return: A secure floating-point number that represents the additive inverse of the object it is called on.
            """
            stype = type(self)
            await extended_coroutine.returnType(stype)
            neg_sign = -self.significand
            new_is_zero = self.is_zero() if await self.has_cached_is_zero() else None
            new_two_to_exponent = (
                self.two_to_exponent()
                if await self.has_cached_two_to_exponent()
                else None
            )
            return stype(
                (neg_sign, self.exponent),
                two_to_exponent=new_two_to_exponent,
                is_zero=new_is_zero,
            )

        return neg()

    def __pos__(self) -> SecureFloatingPoint:
        """
        Function that returns the object it is called on.

        :return: runtime
        """
        return self

    @staticmethod
    def sum(*values: SecureFloatingPoint | int | float) -> SecureFloatingPoint:
        """
        Function that securely computes multiple addition of a secure floating-point with other
        integers, floats, or secure floating-point numbers.

        Sequentially adding floating-point numbers will yield different results depending on the
        order of addition. This function is roughly equivalent to adding the input values in order
        of increasing modulus, e.g. similar to `sum(sorted(values, key=abs))`.

        :param values: secure floating-point addends, int or float addends. One of them must be a
            secure floating-point.
        :return: A secure floating-point number representing the addition.
        :raises TypeError: Provided list does not contain a SecureFloatingPoint object.
        :raises ValueError: If the number addends is not compatible.
        """
        try:
            secflp_idx, secflp = next(
                (i, val)
                for (i, val) in enumerate(values)
                if isinstance(val, SecureFloatingPoint)
            )
        except StopIteration as exc:
            raise TypeError("No SecureFloatingPoint addends provided") from exc
        return secflp.add(*values[:secflp_idx], *values[secflp_idx + 1 :])

    def add(self, *values: SecureFloatingPoint | int | float) -> SecureFloatingPoint:
        r"""
        Function that securely computes multiple addition of a secure floating-point with other
        integers, floats, or secure floating-point numbers.

        Sequentially adding floating-point numbers will yield different results depending on the
        order of addition. This function is roughly equivalent to adding the input values in order
        of increasing modulus, e.g. similar to `sum(sorted(values, key=abs))`.

        For the computation of `two_to_max_exp` we distinguish three different cases which influence
        whether the resulting output has its `two_to_exponent` set in the protocol:

        * All input values have a cached two_to_exponent.
          In this case we can efficiently compute `two_to_max_exp`. We can compute `inv_two_to_emax`
          via a secure multiplicative inverse and are able to efficiently compute `two_to_exponent`
          for the resulting output.
        * None of the input values has a cached `two_to_exponent`.
          In this case we can not efficiently compute `two_to_max_exp`, as this would require to
          perform a secure exponentiation for each input value. Using a neat trick the need of
          `two_to_max_exp` can be circumvented when computing $2^{e_i - e_{max}}$ by evaluating for
          each input a specific polynomial. Using Horner's method this can be done using just
          secure multiplications. In this case `two_to_exponent` is not set for the resulting output.
        * Some input values have a cached `two_to_exponent` and some input values do not.
          In this case we compute `two_to_max_exp ` from `max_exp`, requiring one secure exponentiation.
          We can obtain `inv_two_to_emax` via secure multiplicative inversion. Using the same
          polynomial as before, for inputs that do not have cached `two_to_exponent`, we can compute
          $2^{e_i - e_{max}}$ without computing $2^{e_i}$. From the earlier computed `two_to_max_exp` we
          can efficiently compute `two_to_exponent` for the resulting output.

        :param values: secure floating-point addends.
        :return: A secure floating-point number representing the addition.
        :raises ValueError: If the number of addends is not compatible.
        """

        if len(values) + 1 > self.max_concurrent_additions:
            raise ValueError(
                f"Number of addends ({len(values) + 1}) is incompatible. "
                f"Maximum allowed addends is {self.max_concurrent_additions}"
            )

        sec_flps = [self] + [self._coerce(val) for val in values]

        for sec_flp in sec_flps[1:]:
            assert sec_flp.significand_class == self.significand_class
            assert sec_flp.significand_bit_length == self.significand_bit_length
            assert sec_flp.exponent_class == self.exponent_class
            assert sec_flp.exponent_bit_length == self.exponent_bit_length
            assert sec_flp.two_to_exponent_class == self.two_to_exponent_class
            assert sec_flp.max_concurrent_additions == self.max_concurrent_additions
            assert (
                sec_flp.max_concurrent_multiplications
                == self.max_concurrent_multiplications
            )

        @extended_coroutine.mpc_coro_extended
        async def add() -> SecureFloatingPoint:
            """
            Coroutine that securely computes the addition of an arbitrary number of elements.

            :return: A secure floating-point number representing the addition.
            """
            stype = type(self)
            await extended_coroutine.returnType(stype)

            num_inputs_with_cached_exp = sum(
                [await sec_flp.has_cached_two_to_exponent() for sec_flp in sec_flps]
            )

            addition_strategy: AdditionStrategy
            if num_inputs_with_cached_exp >= len(sec_flps) - 1:
                # Derive two_to_emax from secflp.two_to_exponent, computing at most one non-cached
                # two_to_exponent
                addition_strategy = CacheAllTwoExponentAddition(self)
            elif num_inputs_with_cached_exp == 0:
                # Don't compute two_to_emax
                addition_strategy = IgnoreTwoExponentAddition(self)
            else:
                # Compute two_to_emax
                addition_strategy = ComputeTwoExponentAddition(self)
            return _add(sec_flps, addition_strategy)

        return add()

    def __add__(self, other: SecureFloatingPoint | int | float) -> SecureFloatingPoint:
        """
        Function that securely computes the addition of a secure floating-point number with an integer, float or
        other secure floating-point number.

        :param other: integer, float or secure floating-point number
        :return: A secure floating-point number representing the addition.
        """
        return self.add(other)

    __radd__ = __add__

    def __sub__(self, other: SecureFloatingPoint | int | float) -> SecureFloatingPoint:
        """
        Function that securely computes the subtraction of other from runtime.

        :param other: integer, float or secure floating-point number to be subtracted.
        :return: A secure floating-point number representing the subtraction.
        """
        return self + (-other)

    def __rsub__(self, other: SecureFloatingPoint | int | float) -> SecureFloatingPoint:
        """
        Function that securely computes the reflected subtraction of other from runtime.

        :param other: integer, float or secure floating-point number to be subtracted.
        :return: A secure floating-point number representing the subtraction.
        """
        return (-self) + other

    @staticmethod
    def prod(*values: SecureFloatingPoint | int | float) -> SecureFloatingPoint:
        """
        Function that securely computes multiple multiplications of a secure floating-point with other integers, floats, or
        secure floating-point numbers.

        :param values: secure floating-point, int, float factors. One of them must be a secure floating-point.
        :return: A secure floating-point number representing the multiplication.
        :raises TypeError: Provided list does not contain a SecureFloatingPoint object.
        :raises ValueError: If the number of factors is not compatible.
        """
        try:
            secflp_idx, secflp = next(
                (i, val)
                for (i, val) in enumerate(values)
                if isinstance(val, SecureFloatingPoint)
            )

        except StopIteration as exc:
            raise TypeError("No SecureFloatingPoint factors provided") from exc
        return secflp.mul(*values[:secflp_idx], *values[secflp_idx + 1 :])

    def mul(self, *values: SecureFloatingPoint | int | float) -> SecureFloatingPoint:
        """
        Function that securely computes multiple multiplications of a secure floating-point with other integers, floats, or
        secure floating-point numbers.

        :param values: secure floating-point factors, first element needs to be secflp type.
        :return: A secure floating-point number representing the multiplication.
        :raises ValueError: If the number of factors is not compatible.
        """
        if len(values) + 1 > self.max_concurrent_multiplications:
            raise ValueError(
                f"Number of factors ({len(values) + 1}) is incompatible. "
                f"Maximum allowed factors is {self.max_concurrent_multiplications}"
            )

        sec_flps = [self] + [self._coerce(val) for val in values]

        for sec_flp in sec_flps[1:]:
            assert sec_flp.significand_class == self.significand_class
            assert sec_flp.significand_bit_length == self.significand_bit_length
            assert sec_flp.exponent_class == self.exponent_class
            assert sec_flp.exponent_bit_length == self.exponent_bit_length
            assert sec_flp.two_to_exponent_class == self.two_to_exponent_class
            assert sec_flp.max_concurrent_additions == self.max_concurrent_additions
            assert (
                sec_flp.max_concurrent_multiplications
                == self.max_concurrent_multiplications
            )

        return _mul(sec_flps)

    def __mul__(self, other: SecureFloatingPoint | int | float) -> SecureFloatingPoint:
        """
        function that securely computes the multiplication of a secure floating-point with an integer, float or
        other secure floating-point number.

        :param other: integer, float or secure floating-point number.
        :return: A secure floating-point number representing the multiplication.
        """
        return self.mul(other)

    __rmul__ = __mul__

    def __truediv__(
        self, other: SecureFloatingPoint | int | float
    ) -> SecureFloatingPoint:
        """
        Function that securely computes the division of a secure floating-point number with an integer, float or
        other secure floating-point number.

        It is the responsibility of the user to ensure that no `ZeroDivisionError` can occur. To prevent such errors,
        the user can explicitly check that the divisor is not zero by performing a secure public zero test on the
        significand.

        :param other: integer, float or secure floating-point number
        :return: A secure floating-point number representing the division.
        """
        coerced_other = self._coerce(other)

        @extended_coroutine.mpc_coro_extended
        async def div() -> SecureFloatingPoint:
            """
            Coroutine that securely computes the division of a secure floating-point number with an integer, float or
            other secure floating-point number.

            :return: A secure floating-point number representing the division.
            """
            # The comment below ignores no-member errors in pylint. Pylint complains, because members are defined in
            # __slots__ for efficiency, but pylint does not understand that.
            # pylint: disable=no-member
            stype = type(self)
            await extended_coroutine.returnType(stype)
            l_var = self.significand_bit_length

            self_sign = self.sign()
            other_sign = coerced_other.sign()
            pos_self_significand = self_sign * self.significand
            pos_other_significand = other_sign * coerced_other.significand

            other_significand_smaller = (
                coerced_other.significand * coerced_other.significand
            ) <= (self.significand * self.significand)

            # Positive numerator and denominator ensures rounding towards zero
            numerator = pos_self_significand << l_var
            denominator = (other_significand_smaller + 1) * pos_other_significand

            new_significand = (
                self_sign * other_sign * secret_div(numerator, denominator)
            )
            new_exponent = (
                self.exponent
                - coerced_other.exponent
                - l_var
                + flp.mpc.convert(other_significand_smaller, self.exponent_class)
            )

            is_zero = self.is_zero() if await self.has_cached_is_zero() else None
            return stype((new_significand, new_exponent), is_zero=is_zero)

        return div()

    def __rtruediv__(
        self, other: SecureFloatingPoint | int | float
    ) -> SecureFloatingPoint:
        """
        Function that securely computes the division of a secure floating-point number with an integer, float or
        other secure floating-point number. Reflected arguments.

        :param other: integer, float or secure floating-point number
        :raise ZeroDivionError: Attempted division by zero.
        :return: A secure floating-point number representing the division.
        """
        return other / self

    # MPyC's SecureNumber implements all dundermethods, so we need to "unset" them
    def __mod__(self, other: Any) -> Self:
        """Integer remainder with public divisor."""
        return NotImplemented

    def __rmod__(self, other: Any) -> Self:
        """Integer remainder (with reflected arguments)."""
        return NotImplemented

    def __floordiv__(self, other: Any) -> Self:
        """Integer quotient with public divisor."""
        return NotImplemented

    def __rfloordiv__(self, other: Any) -> Self:
        """Integer quotient (with reflected arguments)."""
        return NotImplemented

    def __divmod__(self, other: Any) -> tuple[Self, Self]:
        """Integer division with public divisor."""
        return NotImplemented

    def __rdivmod__(self, other: Any) -> tuple[Self, Self]:
        """Integer division (with reflected arguments)."""
        return NotImplemented

    def __pow__(self, other: Any) -> Self:
        """Exponentation for public integral exponent."""
        return NotImplemented

    def __rpow__(self, other: Any) -> Self:
        """Exponentation (with reflected arguments) for secret exponent."""
        return NotImplemented

    def __lshift__(self, other: Any) -> Self:
        """Left shift with public integral offset."""
        return NotImplemented

    def __rlshift__(self, other: Any) -> Self:
        """Left shift (with reflected arguments)."""
        return NotImplemented

    def __rshift__(self, other: Any) -> Self:
        """Right shift with public integral offset."""
        return NotImplemented

    def __rrshift__(self, other: Any) -> Self:
        """Right shift (with reflected arguments)."""
        return NotImplemented

    def __and__(self, other: Any) -> Self:
        """Bitwise and."""
        return NotImplemented

    __rand__ = __and__

    def __xor__(self, other: Any) -> Self:
        """Bitwise exclusive-or."""
        return NotImplemented

    def __rxor__(self, other: Any) -> Self:
        """Bitwise exclusive-or (with reflected arguments)."""
        return NotImplemented

    def __invert__(self) -> Self:
        """Bitwise not (inversion)."""
        return NotImplemented

    def __or__(self, other: Any) -> Self:
        """Bitwise or."""
        return NotImplemented

    __ror__ = __or__

    def __lt__(self, other: Any) -> Self:
        """Strictly less-than comparison."""
        return NotImplemented

    def __le__(self, other: Any) -> Self:
        """Less-than or equal comparison."""
        return NotImplemented

    def __eq__(self, other: Any) -> Self:  # type: ignore[override]
        """Equality testing."""
        return NotImplemented

    def __ge__(self, other: Any) -> Self:
        """Greater-than or equal comparison."""
        return NotImplemented

    def __gt__(self, other: Any) -> Self:
        """Strictly greater-than comparison."""
        return NotImplemented

    def __ne__(self, other: Any) -> Self:  # type: ignore[override]
        """Negated equality testing."""
        return NotImplemented


def SecFlp(  # pylint: disable=invalid-name # This name was chosen to conform to MPyC naming conventions
    significand_bit_length: None | int = None,
    exponent_bit_length: None | int = None,
    significand_prime: None | int = None,
    exponent_prime: None | int = None,
    max_concurrent_additions: int = 2,
    max_concurrent_multiplications: int = 2,
) -> type[SecureFloatingPoint]:
    """
    Function that generates a class for secure floating point numbers with certain bit lengths for
    the significand and the exponent.

    :param significand_bit_length: Bit length of the significand.
    :param exponent_bit_length: Bit length of the exponent.
    :param significand_prime: Prime number used as the modulus for the finite field of the significand.
    :param exponent_prime: Prime number used as the modulus for the finite field of the exponent.
    :param max_concurrent_additions: max number of default secure floating point addends.
    :param max_concurrent_multiplications: max number of secure floating point factors.
    :return: A SecureFloatingPoint type initialised using the parameters given.
    """
    if significand_bit_length is None:
        significand_bit_length = flp.mpc.options.bit_length
    if exponent_bit_length is None:
        exponent_bit_length = flp.mpc.options.bit_length
    return secure_floating_point_helper(
        significand_bit_length=significand_bit_length,
        significand_prime=significand_prime,
        exponent_bit_length=exponent_bit_length,
        exponent_prime=exponent_prime,
        max_concurrent_additions=max_concurrent_additions,
        max_concurrent_multiplications=max_concurrent_multiplications,
    )


@functools.cache
def secure_floating_point_helper(
    significand_bit_length: int,
    exponent_bit_length: int,
    significand_prime: None | int = None,
    exponent_prime: None | int = None,
    max_concurrent_additions: int = 2,
    max_concurrent_multiplications: int = 2,
) -> type[SecureFloatingPoint]:
    """
    Helper function that generates a class for a secure floating point number with the
    given parameters.

    :param significand_bit_length: The bit length of the significand.
    :param exponent_bit_length: The bit length of the exponent.
    :param significand_prime: The prime number used as the modulus for the finite field of the significand.
    :param exponent_prime: The prime number used as the modulus for the finite field of the exponent.
    :param max_concurrent_additions: max number of secure floating point addends.
    :param max_concurrent_multiplications: max number of secure floating point factors.
    :return: A SecureFloatingPoint type initialised using the parameters given.
    """
    tau = int(significand_bit_length + ceil(log2(max_concurrent_additions)))

    final_significand_bit_length = int(
        max(
            2 * tau,
            2 * significand_bit_length + 2 * (max_concurrent_multiplications - 2),
        )
    )

    name = (
        f"SecFlp{significand_bit_length}-{exponent_bit_length}|"
        f"{max_concurrent_additions}-{max_concurrent_multiplications}"
    )
    if significand_prime is not None or exponent_prime is not None:
        suffix = f"({significand_prime or '_'},{exponent_prime or '_'})"
        name += suffix

    secure_flp_class: type[SecureFloatingPoint] = cast(
        type[SecureFloatingPoint],
        types.new_class(
            name,
            (SecureFloatingPoint,),
        ),
    )

    secure_flp_class.significand_class = SecInt(final_significand_bit_length)
    secure_flp_class.significand_bit_length = significand_bit_length

    secure_flp_class.exponent_bit_length = exponent_bit_length
    secure_flp_class.exponent_class = SecInt(exponent_bit_length)

    secure_flp_class.two_to_exponent_class = secure_flp_class.significand_class

    secure_flp_class.max_concurrent_additions = max_concurrent_additions
    secure_flp_class.max_concurrent_multiplications = max_concurrent_multiplications

    return secure_flp_class
