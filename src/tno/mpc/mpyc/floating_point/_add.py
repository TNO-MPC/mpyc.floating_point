"""
Module that provides secure (multiple) addition for secure floating points.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil, log2
from typing import TYPE_CHECKING, cast

from mpyc.gmpy import invert
from mpyc.seclists import seclist
from mpyc.sectypes import SecureInteger

from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point import basic_functions, extended_coroutine
from tno.mpc.mpyc.floating_point._plaintext_helper_functions import lagrange_int_poly
from tno.mpc.mpyc.floating_point.secure_bit_length import bit_length
from tno.mpc.mpyc.floating_point.vendor.mpyc._trunc import trunc

if TYPE_CHECKING:
    from tno.mpc.mpyc.floating_point.secure_floating_point import SecureFloatingPoint


class AdditionStrategy(ABC):
    """
    Strategy that describes the parts of the addition protocol that depend on caching
    specifics of the addends.
    """

    tau: int
    two_to_emax: SecureInteger | None
    inv_two_to_emax: SecureInteger
    inv_two_to_tau: SecureInteger

    def __init__(self, sec_flp: SecureFloatingPoint) -> None:
        """
        Initialize Addition strategy.

        :param sec_flp: Secure floating point.
        """
        self._extract_secflp_parameters(sec_flp)

    def _extract_secflp_parameters(self, sec_flp: SecureFloatingPoint) -> None:
        """
        Store useful parameters from the secure floating point.

        :param sec_flp: Secure floating point.
        """
        self.tau = int(
            sec_flp.significand_bit_length
            + ceil(log2(sec_flp.max_concurrent_additions))
        )
        modulus = sec_flp.two_to_exponent_class.field.modulus
        self.inv_two_to_tau = sec_flp.two_to_exponent_class(
            int(invert(pow(2, self.tau, modulus), modulus))
        )
        self.inv_two_to_significand_bit_length = sec_flp.two_to_exponent_class(
            int(invert(pow(2, sec_flp.significand_bit_length, modulus), modulus))
        )

    @abstractmethod
    def compute_max_exp(
        self,
        sec_flps: list[SecureFloatingPoint],
        adjusted_exponents: list[SecureInteger],
    ) -> SecureInteger:
        """
        Compute the maximum exponent.

        :param sec_flps: To be added secure floating points
        :param adjusted_exponents: Adjusted exponents.
        :return: Maximum over the given exponents.
        """

    @abstractmethod
    def compute_bit_length(
        self,
        significand: SecureInteger,
    ) -> tuple[SecureInteger, SecureInteger]:
        """
        Compute the bit length of the significand.

        :param significand: Significand.
        :return: Tuple of bit length b of the significand and 2 ** -b.
        """

    @abstractmethod
    def compute_two_to_exp(self) -> SecureInteger | None:
        """
        Compute the power of the final exponent base two.

        :return: 2 ** exp or None if this value cannot be computed easily.
        """


class CacheAllTwoExponentAddition(AdditionStrategy):
    r"""
    Addition strategy that caches all `two_to_exponents` and uses that for efficient further
    computations.
    """

    def __init__(self, secflp: SecureFloatingPoint) -> None:
        """
        Initialize `CacheAllTwoExponentAddition` strategy.
        """
        super().__init__(secflp)
        self._two_to_n_bits_s: SecureInteger | None = None

    def compute_max_exp(
        self,
        sec_flps: list[SecureFloatingPoint],
        adjusted_exponents: list[SecureInteger],
    ) -> SecureInteger:
        r"""
        Compute the maximum exponent.

        Additionally caches all `two_to_exponents` and computes `2 ** max_exponent`.
        Additionally computes 2 ** - max_exponent.

        :param sec_flps: To be added secure floating points
        :param adjusted_exponents: Adjusted exponents.
        :return: Maximum over the given exponents.
        """
        max_idx, max_adj_exp = flp.mpc.argmax(adjusted_exponents)
        two_to_exponents = [val.two_to_exponent() for val in sec_flps]
        self.two_to_emax: SecureInteger = seclist(two_to_exponents)[  # type: ignore[no-untyped-call]
            flp.mpc.convert(max_idx, type(two_to_exponents[0]))
        ]
        self.inv_two_to_emax = flp.mpc.reciprocal(self.two_to_emax)
        return max_adj_exp

    def compute_bit_length(
        self, significand: SecureInteger
    ) -> tuple[SecureInteger, SecureInteger]:
        """
        Compute the bit length of the significand.

        Additionally computes and stores 2 ** b.

        :param significand: Significand.
        :return: Tuple of bit length b of the significand and 2 ** -b.
        """
        n_bits_s, two_to_n_bits_s, inv_two_to_n_bits_s = bit_length(
            flp.mpc.abs(significand),
            two_to_pow=True,
            inv_two_to_pow=True,
            upper_bound=2 * self.tau,
        )
        self._two_to_n_bits_s = two_to_n_bits_s
        return n_bits_s, inv_two_to_n_bits_s

    def compute_two_to_exp(self) -> SecureInteger:
        """
        Compute the power of the final exponent base two.

        :return: 2 ** exp
        """
        return self._compute_two_to_exp(
            self.two_to_emax,
            self.inv_two_to_tau,
            self.inv_two_to_significand_bit_length,
        )

    def _compute_two_to_exp(
        self,
        two_to_emax: SecureInteger,
        inv_two_to_tau: SecureInteger,
        inv_two_to_significand_bit_length: SecureInteger,
    ) -> SecureInteger:
        """
        Compute the power of the final exponent base two.

        :param two_to_emax: Power of max_exp base two.
        :param inv_two_to_tau: Multiplicative inverse of power of tau base two.
        :param inv_two_to_significand_bit_length: Multiplicative inverse of power of significand bit length base two.
        :return: 2 ** exp
        """
        two_to_exponent_class = type(two_to_emax)
        two_to_exp = (
            two_to_emax
            * cast(SecureInteger, self._two_to_n_bits_s)
            * inv_two_to_tau
            * inv_two_to_significand_bit_length
        )
        return flp.mpc.convert(two_to_exp, two_to_exponent_class)


class ComputeTwoExponentAddition(AdditionStrategy):
    r"""
    Addition strategy that computes `two_to_emax` from `max_exp` and uses that to compute
    the final `two_to_exponent`.
    """

    def __init__(self, secflp: SecureFloatingPoint) -> None:
        """
        Initialize `ComputeTwoExponentAddition` strategy.

        :param sec_flp: Secure floating point.
        """
        super().__init__(secflp)
        self._two_to_n_bits_s: SecureInteger | None = None
        self._cache_all_two_to_exponent_addition = CacheAllTwoExponentAddition(secflp)

    def compute_max_exp(
        self,
        sec_flps: list[SecureFloatingPoint],
        adjusted_exponents: list[SecureInteger],
    ) -> SecureInteger:
        """
        Compute the maximum exponent.

        Additionally computes 2 ** max_exponent.

        :param sec_flps: To be added secure floating points
        :param adjusted_exponents: Adjusted exponents.
        :return: Maximum over the given exponents.
        """
        max_adj_exp = flp.mpc.max(adjusted_exponents)
        self.two_to_emax = basic_functions.exp2_int(
            flp.mpc.convert(max_adj_exp, sec_flps[0].two_to_exponent_class)
        )
        self.inv_two_to_emax = flp.mpc.reciprocal(self.two_to_emax)
        return max_adj_exp

    def compute_bit_length(
        self,
        significand: SecureInteger,
    ) -> tuple[SecureInteger, SecureInteger]:
        """
        Compute the bit length of the significand.

        Additionally computes and stores 2 ** b.

        :param significand: Significand.
        :return: Tuple of bit length b of the significand and 2 ** -b.
        """
        return self._cache_all_two_to_exponent_addition.compute_bit_length(
            significand,
        )

    def compute_two_to_exp(self) -> SecureInteger:
        """
        Compute the power of the final exponent base two.

        :return: 2 ** exp
        """
        return self._cache_all_two_to_exponent_addition._compute_two_to_exp(
            cast(SecureInteger, self.two_to_emax),
            self.inv_two_to_tau,
            self.inv_two_to_significand_bit_length,
        )


class IgnoreTwoExponentAddition(AdditionStrategy):
    r"""
    Addition strategy that does not compute the final `two_to_exponent`.
    """

    @property
    def two_to_emax(self) -> None:  # type: ignore[override]
        """
        Getter for two_to_emax.

        :raise ValueError: always.
        """
        raise ValueError(f"This value should not be requested for {type(self)}.")

    @property
    def inv_two_to_emax(self) -> None:  # type: ignore[override]
        """
        Getter for inv_two_to_emax.

        :raise ValueError: always.
        """
        raise ValueError(f"This value should not be requested for {type(self)}.")

    def compute_max_exp(
        self,
        sec_flps: list[SecureFloatingPoint],
        adjusted_exponents: list[SecureInteger],
    ) -> SecureInteger:
        """
        Compute the maximum exponent.

        :param sec_flps: param not used, present here for api reasons.
        :param adjusted_exponents: Adjusted exponents.
        :return: Maximum over the given exponents.
        """
        del sec_flps
        return flp.mpc.max(adjusted_exponents)

    def compute_bit_length(
        self, significand: SecureInteger
    ) -> tuple[SecureInteger, SecureInteger]:
        """
        Compute the bit length of the significand.

        :param significand: Significand.
        :return: Tuple of bit length b of the significand and 2 ** -b.
        """
        n_bits_s, inv_two_to_n_bits_s = bit_length(
            flp.mpc.abs(significand),
            two_to_pow=False,
            inv_two_to_pow=True,
            upper_bound=2 * self.tau,
        )
        return n_bits_s, inv_two_to_n_bits_s

    def compute_two_to_exp(self) -> None:
        """
        Returns None since the two exponent is too expensive to compute.
        """
        return None


@extended_coroutine.mpc_coro_extended
async def _add(
    sec_flps: list[SecureFloatingPoint],
    addition_strategy: AdditionStrategy,
) -> SecureFloatingPoint:
    """
    Coroutine that securely computes the addition of an arbitrary number of elements.

    :param sec_flps: The to be added secure floating points.
    :param addition_strategy: Strategy that performs part of the addition protocol that is
        specific to the number of addends with cached two_to_exponent.
    :return: A secure floating-point number representing the addition.
    """
    # The comment below ignores no-member errors in pylint. Pylint complains, because members are defined in
    # __slots__ for efficiency, but pylint does not understand that.
    # pylint: disable=no-member
    stype = type(sec_flps[0])
    await extended_coroutine.returnType(stype)

    exponent_bit_length = sec_flps[0].exponent_bit_length
    exponent_class = sec_flps[0].exponent_class
    significand_class = sec_flps[0].significand_class
    significand_bit_length = sec_flps[0].significand_bit_length
    tau = addition_strategy.tau

    # Step 1: Calculate largest exponent, where a floating point representation of zero has the lowest possible exponent
    min_exponent = -(2 ** (exponent_bit_length - 1))
    adjusted_exponents = [
        val.exponent
        + flp.mpc.convert(val.is_zero(), exponent_class) * (min_exponent - val.exponent)
        for val in sec_flps
    ]
    max_adj_exp = addition_strategy.compute_max_exp(sec_flps, adjusted_exponents)

    # Step 2: Calculate delta_i = e_i > e_max - tau
    delta = [adj_exp >= (max_adj_exp - tau) for adj_exp in adjusted_exponents]
    delta = [flp.mpc.convert(delta_i, significand_class) for delta_i in delta]

    # Step 3: Calculate significand
    two_to_tau = 2**tau
    inv_two = (significand_class.field.modulus + 1) // 2
    coefficients = lagrange_int_poly(
        values=tuple(inv_two**i for i in range(tau + 1)),
        prime=significand_class.field.modulus,
    )

    significand = significand_class(0)
    for sec_flp, delta_i in zip(sec_flps, delta):
        if await sec_flp.has_cached_two_to_exponent():
            significand += (
                sec_flp.significand
                * delta_i
                * sec_flp.two_to_exponent()
                * addition_strategy.inv_two_to_emax
                * two_to_tau
            )

        else:  # 2**(e_i - e_max) = q(e_max - e_i)
            x = flp.mpc.convert(max_adj_exp - sec_flp.exponent, significand_class)
            q_value = basic_functions.poly_approximation(coefficients, x)
            significand += sec_flp.significand * delta_i * q_value * two_to_tau

    # Step 4: Compute bit length n_bits_s of |s| and 2**-n_bits_s
    n_bits_s, inv_two_to_n_bits_s = addition_strategy.compute_bit_length(significand)
    n_bits_s = flp.mpc.convert(n_bits_s, exponent_class)

    # Step 5: Divide s' by 2**(n_bits_s−ℓ)
    significand *= inv_two_to_n_bits_s * two_to_tau * two_to_tau
    significand = trunc(significand, 2 * tau - significand_bit_length)

    # Step 6: Set exponent
    exponent = max_adj_exp + n_bits_s - tau - significand_bit_length
    return stype(
        (significand, exponent),
        two_to_exponent=addition_strategy.compute_two_to_exp(),
    )
