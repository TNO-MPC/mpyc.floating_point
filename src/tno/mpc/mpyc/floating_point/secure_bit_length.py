"""
Module containing a protocol to securely compute the bit length of an MPyC secure integer.
"""

from __future__ import annotations

from typing import Literal, TypeVar, cast, overload

from mpyc.finfields import PrimeFieldElement
from mpyc.sectypes import SecureInteger, SecureNumber

from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point import extended_coroutine

SecureNumberT = TypeVar("SecureNumberT", bound=SecureNumber)


@overload
def bit_length(
    secure_element: SecureInteger,
    *,
    two_to_pow: Literal[False] = ...,
    inv_two_to_pow: Literal[False] = ...,
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> SecureInteger: ...


@overload
def bit_length(
    secure_element: SecureInteger,
    *,
    two_to_pow: Literal[True],
    inv_two_to_pow: Literal[False] = ...,
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> tuple[SecureInteger, SecureInteger]: ...


@overload
def bit_length(
    secure_element: SecureInteger,
    *,
    two_to_pow: Literal[False] = ...,
    inv_two_to_pow: Literal[True],
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> tuple[SecureInteger, SecureInteger]: ...


@overload
def bit_length(
    secure_element: SecureInteger,
    *,
    two_to_pow: Literal[True],
    inv_two_to_pow: Literal[True],
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> tuple[SecureInteger, SecureInteger, SecureInteger]: ...


@overload
def bit_length(
    secure_element: SecureInteger,
    *,
    two_to_pow: bool = ...,
    inv_two_to_pow: bool = ...,
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> (
    SecureInteger
    | tuple[SecureInteger, SecureInteger]
    | tuple[SecureInteger, SecureInteger, SecureInteger]
): ...


def bit_length(
    secure_element: SecureInteger,
    *,
    two_to_pow: bool = False,
    inv_two_to_pow: bool = False,
    lower_bound: int = 0,
    upper_bound: int | None = None,
) -> (
    SecureInteger
    | tuple[SecureInteger, SecureInteger]
    | tuple[SecureInteger, SecureInteger, SecureInteger]
):
    """
    Implementation of Thijs Veugen's secure bit length protocol.

    !! The current implementation only supports non-negative inputs !!

    :param secure_element: secret shared value of which the bit length needs to be determined. The
        secret value is assumed to be non-negative.
    :param two_to_pow: If True, also compute and return 2 ** bit_length. This is more efficient than
        computing the exponentiation afterwards.
    :param inv_two_to_pow: If True, also compute and return 2 ** -bit_length. This is more efficient
        than computing the exponentiation afterwards.
    :param lower_bound: Lower bound on the bit length of the provided value. This parameter allows
        for a more efficient protocol, but it will also yield an incorrect result if the lower
        bound is larger than the actual bit length.
    :param upper_bound: Upper bound on the bit length of the provided value. This parameter allows
        for a more efficient protocol, but it will also yield an incorrect result if the upper
        bound is smaller than the actual bit length.
    :return: A secret-shared value representing the bit length of the input. If two_to_pow=True, the
        second return value is its power base two. If inv_two_to_pow=True, the final return value is
        the inverse of its power base two.
    """
    msb_results = most_significant_bit(
        secure_element,
        two_to_pow=two_to_pow,
        inv_two_to_pow=inv_two_to_pow,
        lower_bound=max(lower_bound - 1, 0),
        upper_bound=max(upper_bound - 1, 0) if upper_bound else None,
    )

    if not isinstance(msb_results, tuple):
        # two_to_pow is inv_two_to_pow is False
        return msb_results + 1

    # msb_results is a tuple of length 2 or 3
    length = msb_results[0] + 1
    if two_to_pow:
        two_to_length = 2 * msb_results[1]
    if inv_two_to_pow:
        stype = type(secure_element)
        inv_two = (stype.field.modulus + 1) // 2
        inv_two_to_length = inv_two * msb_results[-1]

    if two_to_pow and inv_two_to_pow:
        return length, two_to_length, inv_two_to_length
    if two_to_pow:
        return length, two_to_length
    return length, inv_two_to_length


@overload
def most_significant_bit(
    secure_element: SecureInteger,
    *,
    two_to_pow: Literal[False] = ...,
    inv_two_to_pow: Literal[False] = ...,
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> SecureInteger: ...


@overload
def most_significant_bit(
    secure_element: SecureInteger,
    *,
    two_to_pow: Literal[True],
    inv_two_to_pow: Literal[False] = ...,
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> tuple[SecureInteger, SecureInteger]: ...


@overload
def most_significant_bit(
    secure_element: SecureInteger,
    *,
    two_to_pow: Literal[False] = ...,
    inv_two_to_pow: Literal[True],
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> tuple[SecureInteger, SecureInteger]: ...


@overload
def most_significant_bit(
    secure_element: SecureInteger,
    *,
    two_to_pow: Literal[True],
    inv_two_to_pow: Literal[True],
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> tuple[SecureInteger, SecureInteger, SecureInteger]: ...


@overload
def most_significant_bit(
    secure_element: SecureInteger,
    *,
    two_to_pow: bool = ...,
    inv_two_to_pow: bool = ...,
    lower_bound: int = ...,
    upper_bound: int | None = ...,
) -> (
    SecureInteger
    | tuple[SecureInteger, SecureInteger]
    | tuple[SecureInteger, SecureInteger, SecureInteger]
): ...


@extended_coroutine.mpc_coro_extended
async def most_significant_bit(
    secure_element: SecureInteger,
    *,
    two_to_pow: bool = False,
    inv_two_to_pow: bool = False,
    lower_bound: int = 0,
    upper_bound: int | None = None,
) -> (
    SecureInteger
    | tuple[SecureInteger, SecureInteger]
    | tuple[SecureInteger, SecureInteger, SecureInteger]
):
    """
    Implementation of Thijs Veugen's secure most significant bit (MSB) protocol.

    The function will return the index of the most significant bit. Thus, MSB(1) = 0, MSB(2) = 1,
    MSB(3) = 1, etc. The edge case MSB(0) returns -1. For consistency, 2 ** -MSB(0) returns 2 and
    2 ** MSB(0) returns the multiplicative inverse of 2 in the finite field.

    :param secure_element: Secret shared value of which the most significant bit needs to be determined.
    :param two_to_pow: If True, also compute and return 2 ** MSB. This is more efficient than
        computing the exponentiation afterwards.
    :param inv_two_to_pow: If True, also compute and return 2 ** -MSB. This is more efficient than
        computing the exponentiation afterwards.
    :param lower_bound: Lower bound on the most significant bit index of the provided value. This
        parameter allows for a more efficient protocol, but it will also yield an incorrect result if
        the lower bound is larger than the actual MSB index.
    :param upper_bound: Upper bound on the most significant bit index of the provided value. This
        parameter allows for a more efficient protocol, but it will also yield an incorrect result if
        the upper bound is smaller than the actual MSB index.
    :raise ValueError: The provided bounds are unacceptable.
    :return: The index of the most significant bit of input secure_element. If two_to_pow=True, the
        second return value is its power base two. If inv_two_to_pow=True, the final return value is
        the inverse of its power base two.
    """
    stype = type(secure_element)
    if two_to_pow and inv_two_to_pow:
        await extended_coroutine.returnType((stype, stype, stype))
    elif two_to_pow or inv_two_to_pow:
        await extended_coroutine.returnType((stype, stype))
    else:
        await extended_coroutine.returnType(stype)

    field: type[PrimeFieldElement] = stype.field
    max_bitlength = cast(int, stype.bit_length)
    # The upper bound m in Veugen's protocol should satisfy secure_element < 2 ** m.
    # Given that the _index_ of the MSB is at most upper_bound, we find that m = upper_bound + 1 is
    # guaranteed to satisfy the above constraint.
    if upper_bound is None:
        m = max_bitlength
    else:
        m = min(upper_bound + 1, max_bitlength)

    if lower_bound < 0:
        raise ValueError(
            f"Expected non-negative lower bound, but received {lower_bound=}."
        )
    if upper_bound is not None and lower_bound > upper_bound:
        raise ValueError(
            f"Expected lower_bound <= upper_bound, but received {lower_bound=} and {upper_bound=}."
        )
    n_bits_to_check = m - lower_bound + 1

    # Steps 1+2: Generate random bits and compute r
    r_bits = await flp.mpc.random_bits(field, max_bitlength + 1)
    r_modl = field(0)
    for r_i in reversed(r_bits):
        r_modl <<= 1
        r_modl += r_i.value
    r_divl = flp.mpc._random(field, 1 << flp.mpc.options.sec_param).value
    # Note that r has max_bitlength + sec_param + 1 bits
    r = (r_divl << len(r_bits)) + r_modl

    # Step 3: Compute and reveal c = x + r.
    secure_element_share = await flp.mpc.gather(secure_element)
    c = await flp.mpc.output(secure_element_share + r)
    c_share = c.value
    c_bits = [(c_share >> lower_bound + i) & 1 for i in range(n_bits_to_check)]

    # Step 4: Locally compute the sequence c⊕r
    r_bits_in_check_range: list[SecureInteger] = [
        stype(r.value) for r in r_bits[lower_bound : m + 1]
    ]
    d_bits = [
        1 - sec_r_bit if c_bit else sec_r_bit
        for (c_bit, sec_r_bit) in zip(c_bits, r_bits_in_check_range)
    ]

    # Step 5: Trace back the carry-over in one round
    h_bits = flp.mpc.schur_prod(
        [
            1 - sec_r_bit
            for (c_bit, sec_r_bit) in zip(c_bits[:-1], r_bits_in_check_range[:-1])
            if not c_bit
        ],
        [d_bit for (c_bit, d_bit) in zip(c_bits[:-1], d_bits[1:]) if not c_bit],
    )
    for i in reversed(range(n_bits_to_check - 1)):
        if not c_bits[i]:
            d_bits[i + 1] = h_bits.pop()

    # Step 6: Compute the prefix-or p of c⊕r
    y = [(1 + el) for el in d_bits]
    z_rev = [y[-1]]
    for y_i in reversed(y[:-1]):
        z_rev.append(z_rev[-1] * y_i)

    # We call flp.mpc.lsb(z_i) rather than z_i % 2. Under the hood, _ % 2 already calls mpc.lsb.
    # However, somehow we run into synchronization issues (multiplayer) if we depend on mpc.mod.
    # This likely has to do with the decorator (changing it into the pc-incrementing variant,
    # works) but we have yet to find a MWE to report this to MPyC (or fix it on our side).
    p = [1 - flp.mpc.lsb(z_i) for z_i in reversed(z_rev)]

    # Step 7: Compute k and 2**k
    k = flp.mpc.sum(p) + lower_bound - 1

    two_to_k: SecureInteger
    if lower_bound == 0:
        two_to_k = p[0] + flp.mpc.sum([2**i * p_i for i, p_i in enumerate(p[1:])])
    else:
        two_to_k = 2 ** (lower_bound - 1) + flp.mpc.sum(
            [2 ** (lower_bound - 1 + i) * p_i for i, p_i in enumerate(p)]
        )

    # Step 8: Compute secure comparison δ = a < 2**k
    delta: SecureInteger = secure_element_share < two_to_k  # type: ignore[operator]

    # Step 9: Compute l = (k - delta) and 2**l = 2**(k - 1) * (2 - delta)
    l = k - delta

    inv_two = (stype.field.modulus + 1) // 2
    two_to_l = two_to_k * inv_two * (2 - delta)
    # If the input is 0, equivalently p[0] == 0, then two_to_l is set to 2 ** -1 rather than 0. This is corrected below.
    two_to_l = two_to_l + (1 - p[0]) * inv_two

    if not (two_to_pow or inv_two_to_pow):
        return l
    if not inv_two_to_pow:
        return l, two_to_l

    # inv_two_to_pow is True
    # Step 7*: Compute 2**-k
    inv_two_to_k: SecureInteger
    if lower_bound == 0:
        inv_two_to_k = p[0] - flp.mpc.sum(
            [inv_two**i * p_i for i, p_i in enumerate(p[1:], start=1)]
        )
    else:
        inv_two_to_k = inv_two ** (lower_bound - 1) - flp.mpc.sum(
            [inv_two ** (lower_bound + i) * p_i for i, p_i in enumerate(p)]
        )
    # Step 9*: compute 2**-l = 2**-k * (delta + 1)
    inv_two_to_l = inv_two_to_k * (delta + 1)
    # If the input is 0, equivalently p[0] == 0, then inv_two_to_l is set to 2 rather than 0. This is corrected below.
    inv_two_to_l = inv_two_to_l + (1 - p[0]) * 2
    if not two_to_pow:
        return l, inv_two_to_l
    return l, two_to_l, inv_two_to_l
