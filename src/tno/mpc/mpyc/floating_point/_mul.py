"""
Module that provides secure (multiple) multiplications for secure floating points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mpyc.sectypes import SecureInteger

from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point import extended_coroutine
from tno.mpc.mpyc.floating_point.secure_bit_length import most_significant_bit
from tno.mpc.mpyc.floating_point.vendor.mpyc._trunc import trunc

if TYPE_CHECKING:
    from tno.mpc.mpyc.floating_point.secure_floating_point import SecureFloatingPoint


@extended_coroutine.mpc_coro_extended
async def _mul(
    sec_flps: list[SecureFloatingPoint],
) -> SecureFloatingPoint:
    """
    Coroutine that securely computes the multiplication of an arbitrary number of elements.

    :param sec_flps: The to be added secure floating points.
    :return: A secure floating-point number representing the multiplication.
    """
    stype = type(sec_flps[0])
    await extended_coroutine.returnType(stype)

    exponent_class = sec_flps[0].exponent_class
    significand_bit_length = sec_flps[0].significand_bit_length

    # Step 1: Initialization
    significand = sec_flps[0].significand

    # Step 2: Significand multiplication
    for sec_val in sec_flps[1:]:
        significand *= sec_val.significand
        significand = trunc(significand, significand_bit_length - 1)

    # Step 3: Compute msb m of |s| and 2**-m
    m, two_to_m, inv_two_to_m = most_significant_bit(
        flp.mpc.abs(significand),
        two_to_pow=True,
        inv_two_to_pow=True,
        lower_bound=stype.significand_bit_length - 1,
        upper_bound=stype.significand_bit_length + len(sec_flps),
    )
    m = flp.mpc.convert(m, exponent_class)

    # Step 4: Scaling the significand
    significand <<= significand_bit_length + len(sec_flps) - 1
    significand *= inv_two_to_m
    significand = trunc(significand, len(sec_flps))

    # Step 5: Computing the exponent
    exponent = exponent_class(0)
    for sec_flp in sec_flps:
        exponent += sec_flp.exponent
    exponent += (len(sec_flps) - 2) * (significand_bit_length - 1) + m

    is_zero: SecureInteger | None = None
    if all([await sec_flp.has_cached_is_zero() for sec_flp in sec_flps]):
        is_zero = 1 - flp.mpc.prod([1 - sec_flp.is_zero() for sec_flp in sec_flps])

    two_to_exponent: SecureInteger | None = None
    if all([await sec_flp.has_cached_two_to_exponent() for sec_flp in sec_flps]):
        two_to_exponent = (
            flp.mpc.prod([sec_flp.two_to_exponent() for sec_flp in sec_flps])
            * 2
            ** (
                (significand_bit_length - 1) * (len(sec_flps) - 1)
                - significand_bit_length
                + 1
            )
            * two_to_m
        )

    return stype(
        (significand, exponent),
        two_to_exponent=two_to_exponent,
        is_zero=is_zero,
    )
