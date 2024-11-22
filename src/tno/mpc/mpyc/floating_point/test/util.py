"""
Module with utility functionality for the unit tests for Secure Floating Point Numbers.
"""

from __future__ import annotations

import sys
from math import floor, log2
from typing import TYPE_CHECKING

from mpyc.gmpy import invert

from tno.mpc.mpyc.floating_point import _rt as flp

if TYPE_CHECKING:
    from tno.mpc.mpyc.floating_point.secure_floating_point import SecureFloatingPoint

# From the sys.float_info.max_exp documentation: The maximum integer e such that radix**(e-1) is a representable finite float.
MAX_EXPONENT = sys.float_info.max_exp - 1
# Similar for sys.float_info.min_exp
MIN_EXPONENT = sys.float_info.min_exp - 1

MAX_POSITIVE_DOUBLE = sys.float_info.max
MIN_POSITIVE_DOUBLE = sys.float_info.min
MAX_NEGATIVE_DOUBLE = -MIN_POSITIVE_DOUBLE
MIN_NEGATIVE_DOUBLE = -MAX_POSITIVE_DOUBLE


async def assert_significand_in_expected_range(secflp: SecureFloatingPoint) -> None:
    r"""
    Assert that the significand of the provided value is in the expected range.

    :param secflp: Secure floating point value to be validated.
    :raise AssertionError: Modulus of non-zero significand is not in the expected range $[2^{l-1}, 2^l)$.
    """
    significand = int(await flp.mpc.output(secflp.significand))
    if significand == 0:
        return
    nr_bits = log2(abs(significand))
    assert (
        floor(nr_bits) == secflp.significand_bit_length - 1
    ), f"""\
Modulus of non-zero significand is not in expected range \
[2 ** {secflp.significand_bit_length-1}, 2 ** {secflp.significand_bit_length}). We found \
log2(modulus)={significand:.2f}.
"""


async def _assert_has_cached_is_zero(secflp: SecureFloatingPoint) -> None:
    """
    Assert that secflp has a cached is_zero.

    :param secflp: Secure floating point value to be validated.
    :raise AssertionError: is_zero is not cached.
    """
    # explicit boolean check
    assert await secflp.has_cached_is_zero() is True


async def assert_not_has_cached_is_zero(secflp: SecureFloatingPoint) -> None:
    """
    Assert that secflp has no cached is_zero.

    :param secflp: Secure floating point value to be validated.
    :raise AssertionError: is_zero is cached.
    """
    # explicit boolean check
    assert await secflp.has_cached_is_zero() is False


async def assert_is_zero_correct(secflp: SecureFloatingPoint) -> None:
    """
    Assert that secflp.is_zero() is valid.

    :param secflp: Secure floating point value to be validated.
    :raise AssertionError: is_zero is not cached.
    :raise AssertionError: is_zero is incorrect.
    """
    await _assert_has_cached_is_zero(secflp)
    assert await flp.mpc.output(secflp.is_zero()) == int(
        await flp.mpc.output(secflp.significand) == 0
    )


async def _assert_has_cached_two_to_exponent(secflp: SecureFloatingPoint) -> None:
    """
    Assert that secflp has a cached two_to_exponent.

    :param secflp: Secure floating point value to be validated.
    :raise AssertionError: two_to_exponent is not cached.
    """
    # explicit boolean check
    assert await secflp.has_cached_two_to_exponent() is True


async def assert_not_has_cached_two_to_exponent(
    secflp: SecureFloatingPoint,
) -> None:
    """
    Assert that secflp has no cached two_to_exponent.

    :param secflp: Secure floating point value to be validated.
    :raise AssertionError: two_to_exponent is cached.
    """
    # explicit boolean check
    assert await secflp.has_cached_two_to_exponent() is False


async def assert_two_to_exponent_correct(secflp: SecureFloatingPoint) -> None:
    """
    Assert that, if available, secflp.two_to_exponent() is valid.

    If secflp.significand is zero, then two_to_exponent is meaningless and we skip verification of the
    value.

    :param secflp: Secure floating point value to be validated.
    :raise AssertionError: two_to_exponent is not cached.
    :raise AssertionError: two_to_exponent is incorrect.
    """
    await _assert_has_cached_two_to_exponent(secflp)

    if await flp.mpc.output(secflp.significand) == 0:
        return

    exponent = await flp.mpc.output(secflp.exponent)
    modulus = secflp.two_to_exponent_class.field.modulus
    expected_result = pow(2, abs(exponent), modulus)
    if exponent < 0:
        expected_result = invert(expected_result, modulus)
    assert_equals_mod(
        expected_result, await flp.mpc.output(secflp.two_to_exponent()), modulus
    )


# pylint: disable=invalid-name
def assert_equals_mod(a: int, b: int, m: int) -> None:
    """
    Assert that the two provided inputs are equivalent given the provided modulus.

    :param a: Reference value.
    :param b: Value to validate.
    :param m: Modulus.
    """
    assert (a - b) % m == 0
