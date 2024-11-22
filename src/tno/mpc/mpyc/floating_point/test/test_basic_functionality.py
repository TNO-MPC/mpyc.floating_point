"""
Tests to verify the correctness of the implemented basic functions
"""

from __future__ import annotations

import math
from math import floor, log2

import pytest
from mpyc.gmpy import invert

from tno.mpc.mpyc.floating_point import SYSTEM_EXPONENT_BITS
from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point import bit_length, most_significant_bit
from tno.mpc.mpyc.floating_point._sys import SYSTEM_EXPONENT_BITS
from tno.mpc.mpyc.floating_point.basic_functions import (
    exp2_fxp_integral,
    exp2_int,
    integer,
    log_fxp,
    natural_log,
    parity,
    sqrt_fxp,
    sqrt_small,
)
from tno.mpc.mpyc.floating_point.test.util import assert_equals_mod

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("mpyc_runtime")]


@pytest.mark.parametrize("exponent", list(range(20)))
async def test_logarithm_standard(exponent: int) -> None:
    """
    Test whether the output of the secure logarithm function is close enough to that of the standard logarithm function
    using base e.

    :param exponent: exponent used in the test.
    """
    secfxp = flp.mpc.SecFxp(64)
    value = flp.mpc.input(secfxp(math.exp(exponent)), senders=0)

    ln_x = natural_log(value)
    ln_x_res = await flp.mpc.output(ln_x)

    assert ln_x_res == pytest.approx(exponent, abs=0.01)


@pytest.mark.parametrize("exponent", list(range(32)))
async def test_logarithm_2(exponent: int) -> None:
    """
    Test whether the output of the secure logarithm function is close enough to that of the standard logarithm function
    using base 2.

    :param exponent: exponent used in the test.
    """
    secfxp = flp.mpc.SecFxp(64)
    secure_value = flp.mpc.input(secfxp(2**exponent), senders=0)

    log2_x = log_fxp(secure_value, 2)
    log2_x_res = await flp.mpc.output(log2_x)

    assert log2_x_res == pytest.approx(exponent, abs=0.01)


@pytest.mark.parametrize("exponent", [-16, 0, 16])
async def test_two_fxp(exponent: int) -> None:
    """
    Test to check whether exponentiation with base 2 works correctly.

    :param exponent: exponent used in the test.
    """
    secfxp = flp.mpc.SecFxp()
    secure_value = flp.mpc.input(secfxp(exponent), senders=0)

    two_x = exp2_fxp_integral(secure_value)
    two_x_res = await flp.mpc.output(two_x)

    assert two_x_res == 2**exponent


@pytest.mark.parametrize(
    "exponent", [2**SYSTEM_EXPONENT_BITS - 1, 0, -(2**SYSTEM_EXPONENT_BITS - 1)]
)
async def test_two_int(exponent: int) -> None:
    """
    Test to check whether exponentiation with base 2 works correctly.

    :param exponent: exponent used in the test.
    """
    secint = flp.mpc.SecInt(SYSTEM_EXPONENT_BITS)
    secure_value = flp.mpc.input(secint(exponent), senders=0)
    modulus = secint.field.modulus

    two_x = exp2_int(secure_value)
    two_x_res = await flp.mpc.output(two_x)

    expected_result = pow(2, abs(exponent), modulus)
    if exponent < 0:
        expected_result = invert(expected_result, modulus)
    assert_equals_mod(expected_result, two_x_res, modulus)


@pytest.mark.parametrize("number", [i + i / (i + 1) for i in range(32)])
async def test_integer_func(number: float) -> None:
    """
    Test whether casting a secure fixed point to a secure integer works correctly.

    :param number: floating point number to be cast to an integer.
    """
    secfxp = flp.mpc.SecFxp()
    secure_value = flp.mpc.input(secfxp(number), senders=0)

    int_x = integer(secure_value)
    int_x_res = await flp.mpc.output(int_x)

    assert int_x_res == int(number)


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 1.1, 2.2, 3.3, 4.4, 5.5])
async def test_parity_func(number: int | float) -> None:
    """
    Test whether the secure parity function works correctly.

    :param number: integer or floating point number used in the test.
    """
    secfxp = flp.mpc.SecFxp()
    secure_value = flp.mpc.input(secfxp(number), senders=0)

    parity_x = parity(secure_value)
    parity_x_res = await flp.mpc.output(parity_x)

    assert parity_x_res == (int(number) % 2)


@pytest.mark.parametrize("value", (i / 10 for i in range(5, 10)))
async def test_sqrt_small(value: float) -> None:
    """
    Test whether the secure square root function on a constrained interval produces values close enough to the
    regular square root function.

    :param value: floating point number between 0.5 and 1.
    """
    secfxp = flp.mpc.SecFxp()
    secure_value = flp.mpc.input(secfxp(value), senders=0)

    sqrt_x = sqrt_small(secure_value)
    sqrt_x_res = await flp.mpc.output(sqrt_x)

    assert sqrt_x_res == pytest.approx(math.sqrt(value), abs=0.01)


@pytest.mark.parametrize("value", (1, 23, 74, 255, 346, 1463, 23586))
async def test_sqrt_standard(value: int) -> None:
    """
    Test whether the secure square root function without constrained input produces values close enough to the
    regular square root function.

    :param value: value that will be squared in the test.
    """
    secfxp = flp.mpc.SecFxp()
    secure_value = flp.mpc.input(secfxp(value), senders=0)

    sqrt_x = sqrt_fxp(secure_value)
    sqrt_x_res = await flp.mpc.output(sqrt_x)

    assert sqrt_x_res == pytest.approx(math.sqrt(value), abs=0.01)


@pytest.mark.parametrize(
    "val", list(range(0, 16)) + [2**flp.mpc.options.bit_length - 1]
)
async def test_secure_bit_length(val: int) -> None:
    """
    Test whether the secure_bit_length function works correctly.

    :param val: value used in the test.
    """
    secint = flp.mpc.SecInt()
    secure_value = flp.mpc.input(secint(val), senders=0)
    expected_length = val.bit_length()

    len_secure, two_to_length_secure, inv_two_to_length_secure = bit_length(
        secure_value, two_to_pow=True, inv_two_to_pow=True
    )
    len_ = await flp.mpc.output(len_secure)
    two_to_length = await flp.mpc.output(two_to_length_secure)
    mult_inverse_check = await flp.mpc.output(
        inv_two_to_length_secure * two_to_length_secure
    )

    assert len_ == expected_length
    assert two_to_length == 2**expected_length
    assert mult_inverse_check == 1


@pytest.mark.parametrize("val", list(range(4, 31)))
async def test_secure_bit_length_with_bounds(val: int) -> None:
    """
    Test whether the secure_bit_length function works correctly when bounds are provided.

    :param val: value used in the test.
    """
    secint = flp.mpc.SecInt()
    secure_value = flp.mpc.input(secint(val), senders=0)
    expected_length = val.bit_length()

    # input 4 has 3 bits
    # input 31 has 5 bits
    len_secure, two_to_length_secure, inv_two_to_length_secure = bit_length(
        secure_value,
        two_to_pow=True,
        inv_two_to_pow=True,
        lower_bound=3,
        upper_bound=5,
    )
    len_ = await flp.mpc.output(len_secure)
    two_to_length = await flp.mpc.output(two_to_length_secure)
    mult_inverse_check = await flp.mpc.output(
        inv_two_to_length_secure * two_to_length_secure
    )

    assert len_ == expected_length
    assert two_to_length == 2**expected_length
    assert mult_inverse_check == 1


@pytest.mark.parametrize(
    "val", list(range(1, 16)) + [2**flp.mpc.options.bit_length - 1]
)
async def test_most_significant_bit(val: int) -> None:
    """
    Test whether the most_significant_bit function works correctly.

    This function is tested with input 0 in another test.

    :param val: value used in the test.
    """
    secint = flp.mpc.SecInt()
    secure_value = flp.mpc.input(secint(val), senders=0)
    expected_msb = floor(log2(abs(val)))

    msb_secure, two_to_msb_secure, inv_two_to_msb_secure = most_significant_bit(
        secure_value, two_to_pow=True, inv_two_to_pow=True
    )
    msb = await flp.mpc.output(msb_secure)
    two_to_msb = await flp.mpc.output(two_to_msb_secure)
    mult_inverse_check = await flp.mpc.output(inv_two_to_msb_secure * two_to_msb_secure)

    assert msb == expected_msb
    assert two_to_msb == 2**expected_msb
    assert mult_inverse_check == 1


async def test_most_significant_bit_zero_input() -> None:
    """
    Test whether the most_significant_bit function works correctly with zero input.
    """
    secint = flp.mpc.SecInt()
    secure_value = flp.mpc.input(secint(0), senders=0)

    msb_secure, two_to_msb_secure, inv_two_to_msb_secure = most_significant_bit(
        secure_value, two_to_pow=True, inv_two_to_pow=True
    )
    msb = await flp.mpc.output(msb_secure)
    inv_two_to_msb = await flp.mpc.output(inv_two_to_msb_secure)
    mult_inverse_check = await flp.mpc.output(inv_two_to_msb_secure * two_to_msb_secure)

    assert msb == -1
    assert inv_two_to_msb == 2
    assert mult_inverse_check == 1


@pytest.mark.parametrize("val", list(range(8, 31)))
async def test_most_significant_bit_with_upper_bound(val: int) -> None:
    """
    Test whether the most_significant_bit function works correctly if an upper bound is provided.

    :param val: value used in the test.
    """
    secint = flp.mpc.SecInt()
    secure_value = flp.mpc.input(secint(val), senders=0)
    expected_msb = floor(log2(abs(val)))

    # input 31 has 5 bits, so MSB should return index 4
    msb_secure, two_to_msb_secure, inv_two_to_msb_secure = most_significant_bit(
        secure_value, two_to_pow=True, inv_two_to_pow=True, upper_bound=4
    )
    msb = await flp.mpc.output(msb_secure)
    two_to_msb = await flp.mpc.output(two_to_msb_secure)
    mult_inverse_check = await flp.mpc.output(inv_two_to_msb_secure * two_to_msb_secure)

    assert msb == expected_msb
    assert two_to_msb == 2**expected_msb
    assert mult_inverse_check == 1


@pytest.mark.parametrize("val", list(range(8, 32)))
async def test_most_significant_bit_with_lower_bound(val: int) -> None:
    """
    Test whether the most_significant_bit function works correctly if a lower bound is provided.

    :param val: value used in the test.
    """
    secint = flp.mpc.SecInt()
    secure_value = flp.mpc.input(secint(val), senders=0)
    expected_msb = floor(log2(abs(val)))

    # input 8 has 4 bits, so MSB should return index 3
    msb_secure, two_to_msb_secure, inv_two_to_msb_secure = most_significant_bit(
        secure_value, two_to_pow=True, inv_two_to_pow=True, lower_bound=3
    )
    msb = await flp.mpc.output(msb_secure)
    two_to_msb = await flp.mpc.output(two_to_msb_secure)
    mult_inverse_check = await flp.mpc.output(inv_two_to_msb_secure * two_to_msb_secure)

    assert msb == expected_msb
    assert two_to_msb == 2**expected_msb
    assert mult_inverse_check == 1
