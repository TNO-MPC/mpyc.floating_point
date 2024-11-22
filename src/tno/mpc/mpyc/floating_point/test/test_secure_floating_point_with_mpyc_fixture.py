"""
Testing module for the implementation of secure floating points
"""

from __future__ import annotations

import operator as op
from functools import partial
from itertools import product
from math import prod, sqrt
from random import randint

import pytest
from mpyc.sectypes import SecureFixedPoint, SecureInteger

from tno.mpc.mpyc.floating_point import (
    SYSTEM_EXPONENT_BITS,
    SYSTEM_SIGNIFICAND_BITS,
    SecureFloatingPoint,
)
from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point import log_flp, mpc_coro_extended, returnType, sqrt_flp
from tno.mpc.mpyc.floating_point._sys import (
    SYSTEM_EXPONENT_BITS,
    SYSTEM_SIGNIFICAND_BITS,
)
from tno.mpc.mpyc.floating_point.secure_floating_point import SecFlp
from tno.mpc.mpyc.floating_point.test.util import (
    MAX_EXPONENT,
    MAX_NEGATIVE_DOUBLE,
    MAX_POSITIVE_DOUBLE,
    MIN_EXPONENT,
    MIN_NEGATIVE_DOUBLE,
    MIN_POSITIVE_DOUBLE,
    assert_is_zero_correct,
    assert_not_has_cached_is_zero,
    assert_not_has_cached_two_to_exponent,
    assert_significand_in_expected_range,
    assert_two_to_exponent_correct,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("mpyc_runtime")]


# region normal test doubles
SIGNS = (-1, 1)
SIGNIFICANDS_FRACTIONAL = (
    1 + randint(0, 2**SYSTEM_SIGNIFICAND_BITS - 1) / 2**SYSTEM_SIGNIFICAND_BITS,
)
EXPONENTS = (MIN_EXPONENT // 4, 0, MAX_EXPONENT // 4)
TEST_FLOATS_SINGLE: list[float] = [0.0] + [
    float(sign * significand_frac * 2**exp)
    for sign, significand_frac, exp in product(
        SIGNS, SIGNIFICANDS_FRACTIONAL, EXPONENTS
    )
]
TEST_FLOATS_DUO = list(product(TEST_FLOATS_SINGLE, TEST_FLOATS_SINGLE))
TEST_FLOATS_QUARTET = (
    list(map(op.add, TEST_FLOATS_DUO, TEST_FLOATS_DUO))
    + list(map(op.add, TEST_FLOATS_DUO, reversed(TEST_FLOATS_DUO)))
    + [[1.0, 2.0, 3.0, 4.0]]  # fmt: skip # https://ci.tno.nl/gitlab/pet/lab/mpc/python-packages/microlibs/mpyc/microlibs/floating-point/-/issues/50
)
# endregion


# Also require high accuracy for values that are very close to zero
pytest_approx_sensitive = partial(pytest.approx, abs=MIN_POSITIVE_DOUBLE)


@pytest.mark.parametrize(
    "value",
    TEST_FLOATS_SINGLE
    + [
        MAX_POSITIVE_DOUBLE,
        MIN_POSITIVE_DOUBLE,
        MAX_NEGATIVE_DOUBLE,
        MIN_NEGATIVE_DOUBLE,
    ],
)
async def test_floating_point_output_conversion(value: float) -> None:
    """
    Test whether creating a secure floating point number from a regular floating point and revealing it produces
    the same floating point.

    :param value: floating point number used in the test.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )

    assert value == await flp.mpc.output(flp.mpc.input(secflp(value), senders=0))


@pytest.mark.parametrize("val", list(range(0, 5)))
@pytest.mark.parametrize("secnum", (flp.mpc.SecInt(), flp.mpc.SecFxp()))
async def test_output_non_floating_points(
    val: int, secnum: type[SecureInteger] | type[SecureFixedPoint]
) -> None:
    """
    Test whether the custom output function also works for secure integers and secure fixed point numbers.

    :param val: Value to be secret-shared.
    :param secnum: Type of secure number.
    """
    secure_value = flp.mpc.input(secnum(val), senders=0)
    assert await flp.mpc.output(secure_value) == val


@pytest.mark.parametrize("significand, exponent", [(1000, 10), (10, 40), (1, 10)])
async def test_floating_point_input(significand: int, exponent: int) -> None:
    """
    Test whether the input function works correctly for secure floating point numbers. This is the function that allows
    a party to secret-share their floating point input.

    :param significand: significand for the secure floating point
    :param exponent: exponent for the secure floating point
    """
    secflp = SecFlp()
    secure_flp = flp.mpc.input(secflp((significand, exponent)), senders=0)

    significand_output = await flp.mpc.output(secure_flp.significand)
    exponent_output = await flp.mpc.output(secure_flp.exponent)
    floating_point_output = await flp.mpc.output(secure_flp)
    (
        correct_sign,
        correct_exp,
    ) = SecureFloatingPoint.correct_floating_point_representation(
        significand, exponent, secflp.significand_bit_length
    )

    assert significand_output == correct_sign
    assert exponent_output == correct_exp
    assert floating_point_output == float(correct_sign * 2**correct_exp)


async def test_floating_point_input_scalar_single_sender() -> None:
    """
    Verify that flp.mpc.input delegates to _input and correctly handles scalar input from a single
    party.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_flp = flp.mpc.input(secflp(1), senders=0)
    assert isinstance(secure_flp, SecureFloatingPoint)


async def test_floating_point_input_list_single_sender() -> None:
    """
    Verify that flp.mpc.input delegates to _input and correctly handles list input from a single
    parties.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_flps = flp.mpc.input([secflp(_) for _ in range(3)], senders=0)
    assert isinstance(secure_flps, list)
    assert all(
        isinstance(secure_flp, SecureFloatingPoint) for secure_flp in secure_flps
    )


async def test_floating_point_secret_inputs_yield_no_cached_attributes() -> None:
    """
    Ensure that flp.mpc.input does not change the cached status of CachedMPyCAttributes.

    This is specifically of interest when the input to SecFlp is already secret-shared, since in
    this case the attributes are not yet known (and therefore not cached).
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )

    sig_, exp_ = secflp.find_floating_point_representation(
        1, secflp.significand_bit_length
    )
    secure_sig_exp_pair = (secflp.significand_class(sig_), secflp.exponent_class(exp_))
    secure_value = secflp(secure_sig_exp_pair)
    shared_value = flp.mpc.input(secure_value, senders=0)

    await assert_not_has_cached_is_zero(shared_value)
    await assert_not_has_cached_two_to_exponent(shared_value)


@pytest.mark.parametrize("value", TEST_FLOATS_SINGLE)
async def test_floating_point_input_is_zero(value: int) -> None:
    """
    Test whether the input function correctly computes is_zero.

    :param value: Value for the secure floating point.
    """
    secflp = SecFlp()
    secval = flp.mpc.input(secflp(value), senders=0)

    await assert_is_zero_correct(secval)


@pytest.mark.parametrize("value", TEST_FLOATS_SINGLE)
async def test_floating_point_input_two_to_exp(value: int) -> None:
    """
    Test whether the input function correctly computes two_to_exponent.

    :param value: Value for the secure floating point.
    """
    secflp = SecFlp()
    secval = flp.mpc.input(secflp(value), senders=0)

    await assert_two_to_exponent_correct(secval)


@pytest.mark.parametrize("val", [i + 1 / i for i in range(1, 30)])
async def test_floating_point_unary(val: float) -> None:
    """
    Test whether the unary operator works correctly for secure floating point numbers.

    :param val: floating point number used in the test.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_value = flp.mpc.input(secflp(val), senders=0)

    secure_value_unary = +secure_value
    answer = await flp.mpc.output(secure_value_unary)

    assert val == answer
    await assert_significand_in_expected_range(secure_value_unary)
    await assert_is_zero_correct(secure_value_unary)
    await assert_two_to_exponent_correct(secure_value_unary)


@pytest.mark.parametrize("val", [i + 1 / i for i in range(1, 30)])
async def test_floating_point_negation(val: float) -> None:
    """
    Test whether the negation operator works properly on secure floating point numbers.

    :param val: floating point number used in the test.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )

    secure_value_pos = flp.mpc.input(secflp(val), senders=0)
    secure_value_neg = flp.mpc.input(secflp(-val), senders=0)
    answer_neg_input = await flp.mpc.output(secure_value_neg)

    secure_value_negated = -secure_value_pos
    answer_securely_negated = await flp.mpc.output(secure_value_negated)

    assert (
        answer_neg_input == answer_securely_negated
    ), f"{answer_neg_input} not equal to {answer_securely_negated}"
    await assert_significand_in_expected_range(secure_value_negated)
    await assert_is_zero_correct(secure_value_negated)
    await assert_two_to_exponent_correct(secure_value_negated)


async def test_floating_point_double_negation() -> None:
    """
    Test whether we can apply negation twice without intermediate awaits.

    Used to be an issue with MPyCCachedAttribute.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_value = flp.mpc.input(secflp(1), senders=0)
    answer = await flp.mpc.output(-(-secure_value))

    assert answer == 1


@pytest.mark.parametrize(
    "num, denom", ((num, denom) for (num, denom) in TEST_FLOATS_DUO if denom != 0)
)
async def test_floating_point_division(num: float, denom: float) -> None:
    """
    Test whether division works correctly for secure floating point numbers.

    :param num: floating points number used to create a secure floating point number that will be the numerator.
    :param denom: floating points number used to create a secure floating point number that will be the denominator.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_value_1 = flp.mpc.input(secflp(num), senders=0)
    secure_value_2 = flp.mpc.input(secflp(denom), senders=0)

    division = secure_value_1 / secure_value_2
    answer = await flp.mpc.output(division)

    assert answer == pytest_approx_sensitive(
        num / denom, rel=2 ** -(secflp.significand_bit_length - 1)
    )
    await assert_significand_in_expected_range(division)
    await assert_is_zero_correct(division)
    await assert_not_has_cached_two_to_exponent(division)


@pytest.mark.parametrize("val1, val2", TEST_FLOATS_DUO)
async def test_floating_point_pair_addition(val1: float, val2: float) -> None:
    """
    Test whether addition works correctly for two secure floating point numbers.

    :param val1: floating point number used to create a secure floating point number that will be added.
    :param val2: floating point number used to create a secure floating point number that will be added.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_value_1 = flp.mpc.input(secflp(val1), senders=0)
    secure_value_2 = flp.mpc.input(secflp(val2), senders=0)

    sum_ = secure_value_1 + secure_value_2
    answer = await flp.mpc.output(sum_)

    assert answer == pytest_approx_sensitive(
        val1 + val2, rel=2 ** -(secflp.significand_bit_length - 1)
    )
    await assert_significand_in_expected_range(sum_)
    await assert_not_has_cached_is_zero(sum_)
    await assert_two_to_exponent_correct(sum_)


async def test_floating_point_trio_addition() -> None:
    """
    Test whether we can sum multiple times without awaits in between.

    Used to be an issue with MPyCCachedAttribute.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_value_1 = flp.mpc.input(secflp(1), senders=0)
    secure_value_2 = flp.mpc.input(secflp(2), senders=0)
    secure_value_3 = flp.mpc.input(secflp(3), senders=0)

    sum_ = secure_value_1 + secure_value_2 + secure_value_3
    answer = await flp.mpc.output(sum_)

    assert answer == pytest_approx_sensitive(
        6, rel=2 ** -(secflp.significand_bit_length - 1)
    )
    await assert_significand_in_expected_range(sum_)
    await assert_not_has_cached_is_zero(sum_)
    await assert_two_to_exponent_correct(sum_)


@pytest.mark.parametrize("test_numbers", TEST_FLOATS_QUARTET)
async def test_floating_point_multiple_addition_all_two_to_exponent_cached(
    test_numbers: tuple[float],
) -> None:
    """
    Test whether addition works correctly for multiple secure floating point numbers if all their
    two_to_exponents are cached.

    :param test_numbers: floating point numbers used to create a secure floating point number that will be added.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
        max_concurrent_additions=len(test_numbers),
    )
    secure_values = [flp.mpc.input(secflp(val), senders=0) for val in test_numbers]

    sum_ = SecureFloatingPoint.sum(*secure_values)
    answer = await flp.mpc.output(sum_)

    # The order of addition matters, e.g. sum(1, -1, 2 ** -64) != sum(2 ** -64, 1, -1).
    # secflp.add resembles addition in order of increasing modulus.
    sorted_test_numbers = sorted(test_numbers, key=abs)
    assert answer == pytest_approx_sensitive(
        sum(sorted_test_numbers), rel=2 ** -(secflp.significand_bit_length - 1)
    )
    await assert_significand_in_expected_range(sum_)
    await assert_not_has_cached_is_zero(sum_)
    await assert_two_to_exponent_correct(sum_)


@pytest.mark.parametrize("test_numbers", TEST_FLOATS_QUARTET)
async def test_floating_point_multiple_addition_not_two_to_exponent_cached(
    test_numbers: tuple[float],
) -> None:
    """
    Test whether addition works correctly for multiple secure floating point numbers if no
    two_to_exponent are cached.

    :param test_numbers: floating point numbers used to create a secure floating point number that will be added.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
        max_concurrent_additions=len(test_numbers),
    )

    # Initialise secure values that do not have cached two exponent
    sig_exp_pairs = [
        secflp.find_floating_point_representation(number, secflp.significand_bit_length)
        for number in test_numbers
    ]
    secure_sig_exp_pairs = [
        (secflp.significand_class(sig), secflp.exponent_class(exp))
        for sig, exp in sig_exp_pairs
    ]
    secure_values = [
        flp.mpc.input(secflp(val), senders=0) for val in secure_sig_exp_pairs
    ]

    for secure_value in secure_values:
        await assert_not_has_cached_two_to_exponent(secure_value)

    sum_ = SecureFloatingPoint.sum(*secure_values)
    answer = await flp.mpc.output(sum_)

    # The order of addition matters, e.g. sum(1, -1, 2 ** -64) != sum(2 ** -64, 1, -1).
    # secflp.add resembles addition in order of increasing modulus.
    sorted_test_numbers = sorted(test_numbers, key=abs)
    assert answer == pytest_approx_sensitive(
        sum(sorted_test_numbers), rel=2 ** -(secflp.significand_bit_length - 1)
    )
    await assert_significand_in_expected_range(sum_)
    await assert_not_has_cached_is_zero(sum_)
    await assert_not_has_cached_two_to_exponent(sum_)


@pytest.mark.parametrize("test_numbers", TEST_FLOATS_QUARTET)
async def test_floating_point_multiple_addition_some_two_to_exponent_cached(
    test_numbers: tuple[float],
) -> None:
    """
    Test whether addition works correctly for multiple secure floating point numbers if some
    two_to_exponent are cached and some are not.

    :param test_numbers: floating point numbers used to create a secure floating point number that will be added.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
        max_concurrent_additions=len(test_numbers),
    )

    # Initialise secure values that do not have cached two exponent
    sig_exp_pairs = [
        secflp.find_floating_point_representation(number, secflp.significand_bit_length)
        for number in test_numbers[: len(test_numbers) // 2]
    ]
    secure_sig_exp_pairs = [
        (secflp.significand_class(sig), secflp.exponent_class(exp))
        for sig, exp in sig_exp_pairs
    ]
    secure_values_has_not_cached_two_to_exp = [
        flp.mpc.input(secflp(val), senders=0) for val in secure_sig_exp_pairs
    ]
    for secure_value in secure_values_has_not_cached_two_to_exp:
        await assert_not_has_cached_two_to_exponent(secure_value)

    # Initialise secure values that do have cached two exponent
    secure_values_has_cached_two_to_exp = [
        flp.mpc.input(secflp(val), senders=0)
        for val in test_numbers[len(test_numbers) // 2 :]
    ]

    secure_values = (
        secure_values_has_not_cached_two_to_exp + secure_values_has_cached_two_to_exp
    )

    sum_ = SecureFloatingPoint.sum(*secure_values)
    answer = await flp.mpc.output(sum_)

    # The order of addition matters, e.g. sum(1, -1, 2 ** -64) != sum(2 ** -64, 1, -1).
    # secflp.add resembles addition in order of increasing modulus.
    sorted_test_numbers = sorted(test_numbers, key=abs)
    assert answer == pytest_approx_sensitive(
        sum(sorted_test_numbers), rel=2 ** -(secflp.significand_bit_length - 1)
    )
    await assert_significand_in_expected_range(sum_)
    await assert_not_has_cached_is_zero(sum_)
    await assert_two_to_exponent_correct(sum_)


@pytest.mark.parametrize("val1, val2", TEST_FLOATS_DUO)
async def test_floating_point_pair_subtraction(val1: float, val2: float) -> None:
    """
    Test whether subtraction works correctly for two secure floating point numbers.

    :param val1: floating point number used to create a secure floating point number that will be added.
    :param val2: floating point number used to create a secure floating point number that will be added.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_value_1 = flp.mpc.input(secflp(val1), senders=0)
    secure_value_2 = flp.mpc.input(secflp(val2), senders=0)

    diff = secure_value_1 - secure_value_2
    answer = await flp.mpc.output(diff)

    assert answer == pytest_approx_sensitive(
        val1 - val2, rel=2 ** -(secflp.significand_bit_length - 1)
    )
    await assert_significand_in_expected_range(diff)
    await assert_not_has_cached_is_zero(diff)
    await assert_two_to_exponent_correct(diff)


@pytest.mark.parametrize("val1, val2", TEST_FLOATS_DUO)
async def test_floating_point_pair_multiplication(val1: float, val2: float) -> None:
    """
    Test whether multiplication works correctly for two secure floating point numbers.

    :param val1: floating point number used to create a secure floating point number that will be multiplied.
    :param val2: floating point number used to create a secure floating point number that will be multiplied.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_value_1 = flp.mpc.input(secflp(val1), senders=0)
    secure_value_2 = flp.mpc.input(secflp(val2), senders=0)

    prod_ = secure_value_1 * secure_value_2
    answer = await flp.mpc.output(prod_)

    assert answer == pytest_approx_sensitive(
        val1 * val2, rel=2 ** -(secflp.significand_bit_length - 1)
    )
    await assert_significand_in_expected_range(prod_)
    await assert_is_zero_correct(prod_)
    await assert_two_to_exponent_correct(prod_)


async def test_floating_point_trio_multiplication() -> None:
    """
    Test whether we can multiply multiple times without awaits in between.

    Used to be an issue with MPyCCachedAttribute.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
    )
    secure_value_1 = flp.mpc.input(secflp(1), senders=0)
    secure_value_2 = flp.mpc.input(secflp(2), senders=0)
    secure_value_3 = flp.mpc.input(secflp(3), senders=0)

    prod_ = secure_value_1 * secure_value_2 * secure_value_3
    answer = await flp.mpc.output(prod_)

    assert answer == pytest_approx_sensitive(
        6, rel=2 ** -(secflp.significand_bit_length - 1)
    )
    await assert_significand_in_expected_range(prod_)
    await assert_is_zero_correct(prod_)
    await assert_two_to_exponent_correct(prod_)


@pytest.mark.parametrize("test_numbers", TEST_FLOATS_QUARTET)
async def test_floating_point_multiple_multiplication(
    test_numbers: tuple[float],
) -> None:
    """
    Test whether multiplication works correctly for multiple secure floating point numbers.

    :param test_numbers: floating point numbers used to create a secure floating point number that will be multiplied.
    """
    secflp = SecFlp(
        significand_bit_length=SYSTEM_SIGNIFICAND_BITS,
        exponent_bit_length=SYSTEM_EXPONENT_BITS,
        max_concurrent_multiplications=len(test_numbers),
    )
    secure_values = [flp.mpc.input(secflp(val), senders=0) for val in test_numbers]

    prod_ = SecureFloatingPoint.prod(*secure_values)
    answer = await flp.mpc.output(prod_)

    sorted_test_numbers = sorted(test_numbers, key=abs)
    accumulated_rel_error = (1 + 2 ** -(secflp.significand_bit_length - 1)) ** (
        len(test_numbers) - 1
    ) - 1
    assert answer == pytest_approx_sensitive(
        prod(sorted_test_numbers), rel=accumulated_rel_error
    )
    await assert_significand_in_expected_range(prod_)
    await assert_is_zero_correct(prod_)
    await assert_two_to_exponent_correct(prod_)


@pytest.mark.parametrize("val", list(range(20)))
async def test_floating_point_logarithm(val: float) -> None:
    """
    Test whether the logarithm function for secure floating point numbers is close enough to the regular logarithm
    function.

    :param val: floating point number used to create the secure floating point number in the test.
    """
    secfxp = flp.mpc.SecFxp(64)
    secflp = SecFlp(significand_bit_length=16, exponent_bit_length=8)
    two_to_pow_val = 2**val
    secure_flp = flp.mpc.input(secflp(two_to_pow_val), senders=0)

    result = log_flp(secure_flp, secfxp_type=secfxp)
    answer = await flp.mpc.output(result)

    assert answer == pytest_approx_sensitive(val, abs=0.001)
    await assert_significand_in_expected_range(result)
    await assert_is_zero_correct(result)
    await assert_not_has_cached_two_to_exponent(result)


@pytest.mark.parametrize("val", list(range(20)) + [2**-12, 2**-2, 2**16, 2.5**18])
async def test_floating_point_square_root(val: float) -> None:
    """
    Test whether the square root function for secure floating point numbers is close enough to the regular logarithm
    function.

    :param val: floating point number used to create the secure floating point number in the test.
    """
    secfxp = flp.mpc.SecFxp(64)
    secflp = SecFlp(significand_bit_length=30, exponent_bit_length=16)
    secure_flp = flp.mpc.input(secflp(val), senders=0)

    result = sqrt_flp(secure_flp, secfxp_type=secfxp)
    answer = await flp.mpc.output(result)

    assert answer == pytest_approx_sensitive(sqrt(val), abs=1e-3, rel=1e-3)
    await assert_significand_in_expected_range(result)
    await assert_is_zero_correct(result)
    await assert_not_has_cached_two_to_exponent(result)


@pytest.mark.parametrize("val1, val2, val3", [(1, 1.0, 1.0), (2, 2.0, 2.0)])
async def test_returntype_with_different_types(
    val1: int, val2: float, val3: float
) -> None:
    """
    Test whether the extended returnType function correctly create dummies of different types when param_1 list of different
    types is provided as input.

    :param val1: input to the secure integer used in the test.
    :param val2: input to the secure fixed point number used in the test.
    :param val3: input to the secure floating point number used in the test.
    """

    @mpc_coro_extended
    async def test_func(
        param_1: SecureInteger,
        param_2: SecureFixedPoint,
        param_3: SecureFloatingPoint,
    ) -> tuple[
        SecureInteger,
        SecureFixedPoint,
        SecureFloatingPoint,
    ]:
        """
        Dummy function that returns its inputs.

        :param param_1: A secure integer.
        :param param_2: A secure fixed-point number.
        :param param_3: A secure floating-point number.
        :return: The input
        """
        await returnType((type(param_1), type(param_2), type(param_3)))
        return param_1, param_2, param_3

    secint = flp.mpc.SecInt()
    secfxp = flp.mpc.SecFxp()
    secflp = SecFlp()
    val1_secint = flp.mpc.input(secint(val1), senders=0)
    val2_secfxp = flp.mpc.input(secfxp(val2), senders=0)
    val3_secflp = flp.mpc.input(secflp(val3), senders=0)

    val1_dummy, val2_dummy, val3_dummy = test_func(
        val1_secint, val2_secfxp, val3_secflp
    )
    val1_result = await flp.mpc.output(val1_dummy)
    val2_result = await flp.mpc.output(val2_dummy)
    val3_result = await flp.mpc.output(val3_dummy)

    assert isinstance(val1_dummy, type(val1_secint))
    assert isinstance(val2_dummy, type(val2_secfxp))
    assert isinstance(val3_dummy, type(val3_secflp))
    assert val1_result == val1
    assert val2_result == val2
    assert val3_result == pytest_approx_sensitive(val3)
