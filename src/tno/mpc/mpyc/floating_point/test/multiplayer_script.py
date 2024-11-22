"""
Script for testing several operations explicitly with multiple players.

Some issues do not surface in tests with a single party (-M 0 or -M 1). For example, the input
function is not fully tested unless some parties call it with a placeholder value (secflp(None)).
This script is called by `test_secure_floating_point.py` with -M 3 so that these cases are tested
more thoroughly.

It is advised to debug this script by running it in three separate terminals:

python src/tno/mpc/mpyc/floating_point/test/multiplayer_script.py -M3 -I0
python src/tno/mpc/mpyc/floating_point/test/multiplayer_script.py -M3 -I1
python src/tno/mpc/mpyc/floating_point/test/multiplayer_script.py -M3 -I2
"""

# pylint: disable=invalid-name

import pytest
from mpyc.runtime import mpc

from tno.mpc.mpyc.floating_point import (
    SecFlp,
    SecureFloatingPoint,
    log_flp,
    most_significant_bit,
    sqrt_flp,
)
from tno.mpc.mpyc.floating_point.basic_functions import exp2_int
from tno.mpc.mpyc.floating_point.test.util import assert_equals_mod

secflp = SecFlp(max_concurrent_additions=3)
secint = mpc.SecInt()


async def test_input_output() -> None:
    """
    Test I/O.

    Verify that flp.mpc.input delegates to _input and correctly handles scalar/ list input from a
    single party/ multiple parties.
    """

    async with mpc:
        print("--- Scalar input by single party ---")
        secflp = SecFlp()
        secure_flp = mpc.input(secflp(1), senders=0)
        assert isinstance(secure_flp, SecureFloatingPoint)
        assert await mpc.output(secure_flp) == 1

        print("--- List input by single party ---")
        secure_flps = mpc.input([secflp(i) for i in range(3)], senders=0)
        assert isinstance(secure_flps, list)
        assert all(
            isinstance(secure_flp, SecureFloatingPoint) for secure_flp in secure_flps
        )
        assert await mpc.output(secure_flps) == list(range(3))

        print("--- Scalar input by multiple parties ---")
        secure_flp_per_party = mpc.input(secflp(mpc.pid))
        assert isinstance(secure_flp_per_party, list)
        assert all(
            isinstance(secure_flp, SecureFloatingPoint)
            for secure_flp in secure_flp_per_party
        )
        assert await mpc.output(secure_flp_per_party) == list(range(3))

        print("--- List input by multiple parties ---")
        secure_flps_per_party = mpc.input([secflp(3 * mpc.pid + i) for i in range(3)])
        assert isinstance(secure_flps_per_party, list)
        assert all(
            isinstance(secure_flps_party_i, list)
            for secure_flps_party_i in secure_flps_per_party
        )
        assert all(
            isinstance(secure_flp_party_i, SecureFloatingPoint)
            for secure_flps_party_i in secure_flps_per_party
            for secure_flp_party_i in secure_flps_party_i
        )
        result = [
            await mpc.output(secure_flps_party_i)
            for secure_flps_party_i in secure_flps_per_party
        ]
        assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


async def test_secure_bit_length() -> None:
    """
    Test functions in the secure_bit_length module.

    We have been bitten HARD by the secure_bit_length module. Mostly because several SecFlp methods
    depend on it and debugging async code (in the MPyC framework) is quite the endeavor.

    :raises ValueError: Some computed MSB values are incorrect.
    """
    async with mpc:
        value = 4
        nr_tests = 20

        expected_msb = max(0, value.bit_length() - 1)

        vector = [secint(value if mpc.pid == 0 else None) for _ in range(nr_tests)]
        sec_vector = [mpc.input(vector, senders=0) for vector in vector]
        msb = [most_significant_bit(_) for _ in sec_vector]
        res_msb = await mpc.output(msb)
        if any(_ != expected_msb for _ in res_msb):
            raise ValueError(
                f"Not all MSB values are correct. Expected all values to equal {expected_msb}, but received the following:\n",
                res_msb,
            )


async def test_basic_functions() -> None:
    """
    Test some of the basic functions.
    """
    async with mpc:
        print("--- 2 ** 3 ---")
        pos_three = mpc.input(secint(3 if mpc.pid == 0 else None), senders=0)
        two_to_three = exp2_int(pos_three)

        res_two_to_three = await mpc.output(two_to_three)
        print(f"2 ** 3 = {res_two_to_three}")
        assert res_two_to_three == 8

        print("--- 2 ** -3 ---")
        neg_three = mpc.input(secint(-3 if mpc.pid == 0 else None), senders=0)
        two_to_pow_neg_three = exp2_int(neg_three)

        res_two_to_pow_neg_three = await mpc.output(two_to_pow_neg_three)
        print(f"2 ** -3 = {res_two_to_pow_neg_three}")
        assert_equals_mod(8 * res_two_to_pow_neg_three, 1, secint.field.modulus)


async def test_arithmetic() -> None:
    """
    Test some general arithmetic with multiple players.
    """
    async with mpc:
        print("--- a ---")
        # Ensure that some parties initialize an empty secflp, and that this does not cause issues.
        a = mpc.input(secflp(0 if mpc.pid == 0 else None), senders=0)
        assert await a.has_cached_is_zero(), "a does not have a cached zero"
        assert (
            await a.has_cached_two_to_exponent()
        ), "a does not have a cached two exponent"

        print("--- b ---")
        b = mpc.input(secflp(2 if mpc.pid == 1 else None), senders=1)
        assert await b.has_cached_is_zero(), "b does not have a cached zero"
        assert (
            await b.has_cached_two_to_exponent()
        ), "b does not have a cached two exponent"

        print("--- c ---")
        c = mpc.input(secflp(3 if mpc.pid == 2 else None), senders=2)
        assert await c.has_cached_is_zero(), "c does not have a cached zero"
        assert (
            await c.has_cached_two_to_exponent()
        ), "c does not have a cached two exponent"

        print("--- a * b ---")
        mul_a_b = a * b
        assert await mul_a_b.has_cached_is_zero(), "a * b does not have a cached zero"
        assert (
            await mul_a_b.has_cached_two_to_exponent()
        ), "a * b has no cached two exponent"

        print("--- b * c ---")
        mul_b_c = b * c
        assert await mul_b_c.has_cached_is_zero(), "b * c does not have a cached zero"
        assert (
            await mul_b_c.has_cached_two_to_exponent()
        ), "b * c has no cached two exponent"

        print("--- a + b ---")
        add_a_b = a + b
        assert not await add_a_b.has_cached_is_zero(), "a + b does have cached zero"
        assert (
            await add_a_b.has_cached_two_to_exponent()
        ), "a + b has no cached two exponent"

        print("--- a + a ---")
        # Performing multiple operations on a single object (without calling output in between)
        # was hard to get right -- make sure to always test proper functioning!
        add_a_a = a + a
        assert not await add_a_a.has_cached_is_zero(), "a + a does have cached zero"
        assert (
            await add_a_a.has_cached_two_to_exponent()
        ), "a + a has no cached two exponent"

        print("--- b - c ---")
        sub_b_c = b - c
        assert not await sub_b_c.has_cached_is_zero(), "b - c has cached zero"
        assert (
            await sub_b_c.has_cached_two_to_exponent()
        ), "b - c has no cached two exponent"

        print("--- c / b ---")
        div_c_b = c / b
        assert await div_c_b.has_cached_is_zero(), "c / b has no cached zero"
        assert (
            not await div_c_b.has_cached_two_to_exponent()
        ), "c / b has a cached two exponent"

        print("--- composite (a + b) * c / b ---")
        composite = a + b
        composite *= c
        composite /= b

        print("--- outputs ---")
        # Validate outputs only after all computations to maximize task creation before Futures are
        # set. If we await the output in between, stuff may seem to work whereas actually the tasks
        # could interfere.
        res_a = await mpc.output(a)
        print(f"a = {res_a}")
        assert res_a == 0

        res_a_is_zero = await mpc.output(a.is_zero())
        print(f"a.is_zero() = {res_a_is_zero}")
        assert res_a_is_zero == 1

        res_b = await mpc.output(b)
        print(f"b = {res_b}")
        assert res_b == 2

        res_b_is_zero = await mpc.output(b.is_zero())
        print(f"b.is_zero() = {res_b_is_zero}")
        assert res_b_is_zero == 0

        res_c = await mpc.output(c)
        print(f"c = {res_c}")
        assert res_c == 3

        res_c_is_zero = await mpc.output(c.is_zero())
        print(f"c.is_zero() = {res_c_is_zero}")
        assert res_c_is_zero == 0

        res_a_mul_b = await mpc.output(mul_a_b)
        print(f"a * b = {res_a_mul_b}")
        assert res_a_mul_b == 0

        res_a_mul_b_is_zero = await mpc.output(mul_a_b.is_zero())
        print(f"(a * b).is_zero() = {res_a_mul_b_is_zero}")
        assert res_a_mul_b_is_zero == 1

        res_b_mul_c = await mpc.output(mul_b_c)
        print(f"b * c = {res_b_mul_c}")
        assert res_b_mul_c == 6

        res_b_mul_c_is_zero = await mpc.output(mul_b_c.is_zero())
        print(f"(b * c).is_zero() = {res_b_mul_c_is_zero}")
        assert res_b_mul_c_is_zero == 0

        res_add_a_b = await mpc.output(add_a_b)
        print(f"a + b = {res_add_a_b}")
        assert res_add_a_b == 2

        res_add_a_a = await mpc.output(add_a_a)
        print(f"a + a = {res_add_a_a}")
        assert res_add_a_a == 0

        res_sub_b_c = await mpc.output(sub_b_c)
        print(f"b - c = {res_sub_b_c}")
        assert res_sub_b_c == -1

        res_div_c_b = await mpc.output(div_c_b)
        print(f"c / b = {res_div_c_b}")
        assert res_div_c_b == 3 / 2

        res_composite = await mpc.output(composite)
        print(f"(a + b) * c / b = {res_composite}")
        assert res_composite == 3


async def test_repeated_operations() -> None:
    """
    Ensure that repeated computations without in-between awaits yield the correct results.

    Especially the cached MPyC properties are sensitive about this.
    """
    async with mpc:
        print("--- -(-a) ---")
        a = mpc.input(secflp(10 if mpc.pid == 0 else None), senders=0)

        doubly_negated_a = -(-a)
        res_doubly_negated_a = await mpc.output(doubly_negated_a)
        print(f"-(-a) = {res_doubly_negated_a}")
        assert res_doubly_negated_a == 10
        assert await doubly_negated_a.has_cached_is_zero()

        print("--- b + b + ... + b ---")
        b = mpc.input(secflp(30 if mpc.pid == 0 else None), senders=0)
        n_reps = 5

        repeatedly_summed_b = secflp(0)
        for _ in range(n_reps):
            repeatedly_summed_b += b

        res_repeatedly_summed_b = await mpc.output(repeatedly_summed_b)
        print(f"b + b + ... + b ({n_reps} times) = {res_repeatedly_summed_b}")
        assert res_repeatedly_summed_b == n_reps * 30
        assert not await repeatedly_summed_b.has_cached_is_zero()
        assert await repeatedly_summed_b.has_cached_two_to_exponent()

        print("--- c * c * ... * c ---")
        c = mpc.input(secflp(20 if mpc.pid == 0 else None), senders=0)
        n_reps = 5

        repeatedly_multiplied_c = secflp(1)
        for _ in range(n_reps):
            repeatedly_multiplied_c *= c

        res_repeatedly_multiplied_c = await mpc.output(repeatedly_multiplied_c)
        print(f"c * c * ... * c ({n_reps} times) = {res_repeatedly_multiplied_c}")
        assert res_repeatedly_multiplied_c == 20**n_reps
        assert await repeatedly_multiplied_c.has_cached_is_zero()
        assert await mpc.output(repeatedly_multiplied_c.is_zero()) == 0
        assert await repeatedly_multiplied_c.has_cached_two_to_exponent()


async def test_algebraic_functions() -> None:
    """
    Test several algebraic functions.
    """
    async with mpc:
        a = mpc.input(secflp(4 if mpc.pid == 0 else None), senders=0)
        secfxp = mpc.SecFxp(256)
        sqrt_a = sqrt_flp(a, secfxp)
        assert await mpc.output(sqrt_a) == pytest.approx(2, abs=0.001)


async def test_transcendental_functions() -> None:
    """
    Test several transcendental functions.
    """
    async with mpc:
        a = mpc.input(secflp(2 if mpc.pid == 0 else None), senders=0)
        secfxp = mpc.SecFxp(64)
        log_a = log_flp(a, secfxp)
        assert await mpc.output(log_a) == pytest.approx(1, abs=0.0001)


async def test_is_zero_penetrates_if_possible() -> None:
    """
    Assert that CachedMPyCAttributes penetrate into placeholder objects even if the attribute is
    set after the construction of the placeholder.
    """
    async with mpc:
        sig_, exp_ = secflp.find_floating_point_representation(
            1, secflp.significand_bit_length
        )
        secure_sig_exp_pair = (
            secflp.significand_class(sig_),
            secflp.exponent_class(exp_),
        )
        non_cachable_secflp_1 = secflp(secure_sig_exp_pair)
        a = mpc.input(non_cachable_secflp_1, senders=0)

        abs_a = abs(a)
        assert not await a.has_cached_is_zero()
        assert not await abs_a.has_cached_is_zero()
        abs_abs_a = abs(abs_a)
        # abs_abs_a has not yet awaited _value_computed_externally. The call below sets the event, which
        # penetrates through into r2.
        abs_a.is_zero()
        assert await abs_a.has_cached_is_zero()
        assert await abs_abs_a.has_cached_is_zero()
        assert await mpc.output(abs_abs_a.is_zero()) == 0


async def test_is_zero_doesnt_penetrate_if_not_possible_1() -> None:
    """
    Assert that CachedMPyCAttributes do not penetrate into placeholder objects if the attribute is
    set after the _value_computed_externally (type Future[Event]) is awaited, and that the program
    doesn't crash if the event is changed after the future is awaited.

    Under test: intermediate call to `has_cached_is_zero`.
    """
    async with mpc:
        sig_, exp_ = secflp.find_floating_point_representation(
            1, secflp.significand_bit_length
        )
        secure_sig_exp_pair = (
            secflp.significand_class(sig_),
            secflp.exponent_class(exp_),
        )
        non_cachable_secflp_1 = secflp(secure_sig_exp_pair)
        a = mpc.input(non_cachable_secflp_1, senders=0)

        abs_a = abs(a)
        assert not await abs_a.has_cached_is_zero()
        abs_abs_a = abs(abs_a)
        # The assert below awaits _value_computed_externally
        assert not await abs_abs_a.has_cached_is_zero()
        # r2 has now awaited _value_computed_externally. The call below sets the event, but this
        # doesn't penetrate through into r2.
        abs_a.is_zero()
        assert await abs_a.has_cached_is_zero()
        assert not await abs_abs_a.has_cached_is_zero()
        assert await mpc.output(abs_abs_a.is_zero()) == 0


async def test_is_zero_doesnt_penetrate_if_not_possible_2() -> None:
    """
    Assert that CachedMPyCAttributes do not penetrate into placeholder objects if the attribute is
    set after the _value_computed_externally (type Future[Event]) is awaited, and that the program
    doesn't crash if the event is changed after the future is awaited.

    Under test: intermediate call to `output`.
    """
    async with mpc:
        sig_, exp_ = secflp.find_floating_point_representation(
            1, secflp.significand_bit_length
        )
        secure_sig_exp_pair = (
            secflp.significand_class(sig_),
            secflp.exponent_class(exp_),
        )
        non_cachable_secflp_1 = secflp(secure_sig_exp_pair)
        a = mpc.input(non_cachable_secflp_1, senders=0)

        abs_a = abs(a)
        assert not await abs_a.has_cached_is_zero()
        abs_abs_a = abs(abs_a)
        # The await below also awaits _value_computed_externally
        await mpc.output(abs_abs_a)
        abs_a.is_zero()
        assert await abs_a.has_cached_is_zero()
        assert not await abs_abs_a.has_cached_is_zero()
        assert await mpc.output(abs_abs_a.is_zero()) == 0


async def all_tests() -> None:
    """
    Run all test functions.
    """
    await test_input_output()
    await test_secure_bit_length()
    await test_basic_functions()
    await test_arithmetic()
    await test_repeated_operations()
    await test_algebraic_functions()
    await test_transcendental_functions()
    await test_is_zero_penetrates_if_possible()
    await test_is_zero_doesnt_penetrate_if_not_possible_1()
    await test_is_zero_doesnt_penetrate_if_not_possible_2()


if __name__ == "__main__":
    mpc.run(all_tests())
