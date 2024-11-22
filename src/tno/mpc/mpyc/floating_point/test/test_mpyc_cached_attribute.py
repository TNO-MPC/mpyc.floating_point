"""
Test the MPyC cached attribute class.
"""

import pytest
from mpyc.sectypes import SecureInteger

from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point._mpyc_cached_attribute import CachedMPyCAttribute

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("mpyc_runtime")]

secint = flp.mpc.SecInt()


async def test_compute_function_computes_expected_value_when_uninitialized() -> None:
    """
    Validate that the compute function returns the computed value if the value was not initialized.
    """

    def compute_func() -> SecureInteger:
        """
        Dummy computation.

        :return: 42
        """
        return secint(42)

    attribute = CachedMPyCAttribute(secint, compute_func=compute_func, runtime=flp.mpc)
    attribute.set(None)
    assert await flp.mpc.output(attribute.compute()) == 42


async def test_compute_function_returns_consistent_cached_value() -> None:
    """
    Validate that the cached value is the same as the computed value.
    """

    def compute_func() -> SecureInteger:
        """
        Dummy computation.

        :return: 42
        """
        return secint(42)

    attribute = CachedMPyCAttribute(secint, compute_func=compute_func, runtime=flp.mpc)
    attribute.set(None)
    for _ in range(5):
        assert await flp.mpc.output(attribute.compute()) == 42


async def test_compute_function_is_called_once() -> None:
    """
    Validate that the compute function is called once.
    """
    nr_calls = 0

    def compute_func() -> SecureInteger:
        """
        Dummy computation that tracks the number of calls.

        :return: 42
        """
        nonlocal nr_calls
        nr_calls += 1
        return secint(42)

    attribute = CachedMPyCAttribute(secint, compute_func=compute_func, runtime=flp.mpc)
    attribute.set(None)
    for _ in range(5):
        await flp.mpc.output(attribute.compute())
    assert nr_calls == 1


async def test_compute_function_not_called_when_value_was_initialized() -> None:
    """
    Validate that the compute function returns the initialized value.
    """

    def compute_func() -> SecureInteger:
        """
        Dummy computation.

        :return: 42
        """
        return secint(42)

    attribute = CachedMPyCAttribute(secint, compute_func=compute_func, runtime=flp.mpc)
    value = secint(3)
    attribute.set(value)
    assert await flp.mpc.output(attribute.compute()) == 3


async def test_compute_function_is_not_called_when_value_was_initialized() -> None:
    """
    Validate that the compute function is never called if the value was initialized.
    """
    nr_calls = 0

    def compute_func() -> SecureInteger:
        """
        Dummy computation that tracks the number of calls.

        :return: 42
        """
        nonlocal nr_calls
        nr_calls += 1
        return secint(42)

    attribute = CachedMPyCAttribute(secint, compute_func=compute_func, runtime=flp.mpc)
    value = secint(3)
    attribute.set(value)
    for _ in range(5):
        await flp.mpc.output(attribute.compute())
    assert nr_calls == 0
