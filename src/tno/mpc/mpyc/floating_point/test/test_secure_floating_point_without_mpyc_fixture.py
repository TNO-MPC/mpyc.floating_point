"""
Testing module for the implementation of secure floating points
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tno.mpc.mpyc.floating_point import (
    SYSTEM_SIGNIFICAND_BITS,
    SecFlp,
    SecureFloatingPoint,
)
from tno.mpc.mpyc.floating_point.test.util import MAX_EXPONENT, MAX_POSITIVE_DOUBLE


@pytest.mark.parametrize(
    "to_convert, significand_bit_length, answer",
    [
        (
            MAX_POSITIVE_DOUBLE,
            SYSTEM_SIGNIFICAND_BITS,
            (
                2**SYSTEM_SIGNIFICAND_BITS - 1,
                MAX_EXPONENT - SYSTEM_SIGNIFICAND_BITS + 1,
            ),
        ),
        (0.00001, 32, (2814749767, -48)),
        (-0.00001, 32, (-2814749767, -48)),
        (0.001, 3, (4, -12)),
        (-0.001, 3, (-4, -12)),
        (0.1, 6, (51, -9)),
        (-0.1, 6, (-51, -9)),
        (2, 6, (32, -4)),
        (10, 6, (40, -2)),
        (-10, 6, (-40, -2)),
        (100, 10, (800, -3)),
        (-100, 10, (-800, -3)),
    ],
)
def test_floating_point_representation(
    to_convert: float, significand_bit_length: int, answer: tuple[int, int]
) -> None:
    """
    Test whether the floating point conversion functionality converts floating points into the correct representation.

    :param to_convert: floating point to be converted.
    :param significand_bit_length: maximum bit length of the significand.
    :param answer: tuple containing the correct conversion.
    """
    significand, exponent = SecureFloatingPoint.find_floating_point_representation(
        to_convert, significand_bit_length
    )

    true_significand, true_exponent = answer
    assert significand == true_significand
    assert exponent == true_exponent


def test_floating_point_output_conversion_when_input_too_large_then_raise_valueerror() -> (
    None
):
    """
    Verify that output_conversion raises a ValueError if the provided input is too large.
    """
    with pytest.raises(ValueError):
        SecFlp().output_conversion(sign=1, exp=2**100)


def test_multiplayer_script() -> None:
    """
    Run a script with three players (-M 3) to detect some issues that only surface in the
    multi-party setting.

    To debug this test, please look at the module docstring of the script under test.
    """
    test_dir = Path(__file__).parents[0]
    multiplayer_test_mod = test_dir / "multiplayer_script.py"
    subprocess.run(
        ["python", str(multiplayer_test_mod), "-M", "3"], timeout=180, check=True
    )
