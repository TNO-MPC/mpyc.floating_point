"""
Tests to verify the correctness of the implemented plaintext helper functions
"""

from __future__ import annotations

from random import randint

from tno.mpc.mpyc.floating_point._plaintext_helper_functions import (
    _lagrange_basis_int_poly,
    lagrange_int_poly,
)
from tno.mpc.mpyc.floating_point.test.util import assert_equals_mod


def _plaintext_poly_approximation(
    coefficients: list[int], value: int, prime: int | None = None
) -> int:
    """
    Evaluate the given polynomial at a certain point.

    :param coefficients: List of polynomial coefficients. The coefficients go from degree zero upward.
        e.g. coefficients[i] correspond to the term of degree i.
    :param value: Value to be evaluated.
    :param prime: If provided, evaluate the polynomial modulo `prime`.
    :return: The polynomial evaluation of value with respect to the coefficients.
    """
    result = 0
    for k, coef in enumerate(coefficients):
        result += coef * value**k
        if prime is not None:
            result %= prime
    return result


def test_lagrange_basis_int_poly() -> None:
    """
    Test whether the lagrange basis polynomial coefficients produce the correct
    Kronecker delta functionality.
    """
    degree = 10
    prime = 441078571883151574149311450471
    for i in range(degree + 1):
        # Obtain coefficients for ith Lagrange basis polynomial L_i
        coefficients = _lagrange_basis_int_poly(i, degree, prime)

        # Assert that L_i(j) = delta_{i,j}
        for j in range(degree + 1):
            result = _plaintext_poly_approximation(coefficients, value=j, prime=prime)
            assert result == int(i == j)


def test_lagrange_int_poly() -> None:
    """
    Test whether evaluating lagrange int polynomial reproduces input.
    """
    prime = 441078571883151574149311450471

    values = tuple(randint(0, prime) for _ in range(50))
    coefficients = lagrange_int_poly(values, prime)

    for i, value in enumerate(values):
        result = _plaintext_poly_approximation(coefficients, i)
        assert_equals_mod(result, value, prime)
