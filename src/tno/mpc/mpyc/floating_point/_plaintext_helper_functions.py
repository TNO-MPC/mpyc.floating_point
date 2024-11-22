"""
Module that provides useful plaintext helper functions.
"""

from __future__ import annotations

import functools
from math import prod

from mpyc.gmpy import invert


def _lagrange_basis_int_poly(i: int, degree: int, prime: int) -> list[int]:
    """
    Helper function that computes the coefficients of the ith Lagrange basis, i.e the
    polynomial l_i that has the property that l_i(m) = 0 if m!=i and l_i(i) = 1.

    :param i: The lagrange basis l_i.
    :param degree: The degree k the resulting polynomial resulting from k+1 data values.
    :param prime: The prime of the finite field
    :return: List of polynomial coefficients.  The coefficients go from degree zero upward.
        e.g. coefficients[i] correspond to the term of degree i.
    """

    def numerator() -> list[int]:
        """
        Helper function that computes the coefficients of the numerator.

        Numerator: x * (x - 1) * (x - 2) * .. * (x - degree)

        :return: List of polynomial coefficients. The coefficients go from degree zero upward.
        e.g. coefficients[i] correspond to the term of degree i.
        """
        coefficients = [0] * (degree + 1)
        coefficients[0] = 1

        for j in range(1, degree + 1):
            if j == i:
                continue
            for m in range(j, 0, -1):
                coefficients[m] = (coefficients[m] - j * coefficients[m - 1]) % prime
        return list(reversed(coefficients))

    def denominator() -> int:
        """
        Helper function that computes the value of the denominator.

        Denominator: prod_{m !=i} (i - m)

        :return: value of denominator
        """
        return prod(i - m for m in range(degree + 1) if m != i) % prime

    inverse_denominator = invert(denominator(), prime)
    coefficients = numerator()
    return [
        int(coefficient * inverse_denominator) % prime for coefficient in coefficients
    ]


@functools.lru_cache
def lagrange_int_poly(values: tuple[int], prime: int) -> list[int]:
    """
    Function that computes the coefficients of a polynomial q(x), that has the property
    that q(i) = values[i] for integers i, using Lagrange interpolation over a finite field.

    :param values: The (integer) y-values of the polynomial evaluated at integer values
    :param prime: The prime of the finite field.
    :return: List of polynomial coefficients. The coefficients go from degree zero upward.
        e.g. coefficients[i] correspond to the term of degree i.
    """
    degree = len(values) - 1

    coefficients = [0] * (degree + 1)
    for j, value in enumerate(values):
        coefficients_basis_j = _lagrange_basis_int_poly(j, degree, prime)
        for m, coef_basis in enumerate(reversed(coefficients_basis_j)):
            coefficients[m] = (coefficients[m] + value * coef_basis) % prime
    return list(reversed(coefficients))


def calculate_two_to_exponent_mod(exponent: int, modulus: int) -> int:
    """
    Compute the power of 2 of the exponent modulo some modulus.

    :param exponent: Integer representing the exponent value.
    :param modulus: Modulus.
    :return: An integer representing the power of 2 of the exponent modulo some modulus.
    """
    if exponent >= 0:
        return int(2**exponent) % modulus
    return int(invert(int(2 ** abs(exponent)), modulus))
