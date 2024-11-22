"""
Module that extends the functions available for secure integers, fixed-point numbers and floating-point numbers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypeVar, cast

from mpyc.sectypes import SecureFixedPoint, SecureInteger, SecureNumber

from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point.extended_coroutine import mpc_coro_extended, returnType
from tno.mpc.mpyc.floating_point.secure_bit_length import bit_length

if TYPE_CHECKING:
    from tno.mpc.mpyc.floating_point.secure_floating_point import SecureFloatingPoint


SecureNumberT = TypeVar("SecureNumberT", bound=SecureNumber)


# region Logarithm
@mpc_coro_extended
async def natural_log(value: SecureFixedPoint) -> SecureFixedPoint:
    """
    Secure natural logarithm of a secure fixed point.

    :param value: Secure fixed point number
    :return: Secure fixed point number containing the natural logarithm of the input
    """
    await returnType(
        type(value)
    )  # return temporary dummy of the same type as the input
    mult_factor, bit_length = anorm(value, apply_sign=False)
    small_log = ln_small(value * mult_factor)
    return small_log + bit_length * math.log(2)


@mpc_coro_extended
async def log_fxp(value: SecureFixedPoint, base: int | float) -> SecureFixedPoint:
    """
    Secure logarithm of a secure fixed point number in a public base

    :param base: plaintext base of the logarithm.
    :param value: Secure fixed point number of which the logarithm is taken with respect to the base.
    :return: Secure fixed point number containing the natural logarithm of the input in the given
        base
    """
    await returnType(type(value))
    return natural_log(value) / math.log(base)


@mpc_coro_extended
async def ln_small(value: SecureFixedPoint) -> SecureFixedPoint:
    """
    Secure natural logarithm approximation of the input value in the domain [0.5, 1).

    :param value: Secure fixed point number in the range [0.5, 1)
    :return: secure fixed point containing an approximation of the natural logarithm of the provided
        input value
    """
    await flp.mpc.returnType(
        type(value)
    )  # return temporary dummy of the same type as the input
    coefs = [
        -0.28768207245178,
        1.33333333333333,
        -0.88888888888889,
        0.79012345679012,
        -0.79012345679012,
        0.842798353909465,
        -0.936442615454961,
        1.070220130948527,
        -1.248590153939948,
    ]
    return poly_approximation(coefs, value - 0.75)


@mpc_coro_extended
async def log_flp(
    secure_flp: SecureFloatingPoint, secfxp_type: type[SecureFixedPoint]
) -> SecureFloatingPoint:
    """
    Base-2 logarithm of a secure floating point number. This method abuses the structure of secure
    floating point numbers to obtain a more efficient solution than the Taylor approximation.
    During this process, the significand needs to be converted from a secure integer to a secure
    fixed point to enable a base-2 logarithm subroutine for fixed points.

    :param secure_flp: Secure floating point number.
    :param secfxp_type: Secure fixed point type that is used in the conversion.
    :return: Secure floating point number containing the base-2 logarithm of the input value.
    :raise ValueError: Significand has smaller bit length than exponent.
    """
    secure_type_flp = type(secure_flp)
    await returnType(secure_type_flp)

    significand_max_bit_length = secure_flp.significand_bit_length
    if significand_max_bit_length < secure_flp.exponent_bit_length:
        raise ValueError(
            "Expected a secure floating point object with larger significand bit length than"
            " exponent bit length, but received"
            f" {significand_max_bit_length}<{secure_flp.exponent_bit_length}."
        )

    significand_fxp = flp.mpc.convert(secure_flp.significand, secfxp_type)
    exponent_fxp = flp.mpc.convert(secure_flp.exponent, secfxp_type)

    significand_log_fxp = log_fxp(significand_fxp, 2)
    new_significand_unscaled_fxp = significand_log_fxp + exponent_fxp

    significand_bit_length = bit_length(
        flp.mpc.abs(
            flp.mpc.convert(new_significand_unscaled_fxp, secure_flp.significand_class)
        ),
        upper_bound=secure_flp.exponent_bit_length
        + math.ceil(
            math.log2(
                1
                + (secure_flp.significand_bit_length - 1)
                * 2**-secure_flp.exponent_bit_length
            )
        ),
    )
    new_exponent_sclass = significand_bit_length - significand_max_bit_length
    new_exponent = flp.mpc.convert(new_exponent_sclass, secure_flp.exponent_class)

    scaling_factor = flp.mpc.convert(
        exp2_int_pos(-new_exponent_sclass, significand_max_bit_length.bit_length()),
        secfxp_type,
    )
    new_significand = flp.mpc.convert(
        new_significand_unscaled_fxp * scaling_factor, secure_flp.significand_class
    )

    # Set significand to zero if input was a secure 1
    res_is_zero = (secure_flp - 1).is_zero()
    new_significand *= 1 - res_is_zero
    return secure_type_flp((new_significand, new_exponent), is_zero=res_is_zero)


# endregion


# region Square Root


@mpc_coro_extended
async def sqrt_fxp(value: SecureFixedPoint) -> SecureFixedPoint:
    """
    Secure square root of the input value.

    :param value: Secure fixed point number
    :return: secure fixed point number containing the square root of the input
    """
    await returnType(
        type(value)
    )  # return temporary dummy of the same type as the input
    value_nonzero = 1 - flp.mpc.is_zero(value)
    mult_factor, bit_length = anorm(value, apply_sign=False)
    bit_length_is_odd = parity(bit_length)
    approximation = sqrt_small(mult_factor * value)
    value_bit_len = cast(int, value.bit_length)
    if_bit_length_even = approximation * exp2_fxp_integral(
        (bit_length / 2), value_bit_len.bit_length()
    )
    if_bit_length_odd = (
        approximation
        * exp2_fxp_integral((bit_length - 1) / 2, value_bit_len.bit_length())
        * math.sqrt(2)
    )
    odd_or_even_result = flp.mpc.if_else(
        bit_length_is_odd, if_bit_length_odd, if_bit_length_even
    )
    return value_nonzero * odd_or_even_result


@mpc_coro_extended
async def sqrt_small(value: SecureFixedPoint) -> SecureFixedPoint:
    """
    Secure approximation of the square root in the domain [0.5,1).

    :param value: Secure fixed point number
    :return: secure fixed point number containing an approximation of the square root of the input
        value
    """
    coefs = [
        0.86602540378,
        0.57735069190,
        -0.19245008973,
        0.12830005982,
        -0.10691671652,
    ]
    return poly_approximation(coefs, value - 0.75)


@mpc_coro_extended
async def sqrt_flp(
    secure_flp: SecureFloatingPoint, secfxp_type: type[SecureFixedPoint]
) -> SecureFloatingPoint:
    """
    Square root of a secure floating point number. This method abuses the structure of secure
    floating point numbers to obtain a more efficient solution than the Taylor approximation.
    During this process, the significand needs to be converted from a secure integer to a secure
    fixed point to enable a square root subroutine for fixed points.

    :param secure_flp: Secure floating point number.
    :param secfxp_type: Secure fixed point type that is used in the conversion.
    :return: Secure floating point number containing the square root of the input value.
    """
    stype = type(secure_flp)
    await returnType(stype)

    significand_max_bit_length = secure_flp.significand_bit_length
    significand_fxp = flp.mpc.convert(secure_flp.significand, secfxp_type)
    exponent_flp = secure_flp.exponent
    parity_exponent_fxp = flp.mpc.convert(parity(exponent_flp), secfxp_type)

    sqrt_input = (1 + parity_exponent_fxp) * significand_fxp
    significand_sqrt_fxp = sqrt_fxp(sqrt_input)

    parity_significand_exponent = parity_exponent_fxp + (
        significand_max_bit_length & 1
    ) * (1 - 2 * parity_exponent_fxp)
    scaling_factor = flp.mpc.convert(
        exp2_fxp_integral(
            (
                significand_max_bit_length
                - (parity_exponent_fxp + parity_significand_exponent)
            )
            // 2,
            significand_max_bit_length.bit_length(),
        ),
        secfxp_type,
    )
    new_significand = flp.mpc.convert(
        significand_sqrt_fxp * scaling_factor, secure_flp.significand_class
    )

    parity_significand_exponent_converted = flp.mpc.convert(
        parity_significand_exponent, secure_flp.exponent_class
    )
    new_exponent = (
        exponent_flp
        + parity_significand_exponent_converted
        - significand_max_bit_length
    ) // 2

    return stype(
        (new_significand, new_exponent),
        is_zero=secure_flp.is_zero() if await secure_flp.has_cached_is_zero() else None,
    )


# endregion


# region Exponentiation


@mpc_coro_extended
async def exp2_fxp_integral(
    exponent: SecureFixedPoint, int_size: int | None = None
) -> SecureFixedPoint:
    """
    Secure base-2 exponentiation of the input.

    :param exponent: exponent of the exponentiation. It is assumed that the
        value is integer. if a non-integer fixed-point number is provided as
        input, there is a chance the wrong answer is returned.
    :param int_size: upper-bound on the bit length of the input value. This
        helps make the calculations more efficient.
    :return: The base-2 exponentation of the exponent.
    """
    frac_bits = exponent.frac_length
    exp_bit_len = cast(int, exponent.bit_length)
    exponent_type = type(exponent)
    int_bits = exp_bit_len - frac_bits
    if int_size is None:
        int_size = int_bits
    await returnType(exponent_type)
    # take the bits of the integer part
    int_n = exponent / 2**frac_bits
    bits = flp.mpc.to_bits(int_n, int_bits)

    # the last bit is the significand bit
    sign = bits[-1]

    pos_result = exponent_type(1)
    neg_result = exponent_type(0.5)

    to_check = min(int_bits - 1, int_size)
    for i in range(to_check):
        pos_result = flp.mpc.if_else(bits[i], pos_result * 2 ** (2**i), pos_result)
        new_neg_result = neg_result / 2 ** (2**i)
        neg_result = flp.mpc.if_else(bits[i], neg_result, new_neg_result)
    result = cast(SecureFixedPoint, flp.mpc.if_else(sign, neg_result, pos_result))
    return result


@mpc_coro_extended
async def exp2_int(
    exponent: SecureInteger, int_size: int | None = None
) -> SecureInteger:
    """
    Secure base-2 exponentiation of the input.

    :param exponent: Exponent of the exponentiation. It is assumed that the value is a secure
        Integer. If the value is negative, the multiplicative inverse of 2**(-exponent) is
        calculated.
    :param int_size: Upper bound on the bit length of the input value. This helps make the
        calculations more efficient.
    :return: The base-2 exponentation of the exponent.
    """
    exponent_type = type(exponent)
    await returnType(exponent_type)

    int_bits_exponent = cast(int, exponent.bit_length)
    bits_to_check = int_bits_exponent
    if int_size is not None:
        bits_to_check = min(bits_to_check, int_size)

    sign = exponent > 0
    pos_exponent = (2 * sign - 1) * exponent
    bits = flp.mpc.to_bits(pos_exponent, bits_to_check)

    result = exponent_type(1)
    two_to_pow = 2
    for i in range(bits_to_check):
        result = flp.mpc.if_else(bits[i], result * two_to_pow, result)
        two_to_pow = pow(two_to_pow, 2, exponent_type.field.modulus)

    # Return the inverse if the exponent is negative
    return flp.mpc.if_else(sign, result, flp.mpc.reciprocal(result))


@mpc_coro_extended
async def exp2_int_pos(
    exponent: SecureInteger, int_size: int | None = None
) -> SecureInteger:
    """
    Secure base-2 exponentiation of the input.

    :param exponent: exponent of the exponentiation. It is assumed that the value is integer and positive.
        if a non-integer fixed-point number is provided as input, there is a chance the wrong answer is returned. If a
        negative value is provided as input, the answer will be wrong.
    :param int_size: upper-bound on the bit length of the input value. This helps make the calculations more efficient.
    :return: The base-2 exponentation of the exponent.
    """

    int_bits_exponent = cast(int, exponent.bit_length)
    exponent_type = type(exponent)
    max_bits_to_check = int_size or int_bits_exponent
    await returnType(exponent_type)
    to_check = min(int(int_bits_exponent.bit_length()), max_bits_to_check)
    bits = flp.mpc.to_bits(exponent, to_check)  # last entry is significand bit

    pos_result = exponent_type(1)
    two_to_pow = 2

    for i in range(to_check):
        pos_result = flp.mpc.if_else(bits[i], pos_result * two_to_pow, pos_result)
        two_to_pow = two_to_pow**2
    result = pos_result
    return result


# endregion


# region Helper Functions


@mpc_coro_extended
async def poly_approximation(
    coefficients: list[int] | list[float], value: SecureNumberT
) -> SecureNumberT:
    """
    Function for securely approximating a certain function using a polynomial

    :param coefficients: List of polynomial coefficients. The coefficients go from degree zero upward.
        e.g. coefficients[i] correspond to the term of degree i.
    :param value: value to be evaluated
    :return: The polynomial evaluation of value with respect to the coefficients.
    :raise TypeError: When the first coefficient is not compatible with the secure type.
    """
    secure_type = type(value)
    await returnType(secure_type)

    try:
        result = secure_type(coefficients[0])
    except TypeError as e:
        raise TypeError(
            f"Type of the first coefficient ({type(coefficients[0])}) is not "
            f"compatible as input for secure type {secure_type}."
        ) from e

    value_pow = value
    for coef in coefficients[1:]:
        result += coef * value_pow
        value_pow *= value
    return result


@mpc_coro_extended
async def integer(value: SecureNumberT) -> SecureNumberT:
    """
    Function that securely truncates a number to its integer part.

    :param value: The secure number.
    :return: the floor function securely applied to value (similar to `int`).
    """
    fractional_length = value.frac_length  # type: ignore
    await returnType(type(value))
    return flp.mpc.from_bits(flp.mpc.to_bits(value, l=None)[fractional_length:])


@mpc_coro_extended
async def parity(value: SecureNumberT) -> SecureNumberT:
    """
    Function that securely calculates the parity of the integer part of the given secure number.

    :param value: The secure number.
    :return: The parity of the integer part of the secure number.
    """
    element_type = type(value)
    if isinstance(value, SecureFixedPoint):
        await returnType((element_type, True))
    else:
        await returnType(element_type)
    fractional_length = cast(int, value.frac_length)  # type: ignore
    par = flp.mpc.to_bits(value, fractional_length + 1)[fractional_length]
    return par


def anorm(
    to_reduce: SecureNumberT, apply_sign: bool = True
) -> tuple[SecureNumberT, SecureNumberT]:
    """
    Secure function that securely calculates 1/2**k, k for k such that
    to_reduce / 2**k is in the domain [0.5, 1)

    :param to_reduce: the number to be normalized.
    :param apply_sign: whether the normalization factor should preserve the sign.
    :return: A tuple containing the secret-shared normalization factor 1/2**k and the secret-shared bit length k.
    :raise ValueError: When the secure type has not been initialized properly.
    """
    bits = flp.mpc.to_bits(to_reduce, l=None)
    most_sign_bit = bits[-1]
    sign = 1 - 2 * most_sign_bit
    bits = bits[:-1]
    secure_type = type(to_reduce)
    one = secure_type(1)

    def __norm(_bits: list[SecureNumberT]) -> list[SecureNumberT]:
        """
        Adapted norm function. This is a helper function that computes the prefix or.

        :param bits: secret-shared bits.
        :return: Three helper values for the final calculation.
        """
        number_of_bits = len(_bits)
        if number_of_bits == 1:
            var_1 = sign * _bits[0] + most_sign_bit
            return [2 - var_1, var_1, 1 - var_1]
        i_0, nz_0, i_2 = __norm(_bits[: number_of_bits // 2])
        i_1, nz_1, var_2 = __norm(_bits[number_of_bits // 2 :])
        i_0 *= 1 << ((number_of_bits + 1) // 2)
        i_2 += (number_of_bits + 1) // 2
        result = flp.mpc.if_else(nz_1, [i_1, one, var_2], [i_0, nz_0, i_2])
        return result

    integer_bits = secure_type.bit_length
    frac_bits = secure_type.frac_length  # type: ignore
    if not integer_bits or not frac_bits:
        raise ValueError(
            f"initialization of the type {secure_type} went wrong. integers_bits or frac_bits has not been set."
        )
    normed_a = __norm(bits)
    result_0 = normed_a[0] * 2 ** (frac_bits - (integer_bits - 1))
    if apply_sign:
        result_0 *= sign
    return (
        result_0,
        (frac_bits - 1) - normed_a[2],
    )


# endregion
