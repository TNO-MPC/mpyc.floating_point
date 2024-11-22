"""
Module that provides secure integer division functionalities.
"""

# pylint: disable=invalid-name

from __future__ import annotations

from typing import cast

from mpyc.sectypes import SecureInteger

from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point import mpc_coro_extended, returnType


# Almost verbatim copy of mpyc.secgroups._divmod without asserts (mpyc==0.8)
# See https://ci.tno.nl/gitlab/pet/lab/mpc/python-packages/microlibs/mpyc/microlibs/floating-point/-/issues/29 and https://github.com/lschoe/mpyc/issues/77
@mpc_coro_extended
async def secure_divmod(
    a: SecureInteger, b: SecureInteger
) -> tuple[SecureInteger, SecureInteger]:
    """
    Secure integer division divmod(a, b) via NR.

    :param a: Numerator.
    :param b: Denominator.
    :return: Tuple with secure results of integer division (a // b) and remainder (a % b).
    """
    secint = type(a)
    await returnType(secint, 2)
    secfxp = flp.mpc.SecFxp(2 * cast(int, secint.bit_length) + 2)
    a1, b1 = flp.mpc.convert([a, b], secfxp)
    q = a1 / b1
    q_int = flp.mpc.convert(q, secint)
    r = a - b * q_int
    q_int, r = (r < 0).if_else([q_int - 1, r + b], [q_int, r])  # correction using one <
    return q_int, r


def secret_div(a: SecureInteger, b: SecureInteger) -> SecureInteger:
    """
    Secure integer division.

    :param a: Numerator.
    :param b: Denominator.
    :return: Secure result of integer division a // b.
    """
    return secure_divmod(a, b)[0]
