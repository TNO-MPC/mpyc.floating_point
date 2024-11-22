"""
Module that provides secure truncation for secure integers
"""

from __future__ import annotations

from asyncio import Future
from collections.abc import Awaitable
from typing import TypeVar, no_type_check, overload

from mpyc.finfields import PrimeFieldElement
from mpyc.sectypes import SecureObject

from tno.mpc.mpyc.floating_point import _rt as flp
from tno.mpc.mpyc.floating_point import mpc_coro_extended, returnType

SecureObjectT = TypeVar("SecureObjectT", bound=SecureObject)

# Verbatim copy of mpyc.runtime.Runtime.trunc (commit ba3c5723908a7f026451d38b653e19f7891d2065).
# The updated functionality will be available in MPyC > v0.10.0.
# See https://github.com/lschoe/mpyc/issues/88


@overload
def trunc(
    x: list[SecureObjectT], f: int | None = ..., l: int | None = ...
) -> list[SecureObjectT]: ...
@overload
def trunc(
    x: SecureObjectT, f: int | None = ..., l: int | None = ...
) -> SecureObjectT: ...
@overload
def trunc(
    x: PrimeFieldElement, f: int | None = ..., l: int | None = ...
) -> Awaitable[PrimeFieldElement]: ...
@overload
def trunc(
    x: list[PrimeFieldElement], f: int | None = ..., l: int | None = ...
) -> Awaitable[list[PrimeFieldElement]]: ...


@no_type_check
@mpc_coro_extended
async def trunc(
    x: SecureObjectT | list[SecureObjectT], f: int | None = None, l: int | None = None
) -> SecureObjectT | list[SecureObjectT]:
    """Secure truncation of f least significant bits of (elements of) x.

    Probabilistic rounding of a / 2**f for a in x.
    """
    x_is_list = isinstance(x, list)
    if not x_is_list:
        x = [x]
    n = len(x)
    sftype = type(x[0])  # all elts assumed of same type
    if issubclass(sftype, flp.mpc.SecureObject):
        if x_is_list:
            await returnType(sftype, n)
        else:
            await returnType(sftype)
        Zp = sftype.field
        l = l or sftype.bit_length
        if f is None:
            f = sftype.frac_length
        if issubclass(sftype, flp.mpc.SecureFixedPoint):
            l += f
    else:
        await returnType(Future)
        Zp = sftype

    k = flp.mpc.options.sec_param
    r_bits = await flp.mpc.random_bits(Zp, f * n)
    r_modf = [None] * n
    for j in range(n):
        s = 0
        for i in range(f - 1, -1, -1):
            s <<= 1
            s += r_bits[f * j + i].value
        r_modf[j] = Zp(s)
    r_divf = flp.mpc._randoms(Zp, n, 1 << k + l - f)
    if flp.mpc.options.no_prss:
        r_divf = await r_divf
    if issubclass(sftype, flp.mpc.SecureObject):
        x = await flp.mpc.gather(x)
    c = await flp.mpc.output(
        [
            a + ((1 << l - 1) + (q.value << f) + r.value)
            for a, q, r in zip(x, r_divf, r_modf)
        ]
    )
    c = [c.value % (1 << f) for c in c]
    y = [(a - c + r.value) >> f for a, c, r in zip(x, c, r_modf)]
    if not x_is_list:
        y = y[0]
    return y
