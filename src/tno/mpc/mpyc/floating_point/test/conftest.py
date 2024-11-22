"""
Pytest fixtures. Configures an asynchronous MPyC runtime for improved bug finding.
"""

from asyncio import BaseEventLoop
from collections.abc import AsyncIterator

import pytest_asyncio
from pytest import MonkeyPatch

from tno.mpc.mpyc.floating_point import _rt as flp


@pytest_asyncio.fixture
async def mpyc_runtime(
    event_loop: BaseEventLoop, monkeypatch: MonkeyPatch
) -> AsyncIterator[None]:
    """
    Fixture that takes care of the MPyC runtime, ensuring that it executes asynchronously.

    :param event_loop: Event loop fixture.
    :param monkeypatch: Monkeypatch fixture.
    :return: Started MPyC environment.
    """
    flp.mpc._loop = event_loop  # pylint: disable=protected-access
    monkeypatch.setattr(flp.mpc.options, "no_async", False)
    # Do not run in `async with flp.mpc` or call `mpc.shutdown`. This will wait for completion of
    # all futures and thereby prevent termination in case of pending exceptions.
    yield
