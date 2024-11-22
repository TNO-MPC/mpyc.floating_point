"""
Module that exports a consistent runtime that is to be used in the entire package.
Customizable so that it can be made consistent with any MPyC runtime (rather than just the default).
"""

from mpyc.runtime import Runtime
from mpyc.runtime import mpc as _mpyc_rt

mpc = _mpyc_rt


def set_runtime(runtime: Runtime) -> None:
    """
    Set the MPyC runtime.

    :param runtime: Runtime object used by the MPyC framework.
    """
    global mpc  # pylint: disable=global-statement
    mpc = runtime
