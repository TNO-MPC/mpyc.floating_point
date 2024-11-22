"""
Root imports for the tno.mpc.mpyc.floating_point package.
"""

# Explicit re-export of all functionalities, such that they can be imported properly. Following
# https://www.python.org/dev/peps/pep-0484/#stub-files and
# https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport
from tno.mpc.mpyc.floating_point._rt import set_runtime as set_runtime
from tno.mpc.mpyc.floating_point._sys import (
    SYSTEM_EXPONENT_BITS as SYSTEM_EXPONENT_BITS,
)
from tno.mpc.mpyc.floating_point._sys import (
    SYSTEM_SIGNIFICAND_BITS as SYSTEM_SIGNIFICAND_BITS,
)
from tno.mpc.mpyc.floating_point.basic_functions import log_flp as log_flp
from tno.mpc.mpyc.floating_point.basic_functions import log_fxp as log_fxp
from tno.mpc.mpyc.floating_point.basic_functions import natural_log as ln_fxp
from tno.mpc.mpyc.floating_point.basic_functions import sqrt_flp as sqrt_flp
from tno.mpc.mpyc.floating_point.basic_functions import sqrt_fxp as sqrt_fxp
from tno.mpc.mpyc.floating_point.extended_coroutine import (
    mpc_coro_extended as mpc_coro_extended,
)
from tno.mpc.mpyc.floating_point.extended_coroutine import returnType as returnType
from tno.mpc.mpyc.floating_point.secure_bit_length import bit_length as bit_length
from tno.mpc.mpyc.floating_point.secure_bit_length import (
    most_significant_bit as most_significant_bit,
)
from tno.mpc.mpyc.floating_point.secure_floating_point import SecFlp as SecFlp
from tno.mpc.mpyc.floating_point.secure_floating_point import (
    SecureFloatingPoint as SecureFloatingPoint,
)

__version__ = "0.13.0"
