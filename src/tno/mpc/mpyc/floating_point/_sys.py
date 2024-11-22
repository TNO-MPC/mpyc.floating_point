"""
Module with utility functionality for Secure Floating Point Numbers.
"""

from __future__ import annotations

import math
import sys

SYSTEM_EXPONENT_BITS = int(math.log2(sys.float_info.max_exp)) + 1
SYSTEM_SIGNIFICAND_BITS = sys.float_info.mant_dig
