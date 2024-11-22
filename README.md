# TNO PET Lab - secure Multi-Party Computation (MPC) - MPyC - Floating Point

Implementation of secure computations on floating-points using the MPyC framework. It serves as an alternative to the [`SecureFloat`](https://lschoe.github.io/mpyc/mpyc.sectypes.html#SecureFloat) class that was introduced in [MPyC](https://github.com/lschoe/mpyc) v0.8. A key difference between the two implementations is that we represent the significand of a secure floating-point by a [`SecureInteger`](https://lschoe.github.io/mpyc/mpyc.sectypes.html#SecureInteger), whereas MPyC's `SecureFloat` represents the significand by a (signed) [`SecureFixedPoint`](https://lschoe.github.io/mpyc/mpyc.sectypes.html#SecureFixedPoint). Furthermore, our implementation contains the following functionality:

- More efficient bit length protocol.
- Optimized protocols for performing multiple additions or multiplications (see [here](./benchmark/README.md)).
- Cached attributes for speeding up sequential operations.

Also, there are initial protocols for computing

1. the secure square root and
2. the logarithm

of a secure floating-point. We should note that the accuracy of those protocols is up for improvement. For example, inaccuracies may occur in the logarithm of input that is close to zero or one.

### PET Lab

The TNO PET Lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of PET solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed PET functionalities to boost the development of new protocols and solutions.

The package `tno.mpc.mpyc.floating_point` is part of the [TNO Python Toolbox](https://github.com/TNO-PET).

_Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws._  
_This implementation of cryptographic software has not been audited. Use at your own risk._

## Install

Easily install the `tno.mpc.mpyc.floating_point` package using `pip`:

```console
$ python -m pip install tno.mpc.mpyc.floating_point
```

_Note:_ If you are cloning the repository and wish to edit the source code, be
sure to install the package in editable mode:

```console
$ python -m pip install -e 'tno.mpc.mpyc.floating_point'
```

If you wish to run the tests you can use:

```console
$ python -m pip install 'tno.mpc.mpyc.floating_point[tests]'
```

_Note:_ A significant performance improvement can be achieved by installing the GMPY2 package.

```console
$ python -m pip install 'tno.mpc.mpyc.floating_point[gmpy]'
```

## Basic usage

The following code snippet shows how secure floating points can be initialized and used.

```python
"""
Example usage. This script can be executed with multiple parties as follows:

  $ python example.py -M3

or, in separate terminals,

  $ python example.py -M3 -I0
  $ python example.py -M3 -I1
  $ python example.py -M3 -I2
"""

from math import log2, sqrt

from mpyc.runtime import mpc
from mpyc.sectypes import SecFxp

from tno.mpc.mpyc.floating_point import SecFlp, log_flp, sqrt_flp


async def main():
    # start the MPyC framework
    async with mpc:
        # create a secure floating point type
        sec_flp_type = SecFlp(significand_bit_length=32, exponent_bit_length=16)

        val_1 = 2**15
        val_2 = 1 / 2**10
        # create two (placeholder) secure floating points
        if mpc.pid == 0:
            sec_val_1 = sec_flp_type(val_1)
            sec_val_2 = sec_flp_type(val_2)
        else:
            sec_val_1 = sec_flp_type(None)
            sec_val_2 = sec_flp_type(None)

        # secret-share floating points
        sec_val_1 = mpc.input(sec_val_1, senders=0)
        sec_val_2 = mpc.input(sec_val_2, senders=0)

        # apply regular computation for comparison
        regular_results = [
            val_1 + val_2,  # addition
            val_2 - val_1,  # subtraction
            val_1 / val_2,  # division
            val_1 * val_2,  # multiplication
            sqrt(val_2),  # square root
            log2(val_1),  # logarithm base 2
        ]

        # create a secure fixed point type to use in certain subroutines
        # if the bit length is too small, the subroutines might produce wrong answers
        sec_fxp_type = SecFxp(96)

        # apply secure computations
        sec_results = [
            sec_val_1 + sec_val_2,  # addition
            sec_val_2 - sec_val_1,  # subtraction
            sec_val_1 / sec_val_2,  # division
            sec_val_1 * sec_val_2,  # multiplication
            sqrt_flp(sec_val_2, secfxp_type=sec_fxp_type),  # square root
            log_flp(sec_val_1, secfxp_type=sec_fxp_type),  # logarithm base 2
        ]

        # reconstruct the secret shares
        revealed_results = await mpc.output(sec_results)

    # print the results
    print(regular_results)
    print(revealed_results)


if __name__ == "__main__":
    # if a custom mpyc runtime rt is created, it can be set with tno.mpc.mpyc.floating_point.set_runtime(rt)
    mpc.run(main())
```

The output of this code snippet is:

```commandline
$ python example.py -M 3
2023-11-02 10:28:07,647 Start MPyC runtime v0.8
2023-11-02 10:28:07,752 All 3 parties connected.
2023-11-02 10:28:09,055 Stop MPyC runtime -- elapsed time: 0:00:01.407784
[32768.0009765625, -32767.9990234375, 33554432.0, 32.0, 0.03125, 15.0]
[32768.0009765625, -32767.9990234375, 33554432.0, 32.0, 0.03125576677848585, 15.000011652708054]
```
