# Benchmarks

This document summarizes performance benchmarks.

## Multiple addition protocol

The `tno.mpc.mpyc.floating_point` package supports addition of two `SecureFloatingPoint` objects as `secflp_1 + secflp_2` as expected. Multiple additions can be performed sequentially in this manner, and as expected the required effort for doing so increases linearly. However, this package also implements a superior multiple-addition protocol that significantly outperforms the sequential addition by reducing the number of bit length computations.

Depending on how many addends are to be combined in the protocol, we need to slightly increase the significand bit length that is used under the hood. This means that when creating a secure floating point type, you also need to specify `max_concurrent_additions` (default value is `2`), see an example below:

```python
from tno.mpc.mpyc.floating_point import SecFlp

sec_flp_type = SecFlp(significand_bit_length=32, exponent_bit_length=16, max_concurrent_additions=4)
```

As a simple benchmark, we ran 100 times for the multiple addition protocol and measured execution times on a single computer (with three parties) for `n=2` up to `n=128` addends (with increasing secret-sharing modulus), and compared it with `n` sequential additions. The considered floating point type (`SecFlp16|8|n|2`) has `significand_bit_length=16`, `exponent_bit_length=8`, `max_concurrent_multiplications=2` and `max_concurrent_additions` either `2` (for sequential additions) or `n` (for combined additions).

![Multiple additions benchmark](./multiple_additions.png)

To benefit from the improved protocol, make sure to sum `SecureFloatingPoint` objects as `SecureFloatingPoint.sum(secflp_1, secflp_2, ...)` or, equivalently, `secflp_1.add(secflp_2, ...)`.

## Multiple multiplication protocol

The `tno.mpc.mpyc.floating_point` package supports multiplication of two `SecureFloatingPoint` objects as `secflp_1 * secflp_2` as expected. Multiple multiplications can be performed sequentially in this manner, and as expected the required effort for doing so increases linearly. However, this package also implements a superior multiple-multiplications protocol that significantly outperforms the sequential multiplication.

Depending on how many multiplicands are to be combined in the protocol, we need to slightly increase the significand bit length that is used under the hood. This means that when creating a secure floating point type, you also need to specify `max_concurrent_multiplications` (default value is `2`), see example below:

```python
from tno.mpc.mpyc.floating_point import SecFlp

sec_flp_type = SecFlp(significand_bit_length=32, exponent_bit_length=16, max_concurrent_multiplications=4)
```

As a simple benchmark, we ran 100 times the multiple multiplication protocol and measured execution times on a single computer (with three parties) for `n=2` up to `n=128` addends (with increasing secret-sharing modulus), and compared it with `n` sequential multiplications. The considered floating point type (`SecFlp16|8|2|n`) has `significand_bit_length=16`, `exponent_bit_length=8`, `max_concurrent_additions=2` and `max_concurrent_multiplications` either `2` (for sequential multiplications) or `n` (for combined multiplications).

![Multiple multiplications benchmark](./multiple_multiplications.png)

To benefit from the improved protocol, make sure to multiply `SecureFloatingPoint` objects as `SecureFloatingPoint.prod(secflp_1, secflp_2, ...)` or, equivalently, `secflp_1.mul(secflp_2, ...)`.
