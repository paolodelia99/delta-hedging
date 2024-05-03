import time

import jax.numpy as jnp
from jax import vmap
from black_scholes import black_scholes_call_price
from jax import random
from jaxfin.models.gbm import UnivGeometricBrownianMotion

SEED = 0

black_scholes_call_price_v = vmap(black_scholes_call_price)

if __name__ == "__main__":
    key = random.PRNGKey(SEED)

    dtype = jnp.float32
    s0 = 100
    mean = 0.0
    sigma = 0.2
    maturity = 1.0
    sigmas = jnp.full((252, 100), sigma, dtype=dtype)
    maturities = jnp.full((252, 100), maturity, dtype=dtype)
    N = 252
    n_sim = 100
    dt = 1 / N
    K = 120
    rate = 0.0
    maturities_ = jnp.arange(maturity, 0, -dt)
    Ks = jnp.full((252, 100), 120, dtype=dtype)
    rates = jnp.full((252, 100), rate, dtype=dtype)

    gmb = UnivGeometricBrownianMotion(s0, mean, sigma, dtype=jnp.float32)

    stock_paths = gmb.simulate_paths(SEED, maturity, N, n_sim)

    start = time.time()

    res = black_scholes_call_price(stock_paths, Ks, rates, sigmas, maturities)

    end = time.time()

    print(f"Matrix args time: {end - start}")

    start = time.time()

    res_1 = [
        black_scholes_call_price(stock_paths[i], K, rate, sigma, maturity - i * dt)
        for i in range(0, N)
    ]

    end = time.time()

    print(f"list comprehension: {end - start}")
