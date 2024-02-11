import jax
import jax.numpy as jnp
from jaxfin.price_engine.black_scholes import european_price

# Just testing jaxfin
if __name__ == '__main__':
    spots = jnp.asarray([100])
    strikes = jnp.asarray([110])
    expires = jnp.asarray([1.0])
    vols = jnp.asarray([0.2])
    discount_rates = jnp.asarray([0.0])
    print(european_price(spots, strikes, expires, vols, discount_rates, dtype=jnp.float32))