import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jaxfin.price_engine.math import cum_normal
from jaxfin.price_engine.utils import cast_arrays


@jit
def compute_d1(spots_1, spots_2, sigma_sqrt_t):
    return (jnp.log(spots_2 / spots_1) / sigma_sqrt_t) + 0.5 * sigma_sqrt_t


@jit
def compute_d1_d2(spots_1, spots_2, expires, sigma_1, sigma_2, corr):
    sigma = jnp.sqrt(jnp.sum(jnp.asarray([sigma_1, sigma_2]) ** 2) - 2 * corr * jnp.prod(jnp.asarray([sigma_1, sigma_2])))

    sigma_sqrt_t = sigma * jnp.sqrt(expires)
    d1 = compute_d1(spots_1, spots_2, sigma_sqrt_t)
    d2 = d1 - sigma_sqrt_t

    return d1, d2


def margrabe(
        spots_1: jax.Array,
        spots_2: jax.Array,
        expires: jax.Array,
        sigma_1: jax.Array,
        sigma_2: jax.Array,
        corr: jax.Array,
        exchange_first: bool = False,
        dtype: jnp.dtype = None
):
    """

    :param spots_1:
    :param spots_2:
    :param expires:
    :param sigma_1:
    :param sigma_2:
    :param corr:
    :param exchange_first:
    :param dtype:
    :return:
    """
    [spots_1, spots_2, expires, sigma_1, sigma_2, corr] = cast_arrays(
        [spots_1, spots_2, expires, sigma_1, sigma_2, corr], dtype
    )

    if exchange_first:
        _spots_1 = spots_2
        _spots_2 = spots_1
    else:
        _spots_1 = spots_1
        _spots_2 = spots_2

    d1, d2 = compute_d1_d2(_spots_1, _spots_2, expires, sigma_1, sigma_2, corr)

    return _spots_2 * cum_normal(d1) - _spots_1 * cum_normal(d2)

def margrabe_deltas(spots_1, spots_2, expires, sigma_1, sigma_2, corr, exchange_first=False, dtype: jnp.dtype = None):
    """

    :param spots_1:
    :param spots_2:
    :param expires:
    :param vols:
    :param corr:
    :param exchange_first:
    :param dtype:
    :return:
    """
    delta_1 = grad(margrabe, argnums=0)(spots_1, spots_2, expires, sigma_1, sigma_2, corr, exchange_first, dtype)
    delta_2 = grad(margrabe, argnums=1)(spots_1, spots_2, expires, sigma_1, sigma_2, corr, exchange_first, dtype)

    return delta_1, delta_2

def margrabe_gammas(spots_1, spots_2, expires, sigma_1, sigma_2, corr, exchange_first=False, dtype: jnp.dtype = None):
    """

    :param spots_1:
    :param spots_2:
    :param expires:
    :param vols:
    :param corr:
    :param exchange_first:
    :param dtype:
    :return:
    """
    gamma_1 = grad(grad(margrabe, argnums=0))(spots_1, spots_2, expires, sigma_1, sigma_2, corr, exchange_first, dtype)
    gamma_2 = grad(grad(margrabe, argnums=1))(spots_1, spots_2, expires, sigma_1, sigma_2, corr, exchange_first, dtype)

    return gamma_1, gamma_2

def margrabe_cross_gamma(spots_1, spots_2, expires, sigma_1, sigma_2, corr, exchange_first=False, dtype: jnp.dtype = None):
    """

    :param spots_1:
    :param spots_2:
    :param expires:
    :param vols:
    :param corr:
    :param exchange_first:
    :param dtype:
    :return:
    """
    cross_gamma_1 = grad(grad(margrabe, argnums=0), argnums=1)(spots_1, spots_2, expires, sigma_1, sigma_2, corr, exchange_first, dtype)
    cross_gamma_2 = grad(grad(margrabe, argnums=1), argnums=0)(spots_1, spots_2, expires, sigma_1, sigma_2, corr, exchange_first, dtype)

    return cross_gamma_1, cross_gamma_2


if __name__ == '__main__':
    vmap_margrabe = vmap(margrabe, in_axes=(0, 0, 0, 0, 0, 0, None, None))

    dtype = jnp.float32
    spots_1 = jnp.asarray(100, dtype=dtype)
    spots_2 = jnp.asarray(120, dtype=dtype)
    expires = jnp.asarray(1.0, dtype=dtype)
    sigma_1 = jnp.asarray(0.2, dtype=dtype)
    sigma_2 = jnp.asarray(0.3, dtype=dtype)
    corr = jnp.asarray(0.7, dtype=dtype)

    spread_opt_price = margrabe(spots_1, spots_2, expires, sigma_1, sigma_2, corr, dtype=dtype)
    delta_1, delta_2 = margrabe_deltas(spots_1, spots_2, expires, sigma_1, sigma_2, corr, dtype=dtype)
    gamma_1, gamma_2 = margrabe_gammas(spots_1, spots_2, expires, sigma_1, sigma_2, corr, dtype=dtype)
    cross_gamma_1, cross_gamma_2 = margrabe_cross_gamma(spots_1, spots_2, expires, sigma_1, sigma_2, corr, dtype=dtype)

    print('Paraneters')
    print(f'Spot 1: {spots_1}')
    print(f'Spot 2: {spots_2}')
    print(f'Expires: {expires}')
    print(f'Sigma 1: {sigma_1}')
    print(f'Sigma 2: {sigma_2}')
    print(f'Corr: {corr}')

    print(f'Margrabre price: {spread_opt_price}')
    print('Greeks:')
    print(f'Delta 1: {delta_1}')
    print(f'Delta 2: {delta_2}')
    print(f'Gamma 1: {gamma_1}')
    print(f'Gamma 2: {gamma_2}')
    print(f'Cross Gamma 1: {cross_gamma_1}')
    print(f'Cross Gamma 2: {cross_gamma_2}')
    print('\n')

    spots_1 = jnp.asarray([110], dtype=dtype)
    spots_2 = jnp.asarray([120], dtype=dtype)
    expires = jnp.asarray([1.0], dtype=dtype)
    sigma_1 = jnp.asarray([0.3], dtype=dtype)
    sigma_2 = jnp.asarray([0.3], dtype=dtype)
    corr = jnp.asarray([0.7], dtype=dtype)

    spread_opt_price = margrabe(spots_1, spots_2, expires, sigma_1, sigma_2, corr, dtype=dtype)

    print(f'Margrabre price: {spread_opt_price}')

    # Try the vmap version
    print('Try the vmap version')
    spots_1 = jnp.asarray([110, 100], dtype=dtype)
    spots_2 = jnp.asarray([120, 120], dtype=dtype)
    expires = jnp.asarray([1.0, 1.0], dtype=dtype)
    sigma_1 = jnp.asarray([0.3, 0.2], dtype=dtype)
    sigma_2 = jnp.asarray([0.3, 0.3], dtype=dtype)
    corr = jnp.asarray([0.7, 0.7], dtype=dtype)

    spread_opt_price = vmap_margrabe(spots_1, spots_2, expires, sigma_1, sigma_2, corr, False, dtype)

    print(f'Margrabre price: {spread_opt_price}')