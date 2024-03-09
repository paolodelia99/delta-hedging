from jax import jit
import jax.numpy as jnp

from scipy.integrate import quad

from functools import partial


def Q1(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return jnp.real((jnp.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.0000000000001j))

    return 1 / 2 + 1 / jnp.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]


def Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return jnp.real(jnp.exp(-u * k * 1j) / (u * 1j) * cf(u))

    return 1 / 2 + 1 / jnp.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]


def cf_heston(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function
    """
    xi = kappa - sigma * rho * u * 1j
    d = jnp.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    g2 = 1 / g1
    cf = jnp.exp(
        1j * u * mu * t
        + (kappa * theta) / (sigma**2) * ((xi - d) * t - 2 * jnp.log((1 - g2 * jnp.exp(-d * t)) / (1 - g2)))
        + (v0 / sigma**2) * (xi - d) * (1 - jnp.exp(-d * t)) / (1 - g2 * jnp.exp(-d * t))
    )
    return cf


def fourier_inv_call(s0, K, T, v0, mu, theta, sigma, kappa, rho):
    """
    Price of a call option using the Fourier inversion method
    """
    cf = partial(cf_heston, t=T, v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    right_lim = 1000
    k = jnp.log(K / s0)
    return s0 * Q1(k, cf, right_lim) - K * jnp.exp(- mu * T) * Q2(k, cf, right_lim)


def fourier_inv_put(s0, K, T, v0, mu, theta, sigma, kappa, rho):
    """
    Price of a put option using the Fourier inversion method
    """
    cf = partial(cf_heston, t=T, v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    right_lim = 1000
    k = jnp.log(K / s0)
    return K * jnp.exp(- mu * T) * (1 - Q2(k, cf, right_lim)) - s0 * (1 - Q1(k, cf, right_lim))
