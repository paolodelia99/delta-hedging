from jax import jit
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

from scipy.integrate import quad

from functools import partial


def _Q1(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return jnp.real((jnp.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.0000000000001j))

    u_values = jnp.linspace(1e-15, right_lim, num=1000)
    integral = trapezoid(integrand(u_values), u_values)
    
    return 1 / 2 + 1 / jnp.pi * integral


def _Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """

    def integrand(u):
        return jnp.real(jnp.exp(-u * k * 1j) / (u * 1j) * cf(u))

    # Trapezoid integration
    u_values = jnp.linspace(1e-15, right_lim, num=1000)
    integral = trapezoid(integrand(u_values), u_values)
    
    return 1 / 2 + 1 / jnp.pi * integral

def _compute_probabilities(k, cf, right_lim, integrand_fn):
    # Trapezoid integration
    u_values = jnp.linspace(1e-15, right_lim, num=1000)
    integral = trapezoid(integrand_fn(u_values), u_values)
    
    return 1 / 2 + 1 / jnp.pi * integral

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


def integrand_q1(u, k, cf):
    return jnp.real((jnp.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.0000000000001j))
    
def integrand_q2(u, k, cf):
    return jnp.real(jnp.exp(-u * k * 1j) / (u * 1j) * cf(u))

@jit
def fourier_inv_call_1(s0, K, T, v0, mu, theta, sigma, kappa, rho):
    """
    Price of a call option using the Fourier inversion method
    """
    cf = partial(cf_heston, t=T, v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    right_lim = 1000
    k = jnp.log(K / s0)

    _integrand_q1 = partial(integrand_q1, k=k, cf=cf)
    _integrand_q2 = partial(integrand_q2, k=k, cf=cf)

    q1 = _compute_probabilities(k, cf, right_lim, _integrand_q1)
    q2 = _compute_probabilities(k, cf, right_lim, _integrand_q2)
    return s0 * q1 - K * jnp.exp(- mu * T) * q2

@jit
def delta_call(s0, K, T, v0, mu, theta, sigma, kappa, rho):
    cf = partial(cf_heston, t=T, v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    right_lim = 1000
    k = jnp.log(K / s0)

    _integrand_q1 = partial(integrand_q1, k=k, cf=cf)

    return _compute_probabilities(k, cf, right_lim, _integrand_q1)


def fourier_inv_put(s0, K, T, v0, mu, theta, sigma, kappa, rho):
    """
    Price of a put option using the Fourier inversion method
    """
    cf = partial(cf_heston, t=T, v0=v0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    right_lim = 1000
    k = jnp.log(K / s0)
    return K * jnp.exp(- mu * T) * (1 - Q2(k, cf, right_lim)) - s0 * (1 - Q1(k, cf, right_lim))
