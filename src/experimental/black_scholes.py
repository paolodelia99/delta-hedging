import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from typing import List

_SQRT_2 = jnp.sqrt(2.0)

@jax.jit
def black_scholes_call_price(spot, strike, rate, volatility, time_to_maturity):
    """
    Calculates the price of a European vanilla call option using the Black-Scholes model.

    Args:
        spot: Current spot price of the underlying asset.
        strike: Strike price of the option.
        rate: Annual risk-free interest rate (as a decimal).
        volatility: Annualized volatility of the underlying asset (as a decimal).
        time_to_maturity: Time to option maturity in years.

    Returns:
        Price of the call option.
    """
    d1 = (jnp.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_to_maturity) / (
            volatility * jnp.sqrt(time_to_maturity))
    d2 = d1 - volatility * jnp.sqrt(time_to_maturity)

    call_price = spot * jax.scipy.stats.norm.cdf(d1) - strike * jnp.exp(
        -rate * time_to_maturity) * jax.scipy.stats.norm.cdf(d2)
    return call_price

@jax.jit
def black_scholes_put_price(spot, strike, rate, volatility, time_to_maturity):
    """
    Calculates the price of a European vanilla put option using the Black-Scholes model.

    Args:
        spot: Current spot price of the underlying asset.
        strike: Strike price of the option.
        rate: Annual risk-free interest rate (as a decimal).
        volatility: Annualized volatility of the underlying asset (as a decimal).
        time_to_maturity: Time to option maturity in years.

    Returns:
        Price of the put option.
    """
    d1 = (jnp.log(spot / strike) + (rate + 0.5 * volatility ** 2) * time_to_maturity) / (
            volatility * jnp.sqrt(time_to_maturity))
    d2 = d1 - volatility * jnp.sqrt(time_to_maturity)

    put_price = strike * jnp.exp(-rate * time_to_maturity) * jax.scipy.stats.norm.cdf(-d2) - spot * jax.scipy.stats.norm.cdf(-d1)
    return put_price

def bs_price(
    spots: jax.Array,
    strikes: jax.Array,
    expires: jax.Array,
    vols: jax.Array,
    discount_rates: jax.Array,
    are_calls: jax.Array = None,
    dtype: jnp.dtype = None,
) -> jax.Array:
    """
    Compute the option prices for european options using the Black-Scholes model.

    :param spots: (jax.Array): Array of current asset prices.
    :param strikes: (jax.Array): Array of option strike prices.
    :param expires: (jax.Array): Array of option expiration times.
    :param vols: (jax.Array): Array of option volatility values.
    :param discount_rates: (jax.Array): Array of risk-free interest rates. Defaults to None.
    :param dividend_rates: (jax.Array): Array of dividend rates. Defaults to None.
    :param are_calls: (jax.Array): Array of booleans indicating whether options are calls (True) or puts (False).
    :param dtype: (jnp.dtype): Data type of the output. Defaults to None.
    :return: (jax.Array): Array of computed option prices.
    """
    [spots, strikes, expires, vols] = cast_arrays(
        [spots, strikes, expires, vols], dtype
    )

    discount_factors = jnp.exp(-discount_rates * expires)

    calls = compute_undiscounted_call_prices(
        spots, strikes, expires, vols, discount_rates
    )

    if are_calls is None:
        return calls

    puts = calls + (strikes * discount_factors) - spots
    return jnp.where(are_calls, calls, puts)

def cast_arrays(array: List[jax.Array], dtype):
    """
    Casts the array to the specified dtype

    :param array: List of arrays
    :param dtype: dtype to cast the array to
    :return: List of arrays with the specified dtype
    """
    if dtype is not None:
        return [jnp.astype(el, dtype) for el in array]

    return array

@jax.jit
def compute_undiscounted_call_prices(spots, strikes, expires, vols, discount_rates):
    """
    Compute the undiscounted call option prices

    :param spots: Current spot price of the underlying
    :param strikes: Strike price of the option
    :param expires: Time to expiration of the option
    :param vols: Volatility of the underlying
    :param discount_rates: Risk-free rate
    :return: Undiscounted call option prices
    """
    [_d1, _d2] = _compute_d1_d2(spots, strikes, expires, vols, discount_rates)

    return cum_normal(_d1) * spots - cum_normal(_d2) * strikes * jnp.exp(-discount_rates * expires)

def _compute_d1_d2(spots, strikes, expires, vols, discount_rates):
    """
    Compute the d1 and d2 terms in the Black-Scholes formula

    :param spots: Current spot price of the underlying
    :param strikes: Strike price of the option
    :param expires: Time to expiration of the option
    :param vols: Volatility of the underlying
    :param discount_rates: Risk-free rate
    :return: d1 and d2 terms
    """
    vol_sqrt_t = vols * jnp.sqrt(expires)

    _d1 = d1(spots, strikes, vols, expires, discount_rates)

    return [_d1, _d1 - vol_sqrt_t]

@jax.jit
def d1(spots, strikes, vols, expires, discount_rates):
    """
    Calculate the d1 term in the Black-Scholes formula

    :param spots: Current spot price of the underlying
    :param strikes: Strike price of the option
    :param vols: Volatility of the underlying
    :param expires: Time to expiration of the option
    :param discount_rates: Risk-free rate
    :return: d1 term
    """
    vol_sqrt_t = vols * jnp.sqrt(expires)

    return jnp.divide(
        (jnp.log(spots / strikes) + (discount_rates + (vols**2 / 2)) * expires),
        vol_sqrt_t,
    )


@jax.jit
def cum_normal(x):
    """
    Cumulative normal distribution function

    :param x: Input value
    :return: Cumulative normal distribution value
    """
    return (erf(x / _SQRT_2) + 1) / 2