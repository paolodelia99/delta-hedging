import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jaxfin.price_engine.black_scholes import delta_vanilla, european_price

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
    d1 = (jnp.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * jnp.sqrt(time_to_maturity))
    d2 = d1 - volatility * jnp.sqrt(time_to_maturity)

    call_price = spot * jax.scipy.stats.norm.cdf(d1) - strike * jnp.exp(-rate * time_to_maturity) * jax.scipy.stats.norm.cdf(d2)
    return call_price


if __name__ == '__main__':
    spot_price = 100.0
    strike_price = 110.0
    risk_free_rate = 0.00
    annual_volatility = 0.2
    time_to_expiry = 1.0
    spot_price_a = jnp.asarray(spot_price)
    strike_price_a = jnp.asarray(strike_price)
    risk_free_rate_a = jnp.asarray(risk_free_rate)
    annual_volatility_a = jnp.asarray(annual_volatility)
    time_to_expiry_a = jnp.asarray(time_to_expiry)


    call_option_price = black_scholes_call_price(spot_price, strike_price, risk_free_rate, annual_volatility,
                                                 time_to_expiry)
    call_option_price_1 = european_price(spot_price_a, strike_price_a, time_to_expiry_a, annual_volatility_a, risk_free_rate_a, dtype=jnp.float32)
    delta = grad(black_scholes_call_price, argnums=0)(spot_price, strike_price, risk_free_rate, annual_volatility, time_to_expiry)
    delta_1 = grad(european_price)(spot_price_a, strike_price_a, time_to_expiry_a, annual_volatility_a, risk_free_rate_a, dtype=jnp.float32)
    print(f"Call option price: ${call_option_price:.2f}")
    print(f'Call option price (jaxfin): ${call_option_price_1:.2f}')
    print(f"Delta option price: ${delta:.2f}")
    print(f'Delta option price (jaxfin): ${delta_1:.2f}')
