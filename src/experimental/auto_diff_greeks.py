import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jaxfin.price_engine.black_scholes import delta_vanilla, bs_price
from black_scholes import black_scholes_call_price, bs_price


def bs_delta(spot, strike, rate, vol, maturity):
    """
    Calculates the delta of a European vanilla call option under the Black-Scholes model.

    Args:
        spot: Current spot price of the underlying asset.
        strike: Strike price of the option.
        rate: Annual risk-free interest rate (as a decimal).
        volatility: Annualized volatility of the underlying asset (as a decimal).
        time_to_maturity: Time to option maturity in years.

    Returns:
        Delta of the option
    """
    return grad(black_scholes_call_price, argnums=0)(spot, strike, rate, vol, maturity)
                                                     
def bs_gamma(spot, strike, rate, vol, maturity):
    """
    Calculates the gamma of a European vanilla call option under the Black-Scholes model.
    
    Args:
        spot: Current spot price of the underlying asset.
        strike: Strike price of the option.
        rate: Annual risk-free interest rate (as a decimal).
        volatility: Annualized volatility of the underlying asset (as a decimal).
        time_to_maturity: Time to option maturity in years.
        
    Returns:
        Gamma of the option
    """
    return grad(grad(black_scholes_call_price, argnums=0), argnums=0)(spot, strike, rate, vol, maturity)

def bs_rho(spot, strike, rate, vol, maturity):
    """
    Calculates the rho of a European vanilla call option under the Black-Scholes model.
    
    Args:
        spot: Current spot price of the underlying asset.
        strike: Strike price of the option.
        rate: Annual risk-free interest rate (as a decimal).
        volatility: Annualized volatility of the underlying asset (as a decimal).
        time_to_maturity: Time to option maturity in years.
    
    Returns:
        Rho of the option
    """
    return grad(black_scholes_call_price, argnums=2)(spot, strike, rate, vol, maturity)

def bs_vega(spot, strike, rate, vol, maturity):
    """
    Calculates the vega of a European vanilla call option under the Black-Scholes model.
    
    Args:
        spot: Current spot price of the underlying asset.
        strike: Strike price of the option.
        rate: Annual risk-free interest rate (as a decimal).
        volatility: Annualized volatility of the underlying asset (as a decimal).
        time_to_maturity: Time to option maturity in years.
        
    Returns:
        Vega of the option
    """
    return grad(black_scholes_call_price, argnums=3)(spot, strike, rate, vol, maturity)

def bs_theta(spot, strike, rate, vol, maturity):
    """
    Calculates the theta of a European vanilla call option under the Black-Scholes model.
    
    Args:
        spot: Current spot price of the underlying asset.
        strike: Strike price of the option.
        rate: Annual risk-free interest rate (as a decimal).
        volatility: Annualized volatility of the underlying asset (as a decimal).
        time_to_maturity: Time to option maturity in years.
        
    Returns:
        Theta of the option
    """
    return grad(black_scholes_call_price, argnums=4)(spot, strike, rate, vol, maturity)


if __name__ == '__main__':
    spot_price = 100.0
    strike_price = 90.0
    risk_free_rate = 0.05
    annual_volatility = 0.3
    time_to_expiry = 1.0
    spot_price_a = jnp.asarray(spot_price)
    strike_price_a = jnp.asarray(strike_price)
    risk_free_rate_a = jnp.asarray(risk_free_rate)
    annual_volatility_a = jnp.asarray(annual_volatility)
    time_to_expiry_a = jnp.asarray(time_to_expiry)
    are_calls = jnp.asarray(False, dtype=jnp.bool_)

    call_option_price = black_scholes_call_price(spot_price, strike_price, risk_free_rate, annual_volatility,
                                                 time_to_expiry)
    call_option_price_1 = bs_price(spot_price_a, strike_price_a, time_to_expiry_a, annual_volatility_a,
                                         risk_free_rate_a, dtype=jnp.float32, are_calls=are_calls)
    delta = bs_delta(spot_price, strike_price, risk_free_rate, annual_volatility, time_to_expiry)
    delta_1 = grad(bs_price)(spot_price_a, strike_price_a, time_to_expiry_a, annual_volatility_a,
                                   risk_free_rate_a, dtype=jnp.float32, are_calls=are_calls)
    gamma = bs_gamma(spot_price, strike_price, risk_free_rate, annual_volatility, time_to_expiry)
    gamma_1 = grad(grad(bs_price))(spot_price_a, strike_price_a, time_to_expiry_a, annual_volatility_a,
                                        risk_free_rate_a, dtype=jnp.float32, are_calls=are_calls)
    theta = bs_theta(spot_price, strike_price, risk_free_rate, annual_volatility, time_to_expiry)
    theta_1 = grad(bs_price, argnums=2)(spot_price_a, strike_price_a, time_to_expiry_a, annual_volatility_a,
                                      risk_free_rate_a, dtype=jnp.float32, are_calls=are_calls)
    vega = bs_vega(spot_price, strike_price, risk_free_rate, annual_volatility, time_to_expiry)
    vega_1 = grad(bs_price, argnums=3)(spot_price_a, strike_price_a, time_to_expiry_a, annual_volatility_a,
                                    risk_free_rate_a, dtype=jnp.float32, are_calls=are_calls)
    rho = bs_rho(spot_price, strike_price, risk_free_rate, annual_volatility, time_to_expiry)
    rho_1 = grad(bs_price, argnums=4)(spot_price_a, strike_price_a, time_to_expiry_a, annual_volatility_a,
                                    risk_free_rate_a, dtype=jnp.float32, are_calls=are_calls)
    print(f"Call option price: ${call_option_price:.2f}")
    print(f'Call option price (jaxfin): ${call_option_price_1:.2f}')
    print(f"Delta: {delta:.5f}")
    print(f'Delta (jaxfin): {delta_1:.5f}')
    print(f'Absolute difference of the deltas: {abs(delta - delta_1):.10f}')
    print(f'Gamma: {gamma:.5f}')
    print(f'Gamma (jaxfin): {gamma_1:.5f}')
    print(f'Absolute difference of the gammas: {abs(gamma - gamma_1):.10f}')
    print(f'Theta: {theta:.5f}')
    print(f'Theta (jaxfin): {theta_1:.5f}')
    print(f'Absolute difference of the thetas: {abs(theta - theta_1):.10f}')
    print(f'Vega: {vega:.5f}')
    print(f'Vega (jaxfin): {vega_1:.5f}')
    print(f'Absolute difference of the vegas: {abs(vega - vega_1):.10f}')
    print(f'Rho: {rho:.5f}')
    print(f'Rho (jaxfin): {rho_1:.5f}')
    print(f'Absolute difference of the rhos: {abs(rho - rho_1):.10f}')
