from time import time

import jax
from jax.scipy.optimize import minimize
from jax import jit, vmap
import jax.numpy as jnp
import scipy as sq
import numpy as np
from jaxfin.price_engine.black_scholes import european_price, vega_european
from jaxfin.price_engine.fft import fourier_inv_call, delta_call_fourier


def iv_approx_1(C, S, T):
    return np.sqrt(2 * np.pi / T) * C / S

@jit
def iv_approx(S, X, T, option_price):
    """Corrado-Miller IV approximation

    Args:
        S (_type_): _description_
        X (_type_): _description_
        T (_type_): _description_
        option_price (_type_): _description_

    Returns:
        _type_: _description_
    """
    sqrt_term = jnp.sqrt(
        (2 * option_price + X - S) ** 2
        - 1.85 * ((S + X) * (X - S) ** 2) / (jnp.pi * jnp.sqrt(X * S))
    )
    return (
        jnp.sqrt(2 * jnp.pi / T)
        / (2 * (S + X))
        * (2 * option_price + X - S + sqrt_term)
    )


def IV_newton(S, X, T, r, Option_Value):

    def obj_function(IV):
        result = Option_Value - european_price(S, X, T, IV, r)
        return result

    x0 = np.sqrt((2 * np.abs(np.log(S / X))) / T)

    try:
        IV_Result = sq.optimize.newton(obj_function, x0=x0)
    except RuntimeError:
        IV_Result = np.nan
    except ValueError:
        IV_Result = np.nan

    return IV_Result


def IV_brent(S, X, T, r, option_price):

    def obj_function(IV, S, option_price):
        result = option_price - european_price(S, X, T, IV, r)
        return result

    obj_function_vec = np.vectorize(obj_function)

    def IV_brent_single(S, option_price):
        try:
            IV_Result = sq.optimize.brenth(
                obj_function_vec, 
                a=0.01, 
                b=2.5, 
                xtol=1e-5, 
                args=(S, option_price),
                maxiter=50
            )
        except RuntimeError:
            IV_Result = iv_approx(S, X, T, option_price)
        except ValueError:
            IV_Result = iv_approx(S, X, T, option_price)

        return IV_Result

    IV_brent_vec = np.vectorize(IV_brent_single)
    return IV_brent_vec(S, option_price)


if __name__ == "__main__":
    stock_price = 100
    K = 100
    expiration = 1.0
    r = 0.0
    vol = 0.2

    # Heston params
    kappa = 1.0
    v0 = 0.04
    theta = 0.04
    rho = -0.7
    sigma = 0.1

    opt_price = european_price(stock_price, K, expiration, vol, r)
    opt_price = np.asarray(opt_price)

    print(f"The price of the european option is: {opt_price}")
    print(
        f"The implied volatility is: {IV_newton(stock_price, K, expiration, r, opt_price)}"
    )

    start = time()
    iv_brent = IV_brent(
        [stock_price, 101, 102, 99, 103, 98],
        K,
        expiration,
        r,
        [opt_price, opt_price, opt_price, opt_price, opt_price, opt_price],
    )
    end = time()

    print(f"The implied volatility is: {iv_brent}")
    print(f"Time taken: {end - start}")

    print("Heston model case")
    opt_price = fourier_inv_call(
        stock_price, K, expiration, v0, r, theta, sigma, kappa, rho
    )

    print(f"The price of the european option under the Heston model is: {opt_price}")

    start = time()
    iv_brent = IV_brent(stock_price, K, expiration, r, opt_price)
    end = time()

    print(f"The implied volatility is: {iv_brent}")
    print(f"Time taken: {end - start}")

    stock_prices = np.asarray([100, 101, 102, 99, 98])
    opt_prices = np.asarray(
        [
            fourier_inv_call(s, K, expiration, v0, r, theta, sigma, kappa, rho)
            for s in stock_prices
        ]
    )

    print(f"IVBrent vectorized")

    start = time()
    iv_brent = IV_brent(stock_prices, K, expiration, r, opt_prices)
    end = time()

    print(f"The implied volatility is: {iv_brent}")
    print(f"Time taken: {end - start}")

    print("IV approx method")

    start = time()
    iv_approxs = [
        iv_approx(s, K, expiration, c) for s, c in zip(stock_prices, opt_prices)
    ]
    end = time()

    print(f"The implied volatility is: {iv_approxs}")
    print(f"Time taken: {end - start}")
