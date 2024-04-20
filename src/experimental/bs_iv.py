from time import time

import jax
import scipy as sq
import numpy as np
from jaxfin.price_engine.black_scholes import european_price
from jaxfin.price_engine.fft import fourier_inv_call, delta_call_fourier

def IV_newton(S, X, T, r, Option_Value):

    def obj_function(IV):
        result = Option_Value - european_price(S, X, T, IV, r)
        return result

    x0 = np.sqrt((2 * np.abs(np.log(S/X))) / T)

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
      IV_Result = sq.optimize.brenth(obj_function_vec, a=0.01, b=2.5, xtol=0.000001, args=(S, option_price))
    except RuntimeError:
      IV_Result = np.nan
    except ValueError:
      IV_Result = np.nan

    return IV_Result

  IV_brent_vec = np.vectorize(IV_brent_single)
  return IV_brent_vec(S, option_price)


if __name__ == '__main__':
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

    print(f'The price of the european option is: {opt_price}')
    print(f'The implied volatility is: {IV_newton(stock_price, K, expiration, r, opt_price)}')
    
    start = time()
    iv_brent = IV_brent([stock_price, 101, 102, 99, 103, 98], K, expiration, r, [opt_price, opt_price, opt_price, opt_price, opt_price, opt_price])
    end = time()

    print(f'The implied volatility is: {iv_brent}')
    print(f'Time taken: {end - start}')

    print('Heston model case')
    opt_price = fourier_inv_call(stock_price, K, expiration, v0, r, theta, sigma, kappa, rho)

    print(f'The price of the european option under the Heston model is: {opt_price}')

    start = time()
    iv_brent = IV_brent(stock_price, K, expiration, r, opt_price)
    end = time()

    print(f'The implied volatility is: {iv_brent}')
    print(f'Time taken: {end - start}')