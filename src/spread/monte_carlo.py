import jax
import jax.numpy as jnp
import jax.random as random 

from typing import Tuple

def mc():
    pass


def sample_paths(
    S0: jax.Array, 
    K: jax.Array, 
    r, 
    sigma_1, 
    sigma_2, 
    corr_12, 
    corr_1v, 
    corr_2v, 
    sigma_v, 
    maturity, 
    n, 
    n_sim) -> Tuple[jax.Array, jax.Array]:
    """Sample paths from the Multivariate Heston model

    Args:
        seed (int): The seed for the random number generator
        maturity (float): The time in years
        n (int): The number of steps
        n_sim (int): The number of simulations

    Returns:
        Tuple[jax.Array, jax.Array]: The simulated paths and the variance processes of the assets
    """
    dt = maturity / n
    dt_sq = jnp.sqrt(dt)

    Z_1 = random.normal(loc=0, scale=1, size=(n_sim, n - 1))
    Z_2 = random.normal(loc=0, scale=1, size=(n_sim, n - 1))
    
    W_1 = Z_1
    W_2 = W_1 @ corr_12 + Z_2 @ jnp.sqrt(1 - corr_12**2)

    W_v1 = W_1 
    W_v2 = s0 # sos

    v1_paths = jnp.zeros((n_sim, n))
    v2_paths = jnp.zeros((n_sim, n))
    S1_paths = jnp.zeros((n_sim, n))
    S2_paths = jnp.zeros((n_sim, n))
    v1_paths.at[:, 0, :].set(v0)
    v2_paths.at[:, 0, :].set(v0)
    S1_paths.at[:, 0, :].set(S1_0)
    S1_paths.at[:, 0, :].set(S2_0)

    # Compute trajectories of v and S using vectorized operations
    for t in range(1, n):
        v_paths[:, t, :] = np.abs(
            v_paths[:, t - 1, :]
            + self._kappa * (self._theta - v_paths[:, t - 1, :]) * dt
            + self._sigma * np.sqrt(v_paths[:, t - 1, :]) * dt_sq * W_v[:, t - 1, :]
        )
        S_paths[:, t, :] = S_paths[:, t - 1, :] * np.exp(
            (self._mean - 0.5 * v_paths[:, t - 1, :]) * dt
            + np.sqrt(v_paths[:, t - 1, :]) * dt_sq * W_S[:, t - 1, :]
        )

    return jnp.asarray(S_paths.transpose(1, 0, 2)), jnp.asarray(
        v_paths.transpose(1, 0, 2)
    )

if __name__ == '__main__':
    key = random.PRNGKey(44)

    key, subkey = random.split(key)

    print(subkey)
    W1 = random.normal(key, shape=(252, 10))

    key, subkey = random.split(key)

    print(subkey)

    W2 = random.normal(key, shape=(252, 10))