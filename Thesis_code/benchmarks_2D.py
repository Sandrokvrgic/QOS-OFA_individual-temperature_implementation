#https://gael-varoquaux.info/scipy-lecture-notes/intro/scipy/auto_examples/plot_2d_minimization.html
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax
import jax.numpy as jnp

# IDS
SIX_HUMP_WIDE = 0
SIX_HUMP_CLASSIC = 1
SIX_HUMP_LOW_CONTRAST = 2
SIX_HUMP_HIGH_CONTRAST = 3
SIX_HUMP_ZERO_MIN = 4


# FUNCTIONS
def evaluate_objective(x: jnp.ndarray, objective_id: int) -> jnp.ndarray:
    """
    Evaluate a single point x for the given objective_id.

    x: shape (D,)
    objective_id: one of the defined IDs, e.g. SIX_HUMP (0).
    """
    f = (
        lambda z: six_hump(z),
        lambda z: six_hump(z),
        lambda z : six_hump_low_contrast(z),
        lambda z : six_hump_high_contrast(z),
        lambda z: six_hump_zero_min(z), 
        )
    
    return jax.lax.switch(objective_id, f, x)

    
def get_default_bounds(objective_id: int):
    """
    Return default bounds [a_i, b_i] for each dimension, shape (D, 2).
    You can use this when building your Params if you want.

    For now, only implemented for six-hump examples.
    """
    if objective_id in (SIX_HUMP_CLASSIC,SIX_HUMP_LOW_CONTRAST,SIX_HUMP_HIGH_CONTRAST,SIX_HUMP_ZERO_MIN):
        return jnp.array([[-2.0,  2.0], [-1.0,  1.0]], dtype=jnp.float32)
    elif objective_id == SIX_HUMP_WIDE:
        return jnp.array([[-5.0,  5.0], [-5.0,  5.0]], dtype=jnp.float32)
    else:
        raise ValueError(f"No bounds for objective_id: {objective_id}")
    

# BENCHMARKS
# Six Hump Camelback  function
def six_hump(x):
    return ((4 - 2.1*x[0]**2 + x[0]**4 / 3.) * x[0]**2 + x[0] * x[1]+ (-4 + 4*x[1]**2) * x[1] **2)


# Global minimum value for the six hump function, used for shifting and scaling.
SIX_HUMP_FMIN = -1.031628453489877
EPS_SHIFT = 1e-6


def six_hump_zero_min(x):
    """
    Shifted Six-Hump Camelback function.

    Same landscape shape as the original, but shifted upward so that:
        global minimum = 0
        all other values > 0

    useful when an algorithm assumes non-negative objective values.
    """
    g = six_hump(x) - SIX_HUMP_FMIN
    return jnp.maximum(g, 0.0)


def six_hump_shifted_positive(x):
    return six_hump(x) - SIX_HUMP_FMIN + EPS_SHIFT


def six_hump_low_contrast(x):
    g = six_hump_shifted_positive(x)
    return g**0.2


def six_hump_high_contrast(x):
    g = six_hump_shifted_positive(x)
    return g**2