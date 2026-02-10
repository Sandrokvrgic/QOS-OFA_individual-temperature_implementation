#https://gael-varoquaux.info/scipy-lecture-notes/intro/scipy/auto_examples/plot_2d_minimization.html
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax
import jax.numpy as jnp

### IDS
SIX_HUMP_WIDE = 0
SIX_HUMP_CLASSIC = 1

#### FUNCTIONS
def evaluate_objective(x: jnp.ndarray, objective_id: int) -> jnp.ndarray:
    """
    Evaluate a single point x for the given objective_id.

    x: shape (D,)
    objective_id: one of the defined IDs, e.g. SIX_HUMP (0).

    This function is JAX-friendly and can be used inside vmaps/jits.
    """
    # Define one function per objective_id index.
    # If in the future you add more objectives, add them here in the right order.
    f = (lambda z: six_hump(z), lambda z: six_hump(z))  # objective_id == 0 (SIX_HUMP_WIDE), objective_id == 1 (SIX_HUMP_CLASSIC)

    return jax.lax.switch(objective_id, f, x)

    
def get_default_bounds(objective_id: int):
    """
    Return default bounds [a_i, b_i] for each dimension, shape (D, 2).
    You can use this when building your Params if you want.

    For now, only implemented for six-hump examples.
    """
    if objective_id == SIX_HUMP_CLASSIC:
        return jnp.array([[-2.0,  2.0], [-1.0,  1.0]], dtype=jnp.float32)
    elif objective_id == SIX_HUMP_WIDE:
        return jnp.array([[-5.0,  5.0], [-5.0,  5.0]], dtype=jnp.float32)
    else:
        raise ValueError(f"No bounds for objective_id: {objective_id}")
    

### BENCHMARKS
# Six Hump Camelback  function
def six_hump(x):
    return ((4 - 2.1*x[0]**2 + x[0]**4 / 3.) * x[0]**2 + x[0] * x[1]+ (-4 + 4*x[1]**2) * x[1] **2)
