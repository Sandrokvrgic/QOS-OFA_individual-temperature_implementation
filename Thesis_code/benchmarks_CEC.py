from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

### IDS for the reduced CEC suite.
CEC_BENT_CIGAR   = 0
CEC_ZAKHAROV     = 1
CEC_RASTRIGIN    = 2
CEC_SCAFFERS_F6  = 3
CEC_LEVY         = 4
CEC_HYBRID_1     = 5
CEC_HYBRID_4     = 6


### HELPER FUNCTIONS

def get_cec_suite_ids():
    """
    Return the tuple of benchmark IDs used in the reduced CEC suite.
    """
    return (
        CEC_BENT_CIGAR,
        CEC_ZAKHAROV,
        CEC_RASTRIGIN,
        CEC_SCAFFERS_F6,
        CEC_LEVY,
        CEC_HYBRID_1,
        CEC_HYBRID_4,
    )


def get_objective_name(objective_id: int) -> str:
    names = (
        "CEC Bent Cigar",
        "CEC Zakharov",
        "CEC Rastrigin",
        "CEC Expanded Scaffer's F6",
        "CEC Levy",
        "CEC Hybrid Function 1",
        "CEC Hybrid Function 4",
    )
    return names[objective_id]


def get_objective_family(objective_id: int) -> str:
    families = (
        "unimodal",
        "unimodal",
        "multimodal",
        "multimodal",
        "multimodal",
        "hybrid",
        "hybrid",
    )
    return families[objective_id]


def get_default_bounds(objective_id: int, dim: int):
    """
    Return default bounds [a_i, b_i] for each dimension, shape (D, 2).

    For the reduced CEC suite, all functions use [-100, 100]^D
    by default in the official reports.
    """
    lo = -100.0
    hi = 100.0
    return jnp.tile(jnp.array([[lo, hi]], dtype=jnp.float32), (dim, 1))


### CEC SETTINGS

CEC_DIM = 30
CEC_BOUNDS = (-100.0, 100.0)
DATA_DIR = Path(__file__).resolve().parent / "Shift_data"

# Internal benchmark ID = official CEC 2017 function number
CEC_FILE_NUMBERS = {
    CEC_BENT_CIGAR: 1,
    CEC_ZAKHAROV: 2,
    CEC_RASTRIGIN: 4,
    CEC_SCAFFERS_F6: 5,
    CEC_LEVY: 8,
    CEC_HYBRID_1: 10,
    CEC_HYBRID_4: 13,
}


def _load_txt_array(path: Path, dtype=jnp.float32) -> jnp.ndarray:
    """
    Load a numeric txt file and return it as a JAX array.
    """
    arr = np.loadtxt(path)
    return jnp.array(arr, dtype=dtype)


def load_shift_vector(func_num: int, dim: int = CEC_DIM, dtype=jnp.float32) -> jnp.ndarray:
    """
    Load the shift vector from:
        shift_data_<func_num>.txt

    The official file may contain more values than needed; we take the first dim.
    Returned shape: (dim,)
    """
    path = DATA_DIR / f"shift_data_{func_num}.txt"
    arr = _load_txt_array(path, dtype=dtype).reshape(-1)
    return arr[:dim]


def load_rotation_matrix(func_num: int, dim: int = CEC_DIM, dtype=jnp.float32) -> jnp.ndarray:
    """
    Load the rotation matrix from:
        M_<func_num>_D<dim>.txt

    Returned shape: (dim, dim)
    """
    path = DATA_DIR / f"M_{func_num}_D{dim}.txt"
    arr = _load_txt_array(path, dtype=dtype)
    return arr.reshape(dim, dim)

def shift_input(x: jnp.ndarray, shift: jnp.ndarray) -> jnp.ndarray:
    return x - shift


def rotate_input(x: jnp.ndarray, rot: jnp.ndarray) -> jnp.ndarray:
    return rot @ x


def transform_shift_rotate(x: jnp.ndarray, shift: jnp.ndarray, rot: jnp.ndarray) -> jnp.ndarray:
    """
    Standard CEC transform:
        z = M (x - o)
    """
    return rotate_input(shift_input(x, shift), rot)

def cec_bent_cigar(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    z = transform_shift_rotate(x, params["shift"], params["rot"])
    return bent_cigar(z) + params["bias"]


def cec_zakharov(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    z = transform_shift_rotate(x, params["shift"], params["rot"])
    return zakharov(z) + params["bias"]


def cec_rastrigin(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    z = transform_shift_rotate(x, params["shift"], params["rot"])
    return rastrigin(z) + params["bias"]


def cec_scaffers_f6(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    z = transform_shift_rotate(x, params["shift"], params["rot"])
    return scaffers_f6(z) + params["bias"]


def cec_levy(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    Official-style transformed Levy wrapper.

    Since the base Levy optimum is at x = (1, ..., 1), we shift the transformed
    coordinates by +1 so that the benchmark optimum is located at the shift vector.
    """
    z = transform_shift_rotate(x, params["shift"], params["rot"])
    return levy(z + 1.0) + params["bias"]

def cec_hybrid_1(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    CEC-style transformed Hybrid Function 1 wrapper.

    Structure:
    - transform with shift + rotation
    - apply shuffle
    - split by proportions [0.2, 0.4, 0.4]
    - apply:
        g1 = Zakharov
        g2 = Rosenbrock
        g3 = Rastrigin
    - add CEC bias
    """
    z = transform_shift_rotate(x, params["shift"], params["rot"])

    z1, z2, z3 = _split_by_proportions(z, [0.2, 0.4, 0.4])

    val = (
        zakharov(z1)
        + rosenbrock(z2 + 1.0)
        + rastrigin(z3)
    )

    return val + params["bias"]

def cec_hybrid_4(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    """
    CEC-style transformed Hybrid Function 4 wrapper.

    IMPORTANT:
    The provided M_13_D30.txt already comes from the hybrid generator pipeline
    with the subcomponent permutation embedded in the saved matrix.
    Therefore we do NOT apply shuffle_data again here.
    """
    z = transform_shift_rotate(x, params["shift"], params["rot"])

    z1, z2, z3, z4 = _split_by_proportions(z, [0.2, 0.2, 0.2, 0.4])

    val = (
        high_conditioned_elliptic(z1)
        + ackley(z2)
        + schaffers_f7(z3)
        + rastrigin(z4)
    )

    return val + params["bias"]


def get_cec_params(dim: int = CEC_DIM, dtype=jnp.float32):
    """
    Load the shift / rotation / shuffle data for the active 7-function CEC suite.

    Bias values follow the CEC 2017 benchmark table:
    - F1  -> 100
    - F2  -> 200
    - F4  -> 400
    - F5  -> 500
    - F8  -> 800
    - F10 -> 1000
    - F13 -> 1300
    """
    return {
        CEC_BENT_CIGAR: {
            "func_num": 1,
            "shift": load_shift_vector(1, dim, dtype),
            "rot": load_rotation_matrix(1, dim, dtype),
            "bias": jnp.array(100.0, dtype=dtype),
        },
        CEC_ZAKHAROV: {
            "func_num": 2,
            "shift": load_shift_vector(2, dim, dtype),
            "rot": load_rotation_matrix(2, dim, dtype),
            "bias": jnp.array(200.0, dtype=dtype),
        },
        CEC_RASTRIGIN: {
            "func_num": 4,
            "shift": load_shift_vector(4, dim, dtype),
            "rot": load_rotation_matrix(4, dim, dtype),
            "bias": jnp.array(400.0, dtype=dtype),
        },
        CEC_SCAFFERS_F6: {
            "func_num": 5,
            "shift": load_shift_vector(5, dim, dtype),
            "rot": load_rotation_matrix(5, dim, dtype),
            "bias": jnp.array(500.0, dtype=dtype),
        },
        CEC_LEVY: {
            "func_num": 8,
            "shift": load_shift_vector(8, dim, dtype),
            "rot": load_rotation_matrix(8, dim, dtype),
            "bias": jnp.array(800.0, dtype=dtype),
        },
        CEC_HYBRID_1: {
            "func_num": 10,
            "shift": load_shift_vector(10, dim, dtype),
            "rot": load_rotation_matrix(10, dim, dtype),
            "bias": jnp.array(1000.0, dtype=dtype),
        },
        CEC_HYBRID_4: {
            "func_num": 13,
            "shift": load_shift_vector(13, dim, dtype),
            "rot": load_rotation_matrix(13, dim, dtype),
            "bias": jnp.array(1300.0, dtype=dtype),
        }
    }


def evaluate_objective(x: jnp.ndarray, objective_id: int) -> jnp.ndarray:
    """
    Evaluate a single point x for the given CEC objective_id.

    x: shape (D,)
    objective_id: one of the CEC_* IDs above
    """
    params = get_cec_params(dim=x.shape[0], dtype=x.dtype)

    if objective_id == CEC_BENT_CIGAR:
        return cec_bent_cigar(x, params[CEC_BENT_CIGAR])
    elif objective_id == CEC_ZAKHAROV:
        return cec_zakharov(x, params[CEC_ZAKHAROV])
    elif objective_id == CEC_RASTRIGIN:
        return cec_rastrigin(x, params[CEC_RASTRIGIN])
    elif objective_id == CEC_SCAFFERS_F6:
        return cec_scaffers_f6(x, params[CEC_SCAFFERS_F6])
    elif objective_id == CEC_LEVY:
        return cec_levy(x, params[CEC_LEVY])
    elif objective_id == CEC_HYBRID_1:
        return cec_hybrid_1(x, params[CEC_HYBRID_1])
    elif objective_id == CEC_HYBRID_4:
        return cec_hybrid_4(x, params[CEC_HYBRID_4])
    else:
        raise ValueError(f"Unknown objective_id: {objective_id}")


### CEC BENCHMARK FUNCTIONS

# UNIMODAL
def bent_cigar(x: jnp.ndarray) -> jnp.ndarray:
    """
    CEC Bent Cigar benchmark function.

    Source:
    - CEC 2017: "Shifted and Rotated Bent Cigar Function", Function 1 (F1)
    - Also in CEC 2014 as "Rotated Bent Cigar Function", Function 2 (F2)

    Base formula:
        f(x) = x_1^2 + 10^6 * sum_{i=2..D} x_i^2

    Type:
    - Unimodal: there is one global optimum only.
    - Non-separable in the official CEC benchmark version after shift/rotation.
    - Ill-conditioned / anisotropic: one direction behaves very differently from the others.
    - Smooth but with a very narrow valley/ridge structure.

    Optimum of the base function:
    - Global minimum at x = 0
    - f(0) = 0
    """
    return x[0] ** 2 + 1e6 * jnp.sum(x[1:] ** 2)


def zakharov(x: jnp.ndarray) -> jnp.ndarray:
    """
    CEC Zakharov benchmark function.

    Source:
    - CEC 2017: "Shifted and Rotated Zakharov Function", Function 2 (F2)

    Base formula:
        f(x) = sum_{i=1..D} x_i^2 + (sum_{i=1..D} 0.5 * i * x_i)^2 + (sum_{i=1..D} 0.5 * i * x_i)^4

    Type:
    - Unimodal: there is one global optimum only.
    - Non-separable in the official CEC benchmark version after shift/rotation.
    - Smooth polynomial function.
    - Has variable coupling through the linear weighted sum term, whose square and
      fourth power bend the landscape away from a simple isotropic bowl.

    Optimum of the base function:
    - Global minimum at x = 0
    - f(0) = 0
    """
    i = jnp.arange(1, x.shape[0] + 1, dtype=x.dtype)
    linear_term = jnp.sum(0.5 * i * x)
    return jnp.sum(x ** 2) + linear_term ** 2 + linear_term ** 4


# MULTIMODAL
def rastrigin(x: jnp.ndarray) -> jnp.ndarray:
    """
    CEC Rastrigin benchmark function.

    Source:
    - CEC 2017: "Shifted and Rotated Rastrigin’s Function", Function 4 (F4)
    - Also in CEC 2014 as "Shifted and Rotated Rastrigin’s Function", Function 9 (F9)

    Base formula:
        f(x) = sum_{i=1..D} [x_i^2 - 10*cos(2*pi*x_i) + 10]

    Type:
    - Multimodal: there are many local optima.
    - Non-separable in the official CEC benchmark version after shift/rotation.
    - Highly regular and periodic landscape.
    - Local optima's number is huge.

    Optimum of the base function:
    - Global minimum at x = 0
    - f(0) = 0
    """
    return jnp.sum(x**2 - 10.0 * jnp.cos(2.0 * jnp.pi * x) + 10.0)


def scaffers_f6(x: jnp.ndarray) -> jnp.ndarray:
    """
    CEC Expanded Scaffer's F6 benchmark function.

    Source:
    - CEC 2017: "Shifted and Rotated Expanded Scaffer’s F6 Function", Function 5 (F5)
    - Also in CEC 2014 as "Shifted and Rotated Expanded Scaffer’s F6 Function", Function 16 (F16)

    Base formula:
        g(a, b) = 0.5 + (sin(sqrt(a^2 + b^2))^2 - 0.5) / (1 + 0.001*(a^2 + b^2))^2

        f(x) = sum_{i=1..D-1} g(x_i, x_{i+1}) + g(x_D, x_1)

    Type:
    - Multimodal: there are many local optima.
    - Non-separable in the official CEC benchmark version after shift/rotation.
    - Rugged and oscillatory landscape.
    - Neighboring variables are coupled through the pairwise g(·,·) terms.

    Optimum of the base function:
    - Global minimum at x = 0
    - f(0) = 0
    """
    x_next = jnp.roll(x, -1)
    r2 = x**2 + x_next**2
    num = jnp.sin(jnp.sqrt(r2))**2 - 0.5
    den = (1.0 + 0.001 * r2)**2
    g = 0.5 + num / den
    return jnp.sum(g)


def levy(x: jnp.ndarray) -> jnp.ndarray:
    """
    CEC Levy benchmark function.

    Source:
    - CEC 2017: "Shifted and Rotated Levy Function", Function 8 (F8)

    Base formula:
        w_i = 1 + (x_i - 1)/4
        f(x) = sin(pi*w_1)^2 + sum_{i=1..D-1} (w_i - 1)^2 * [1 + 10*sin(pi*w_i + 1)^2] + (w_D - 1)^2 * [1 + sin(2*pi*w_D)^2]

    Type:
    - Multimodal: there are many local optima.
    - Non-separable in the official CEC benchmark version after shift/rotation.
    - Oscillatory landscape with multiple deceptive regions.
    - The transformed variables w_i shift the structure away from a simple origin-centered form.

    Optimum of the base function:
    - Global minimum at x = (1, 1, ..., 1)
    - f(x) = 0
    """
    w = 1.0 + (x - 1.0) / 4.0

    term1 = jnp.sin(jnp.pi * w[0]) ** 2
    term2 = jnp.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * jnp.sin(jnp.pi * w[:-1] + 1.0) ** 2))
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + jnp.sin(2.0 * jnp.pi * w[-1]) ** 2)

    return term1 + term2 + term3



# HYBRID AND COMPOSITION FUNCTIONS
# Hybrid 1
def rosenbrock(x: jnp.ndarray) -> jnp.ndarray:
    """
    Rosenbrock benchmark function.

    Source:
    - Base formula in CEC 2017 basic functions as Rosenbrock’s Function

    Base formula:
        f(x) = sum_{i=1..D-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]

    Type:
    - Multimodal in the practical optimization sense used by CEC.
    - Non-separable.
    - Has a narrow curved valley.

    Optimum of the base function:
    - Global minimum at x = (1, 1, ..., 1)
    - f(x) = 0
    """
    return jnp.sum(100.0 * (x[:-1]**2 - x[1:])**2 + (x[:-1] - 1.0)**2)


def _split_by_proportions(x: jnp.ndarray, proportions) -> tuple[jnp.ndarray, ...]:
    """
    Split x into contiguous blocks according to proportions.

    JAX-tracing-safe version:
    - split sizes are computed with Python/NumPy from the static shape x.shape[0]
    - no int(...) conversion from JAX tracer values
    """
    D = x.shape[0]  # static Python int for fixed-shape JAX arrays

    sizes = [int(np.floor(float(p) * D)) for p in proportions]
    sizes[-1] = D - sum(sizes[:-1])

    cuts = np.cumsum(sizes[:-1]).astype(int).tolist()
    return tuple(jnp.split(x, cuts))


def hybrid_1(x: jnp.ndarray) -> jnp.ndarray:
    """
    CEC Hybrid Function 1 benchmark function.

    Source:
    - CEC 2017: "Hybrid Function 1", Function 10 (F10)

    Base formula:
        F(x) = g1(z1) + g2(z2) + g3(z3)

        where:
        - N = 3
        - p = [0.2, 0.4, 0.4]
        - g1 = Zakharov Function
        - g2 = Rosenbrock Function
        - g3 = Rastrigin’s Function

    Type:
    - Hybrid function: different variable subcomponents use different base functions.
    - Non-separable in the official CEC benchmark version.
    - Mixed landscape structure: combines smooth polynomial behavior, curved-valley
      behavior, and highly multimodal periodic behavior.

    Optimum of the base function:
    - For this base implementation, the global minimum is obtained when:
        z1 = 0   for the Zakharov part,
        z2 = 1   for the Rosenbrock part,
        z3 = 0   for the Rastrigin part.
    - The minimum value is f(x) = 0.
    """
    z1, z2, z3 = _split_by_proportions(x, [0.2, 0.4, 0.4])
    return zakharov(z1) + rosenbrock(z2) + rastrigin(z3)

# Hybrid 4
def ackley(x: jnp.ndarray) -> jnp.ndarray:
    """
    Ackley benchmark function.

    Source:
    - Base formula in CEC 2017 basic functions as Ackley’s Function

    Base formula:
        f(x) = -20*exp(-0.2*sqrt((1/D)*sum_{i=1..D} x_i^2)) - exp((1/D)*sum_{i=1..D} cos(2*pi*x_i)) + 20 + e

    Type:
    - Multimodal.
    - Non-separable.
    - Oscillatory landscape with a central basin.

    Optimum of the base function:
    - Global minimum at x = 0
    - f(x) = 0
    """
    D = x.shape[0]
    s1 = jnp.sum(x**2)
    s2 = jnp.sum(jnp.cos(2.0 * jnp.pi * x))
    return -20.0 * jnp.exp(-0.2 * jnp.sqrt(s1 / D)) - jnp.exp(s2 / D) + 20.0 + jnp.e


def schaffers_f7(x: jnp.ndarray) -> jnp.ndarray:
    """
    Schaffer's F7 benchmark function.

    Source:
    - Used as a base component in CEC 2017 hybrid/composition functions
    - Base formula appears in CEC 2017 basic functions as Schaffer’s F7 Function

    Base formula:
        s_i = sqrt(x_i^2 + x_{i+1}^2),  i = 1,...,D-1

        f(x) = ( 1/(D-1) * sum_{i=1..D-1} [ sqrt(s_i) + sqrt(s_i)*sin(50*s_i^0.2)^2 ] )^2

    Type:
    - Multimodal.
    - Non-separable.
    - Rugged and oscillatory.

    Optimum of the base function:
    - Global minimum at x = 0
    - f(x) = 0
    """
    s = jnp.sqrt(x[:-1]**2 + x[1:]**2)
    term = jnp.sqrt(s) + jnp.sqrt(s) * jnp.sin(50.0 * s**0.2)**2
    return (jnp.sum(term) / (x.shape[0] - 1))**2


def high_conditioned_elliptic(x: jnp.ndarray) -> jnp.ndarray:
    """
    High Conditioned Elliptic benchmark function.

    Source:
    - Base formula appears in CEC 2017 basic functions as High Conditioned Elliptic Function

    Base formula:
        f(x) = sum_{i=1..D} 10^( 6 * (i-1)/(D-1) ) * x_i^2

    Type:
    - Unimodal.
    - Strongly ill-conditioned.
    - Separable in base form, non-separable in official rotated CEC versions.

    Optimum of the base function:
    - Global minimum at x = 0
    - f(x) = 0
    """
    D = x.shape[0]
    if D == 1:
        return x[0] ** 2
    i = jnp.arange(D, dtype=x.dtype)
    weights = 10.0 ** (6.0 * i / (D - 1))
    return jnp.sum(weights * x**2)


def hybrid_4(x: jnp.ndarray) -> jnp.ndarray:
    """
    CEC Hybrid Function 4 benchmark function.

    Source:
    - CEC 2017: "Hybrid Function 4", Function 13 (F13)

    Base formula:
        F(x) = g1(z1) + g2(z2) + g3(z3) + g4(z4)

        where:
        - N = 4
        - p = [0.2, 0.2, 0.2, 0.4]
        - g1 = High Conditioned Elliptic Function
        - g2 = Ackley’s Function
        - g3 = Schaffer’s F7 Function
        - g4 = Rastrigin’s Function

    Type:
    - Hybrid function: different variable subcomponents use different base functions.
    - Non-separable in the official CEC benchmark version.
    - Mixed landscape structure: combines smooth polynomial behavior, oscillatory behavior with a central basin, rugged and oscillatory

    Optimum of the base function:
    - For this base implementation, the global minimum is obtained when:
        z1 = 0   for the elliptic part,
        z2 = 0   for the Ackley part,
        z3 = 0   for the Schaffer’s F7 part,
        z4 = 0   for the Rastrigin part.
    - The minimum value is f(x) = 0.
    """
    z1, z2, z3, z4 = _split_by_proportions(x, [0.2, 0.2, 0.2, 0.4])
    return (high_conditioned_elliptic(z1) + ackley(z2) + schaffers_f7(z3) + rastrigin(z4))
