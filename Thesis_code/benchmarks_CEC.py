from pathlib import Path
import numpy as np
import jax.numpy as jnp

### IDS for the reduced CEC suite.
CEC_ZAKHAROV     = 1
CEC_RASTRIGIN    = 2
CEC_SCAFFERS_F6  = 3
CEC_LEVY         = 4
CEC_HYBRID_1     = 5  
CEC_HYBRID_6     = 8     
CEC_HYBRID_10    = 10
CEC_COMP_2       = 11  
CEC_COMP_4       = 12    
CEC_COMP_7       = 14  
CEC_COMP_8       = 15 


### HELPER FUNCTIONS
def get_cec_suite_ids():
    """
    Return the tuple of benchmark IDs used in the reduced CEC suite.
    """
    return (
        CEC_ZAKHAROV,
        CEC_RASTRIGIN,
        CEC_SCAFFERS_F6,
        CEC_LEVY,
        CEC_HYBRID_1,
        CEC_HYBRID_6,
        CEC_HYBRID_10,
        CEC_COMP_2,
        CEC_COMP_4,
        CEC_COMP_7,
        CEC_COMP_8,
    )


def get_objective_name(objective_id: int) -> str:
    names = {
        CEC_ZAKHAROV: "CEC Zakharov",
        CEC_RASTRIGIN: "CEC Rastrigin",
        CEC_SCAFFERS_F6: "CEC Expanded Scaffer's F6",
        CEC_LEVY: "CEC Levy",
        CEC_HYBRID_1: "CEC Hybrid Function 1",
        CEC_HYBRID_6: "CEC Hybrid Function 6",
        CEC_HYBRID_10: "CEC Hybrid Function 10",
        CEC_COMP_2: "CEC Composition Function 2",
        CEC_COMP_4: "CEC Composition Function 4",
        CEC_COMP_7: "CEC Composition Function 7",
        CEC_COMP_8: "CEC Composition Function 8",
    }

    try:
        return names[objective_id]
    except KeyError as exc:
        raise ValueError(f"Unknown CEC objective_id: {objective_id}") from exc


def get_objective_family(objective_id: int) -> str:
    families = {
        CEC_ZAKHAROV: "unimodal",
        CEC_RASTRIGIN: "multimodal",
        CEC_SCAFFERS_F6: "multimodal",
        CEC_LEVY: "multimodal",
        CEC_HYBRID_1: "hybrid",
        CEC_HYBRID_6: "hybrid",
        CEC_HYBRID_10: "hybrid",
        CEC_COMP_2: "composition",
        CEC_COMP_4: "composition",
        CEC_COMP_7: "composition",
        CEC_COMP_8: "composition",
    }

    try:
        return families[objective_id]
    except KeyError as exc:
        raise ValueError(f"Unknown CEC objective_id: {objective_id}") from exc


def get_default_bounds(objective_id: int, dim: int):
    """
    Return default bounds [a_i, b_i] for each dimension, shape (D, 2).

    For the reduced CEC suite, all functions use [-100, 100]^D
    by default in the official reports.
    """
    lo = -100.0
    hi = 100.0
    return jnp.tile(jnp.array([[lo, hi]], dtype=jnp.float64), (dim, 1))


### CEC SETTINGS

CEC_DIM = 30
DATA_DIR = Path(__file__).resolve().parent / "Shift_data"


def _load_txt_array(path: Path, dtype=jnp.float64) -> jnp.ndarray:
    """
    Load a numeric txt file and return it as a JAX array.
    """
    arr = np.loadtxt(path)
    return jnp.array(arr, dtype=dtype)


def load_shift_vector(func_num: int, dim: int = CEC_DIM, dtype=jnp.float64) -> jnp.ndarray:
    """
    Load the shift vector from:
        shift_data_<func_num>.txt

    The official file may contain more values than needed; we take the first dim.
    Returned shape: (dim,)
    """
    path = DATA_DIR / f"shift_data_{func_num}.txt"
    arr = _load_txt_array(path, dtype=dtype).reshape(-1)
    return arr[:dim]


def load_rotation_matrix(func_num: int, dim: int = CEC_DIM, dtype=jnp.float64) -> jnp.ndarray:
    """
    Load the rotation matrix from:
        M_<func_num>_D<dim>.txt

    Returned shape: (dim, dim)
    """
    path = DATA_DIR / f"M_{func_num}_D{dim}.txt"
    arr = _load_txt_array(path, dtype=dtype)
    return arr.reshape(dim, dim)


def load_shuffle_vector(func_num: int, dim: int = CEC_DIM) -> jnp.ndarray:
    """
    Load the shuffle vector from:
        shuffle_data_<func_num>_D<dim>.txt

    CEC shuffle files are stored as 1-based indices.
    Python/JAX uses 0-based indexing, so we convert 1..D to 0..D-1.
    """
    path = DATA_DIR / f"shuffle_data_{func_num}_D{dim}.txt"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    shuffle = np.loadtxt(path, dtype=int).reshape(-1)

    if shuffle.min() == 1 and shuffle.max() == dim:
        shuffle = shuffle - 1

    return jnp.asarray(shuffle[:dim], dtype=jnp.int32)


def load_composition_shift_matrix(func_num: int, dim: int, n_components: int, dtype=jnp.float64) -> jnp.ndarray:
    """
    Load composition shift vectors.

    Expected shape:
        (n_components, dim)

    File:
        shift_data_<func_num>.txt
    """
    path = DATA_DIR / f"shift_data_{func_num}.txt"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)

    needed = n_components * dim

    if arr.size < needed:
        raise ValueError(f"{path.name} has {arr.size} values, but {needed} are needed " f"for n_components={n_components}, dim={dim}.")

    return jnp.asarray(arr[:needed].reshape(n_components, dim), dtype=dtype)


def load_composition_rotation_matrices(func_num: int, dim: int, n_components: int, dtype=jnp.float64) -> jnp.ndarray:
    """
    Load composition rotation matrices.

    Expected logical shape:
        (n_components, dim, dim)

    File:
        M_<func_num>_D<dim>.txt
    """
    path = DATA_DIR / f"M_{func_num}_D{dim}.txt"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)

    needed = n_components * dim * dim
    if arr.size < needed:
        raise ValueError(
            f"{path.name} has {arr.size} values, but {needed} are needed "
            f"for n_components={n_components}, dim={dim}."
        )

    return jnp.asarray(arr[:needed].reshape(n_components, dim, dim), dtype=dtype)


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


def _composition_weights(x: jnp.ndarray, shifts: jnp.ndarray, sigmas: jnp.ndarray, eps: float = 1e-300) -> jnp.ndarray:
    """
    Compute CEC-style composition weights.

    x:      shape (D,)
    shifts: shape (N, D)
    sigmas: shape (N,)
    """
    D = x.shape[0]

    diff = x[None, :] - shifts
    d2 = jnp.sum(diff ** 2, axis=1)

    # Standard CEC-style weight:
    # w_i = 1/sqrt(sum((x-o_i)^2)) * exp(-sum((x-o_i)^2)/(2*D*sigma_i^2))
    raw = (1.0 / jnp.sqrt(jnp.maximum(d2, eps))) * jnp.exp(-d2 / (2.0 * D * (sigmas ** 2)))

    # If exactly on one component optimum, make that component dominate.
    hit = d2 <= 1e-14
    any_hit = jnp.any(hit)

    hit_weights = hit.astype(x.dtype)
    hit_weights = hit_weights / jnp.maximum(jnp.sum(hit_weights), eps)

    raw_sum = jnp.sum(raw)
    normal_weights = raw / jnp.maximum(raw_sum, eps)

    return jnp.where(any_hit, hit_weights, normal_weights)


def cec_composition_generic(x: jnp.ndarray, params: dict, components, sigmas, lambdas, inner_biases) -> jnp.ndarray:
    """
    Generic CEC composition function.

    F(x) = sum_i w_i * (lambda_i * g_i(x) + inner_bias_i) + global_bias
    """
    shifts = params["shift"]      # (N, D)
    rots = params["rot"]          # (N, D, D)

    sigmas = jnp.asarray(sigmas, dtype=x.dtype)
    lambdas = jnp.asarray(lambdas, dtype=x.dtype)
    inner_biases = jnp.asarray(inner_biases, dtype=x.dtype)

    weights = _composition_weights(x, shifts, sigmas)

    vals = []

    for i, component in enumerate(components):
        z = rots[i] @ (x - shifts[i])
        vals.append(_component_eval(component, z))

    vals = jnp.stack(vals)

    composed = jnp.sum(weights * (lambdas * vals + inner_biases))
    return composed + params["bias"]


def discus(x: jnp.ndarray) -> jnp.ndarray:
    return 1e6 * x[0] ** 2 + jnp.sum(x[1:] ** 2)


def schwefel_component(x: jnp.ndarray) -> jnp.ndarray:
    """
    Modified Schwefel component in CEC style.
    Input x is assumed to already be the hybrid block.
    """
    z = 10.0 * x + 4.209687462275036e2
    D = x.shape[0]

    abs_z = jnp.abs(z)

    case_mid = z * jnp.sin(jnp.sqrt(abs_z))

    z_mod_pos = 500.0 - jnp.mod(z, 500.0)
    case_pos = z_mod_pos * jnp.sin(jnp.sqrt(jnp.abs(z_mod_pos))) - ((z - 500.0) / 100.0) ** 2 / D

    z_mod_neg = jnp.mod(jnp.abs(z), 500.0) - 500.0
    case_neg = z_mod_neg * jnp.sin(jnp.sqrt(jnp.abs(z_mod_neg))) - ((z + 500.0) / 100.0) ** 2 / D

    g = jnp.where(z > 500.0, case_pos, jnp.where(z < -500.0, case_neg, case_mid))

    return 418.9828872724338 * D - jnp.sum(g)


def rosenbrock_component(x: jnp.ndarray) -> jnp.ndarray:
    return rosenbrock((2.048 / 100.0) * x + 1.0)


def rastrigin_component(x: jnp.ndarray) -> jnp.ndarray:
    return rastrigin((5.12 / 100.0) * x)


def griewank_component(x: jnp.ndarray) -> jnp.ndarray:
    z = (600.0 / 100.0) * x
    i = jnp.arange(1, x.shape[0] + 1, dtype=x.dtype)
    return 1.0 + jnp.sum(z ** 2) / 4000.0 - jnp.prod(jnp.cos(z / jnp.sqrt(i)))


def katsuura_component(x: jnp.ndarray) -> jnp.ndarray:
    z = (5.0 / 100.0) * x
    D = z.shape[0]
    i = jnp.arange(1, D + 1, dtype=z.dtype)
    j = jnp.arange(1, 33, dtype=z.dtype)

    two_j = 2.0 ** j
    vals = jnp.abs(two_j[None, :] * z[:, None] - jnp.floor(two_j[None, :] * z[:, None] + 0.5)) / two_j[None, :]
    temp = jnp.sum(vals, axis=1)

    prod_term = jnp.prod((1.0 + i * temp) ** (10.0 / (D ** 1.2)))
    return (10.0 / (D ** 2)) * prod_term - (10.0 / (D ** 2))


def happycat_component(x: jnp.ndarray) -> jnp.ndarray:
    z = (5.0 / 100.0) * x - 1.0
    D = z.shape[0]
    r2 = jnp.sum(z ** 2)
    sum_z = jnp.sum(z)
    return jnp.abs(r2 - D) ** 0.25 + (0.5 * r2 + sum_z) / D + 0.5


def hgbat_component(x: jnp.ndarray) -> jnp.ndarray:
    z = (5.0 / 100.0) * x - 1.0
    D = z.shape[0]
    r2 = jnp.sum(z ** 2)
    sum_z = jnp.sum(z)
    return jnp.abs(r2 ** 2 - sum_z ** 2) ** 0.5 + (0.5 * r2 + sum_z) / D + 0.5


def _split_by_proportions_ceil(x: jnp.ndarray, proportions) -> tuple[jnp.ndarray, ...]:
    """
    CEC-style split: use ceil for all blocks except the last.
    For D=30 and D=50 this usually matches the intended proportions exactly.
    """
    D = x.shape[0]
    sizes = [int(np.ceil(float(p) * D)) for p in proportions[:-1]]
    sizes.append(D - sum(sizes))

    cuts = np.cumsum(sizes[:-1]).astype(int).tolist()
    return tuple(jnp.split(x, cuts))


def _component_eval(name: str, block: jnp.ndarray) -> jnp.ndarray:
    if name == "bent_cigar":
        return bent_cigar(block)
    if name == "elliptic":
        return high_conditioned_elliptic(block)
    if name == "discus":
        return discus(block)
    if name == "zakharov":
        return zakharov(block)
    if name == "rosenbrock":
        return rosenbrock_component(block)
    if name == "rastrigin":
        return rastrigin_component(block)
    if name == "ackley":
        return ackley(block)
    if name == "schaffer_f7":
        return schaffers_f7(block)
    if name == "escaffer6":
        return scaffers_f6(block)
    if name == "schwefel":
        return schwefel_component(block)
    if name == "katsuura":
        return katsuura_component(block)
    if name == "happycat":
        return happycat_component(block)
    if name == "hgbat":
        return hgbat_component(block)
    if name == "griewank":
        return griewank_component(block)

    raise ValueError(f"Unknown hybrid component: {name}")


def cec_hybrid_generic(x: jnp.ndarray, params: dict, proportions, components) -> jnp.ndarray:
    z = transform_shift_rotate(x, params["shift"], params["rot"])
    z = z[params["shuffle"]]
    blocks = _split_by_proportions_ceil(z, proportions)

    val = jnp.asarray(0.0, dtype=x.dtype)
    for block, component in zip(blocks, components):
        val = val + _component_eval(component, block)
    return val + params["bias"]


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
    z = z[params["shuffle"]]
    z1, z2, z3 = _split_by_proportions(z, [0.2, 0.4, 0.4])
    val = (zakharov(z1) + rosenbrock(z2 + 1.0) + rastrigin(z3))

    return val + params["bias"]

def cec_hybrid_6(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    # CEC16: Hybrid Function 6, N=4
    return cec_hybrid_generic(x, params, proportions=[0.2, 0.2, 0.3, 0.3], components=["escaffer6", "hgbat", "rosenbrock", "schwefel"])


def cec_hybrid_10(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    # CEC20: Hybrid Function 10, N=6
    return cec_hybrid_generic(x, params, proportions=[0.1, 0.1, 0.2, 0.2, 0.2, 0.2], components=["happycat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer_f7"])


def cec_composition_2(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    # CEC22 / Composition Function 2, N=3
    return cec_composition_generic(x, params, components=["rastrigin", "griewank", "schwefel"], sigmas=[10.0, 20.0, 30.0], lambdas=[1.0, 10.0, 1.0], inner_biases=[0.0, 100.0, 200.0])


def cec_composition_4(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    # CEC24 / Composition Function 4, N=4
    return cec_composition_generic(x, params, components=["ackley", "elliptic", "griewank", "rastrigin"], sigmas=[10.0, 20.0, 30.0, 40.0], lambdas=[10.0, 1e-6, 10.0, 1.0], inner_biases=[0.0, 100.0, 200.0, 300.0])


def cec_composition_7(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    # CEC27 / Composition Function 7, N=6
    return cec_composition_generic(x, params, components=["hgbat", "rastrigin", "schwefel", "bent_cigar", "elliptic", "escaffer6"], sigmas=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], lambdas=[10.0, 10.0, 2.5, 1e-26, 1e-6, 5e-4], inner_biases=[0.0, 100.0, 200.0, 300.0, 400.0, 500.0])


def cec_composition_8(x: jnp.ndarray, params: dict) -> jnp.ndarray:
    # CEC28 / Composition Function 8, N=6
    return cec_composition_generic(x, params, components=["ackley", "griewank", "discus", "rosenbrock", "happycat", "escaffer6"], sigmas=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], lambdas=[10.0, 10.0, 1e-6, 1.0, 1.0, 5e-4], inner_biases=[0.0, 100.0, 200.0, 300.0, 400.0, 500.0])


def get_cec_params(dim: int = CEC_DIM, dtype=jnp.float64):
    """
    Load transformation data for the active 11-function thesis CEC suite.

    F3  - Zakharov
    F5  - Rastrigin
    F6  - Expanded Scaffer's F6
    F9  - Levy
    F11 - Hybrid Function 1
    F16 - Hybrid Function 6
    F20 - Hybrid Function 10
    F22 - Composition Function 2
    F24 - Composition Function 4
    F27 - Composition Function 7
    F28 - Composition Function 8
    """
    return {

    CEC_ZAKHAROV: {
        "func_num": 3,
        "shift": load_shift_vector(3, dim, dtype),
        "rot": load_rotation_matrix(3, dim, dtype),
        "bias": jnp.array(300.0, dtype=dtype),
    },

    CEC_RASTRIGIN: {
        "func_num": 5,
        "shift": load_shift_vector(5, dim, dtype),
        "rot": load_rotation_matrix(5, dim, dtype),
        "bias": jnp.array(500.0, dtype=dtype),
    },

    CEC_SCAFFERS_F6: {
        "func_num": 6,
        "shift": load_shift_vector(6, dim, dtype),
        "rot": load_rotation_matrix(6, dim, dtype),
        "bias": jnp.array(600.0, dtype=dtype),
    },

    CEC_LEVY: {
        "func_num": 9,
        "shift": load_shift_vector(9, dim, dtype),
        "rot": load_rotation_matrix(9, dim, dtype),
        "bias": jnp.array(900.0, dtype=dtype),
    },

    CEC_HYBRID_1: {
        "func_num": 11,
        "shift": load_shift_vector(11, dim, dtype),
        "rot": load_rotation_matrix(11, dim, dtype),
        "shuffle": load_shuffle_vector(11, dim),
        "bias": jnp.array(1100.0, dtype=dtype),
    },

    CEC_HYBRID_6: {
        "func_num": 16,
        "shift": load_shift_vector(16, dim, dtype),
        "rot": load_rotation_matrix(16, dim, dtype),
        "shuffle": load_shuffle_vector(16, dim),
        "bias": jnp.array(1600.0, dtype=dtype),
    },


    CEC_HYBRID_10: {
        "func_num": 20,
        "shift": load_shift_vector(20, dim, dtype),
        "rot": load_rotation_matrix(20, dim, dtype),
        "shuffle": load_shuffle_vector(20, dim),
        "bias": jnp.array(2000.0, dtype=dtype),
    },
        CEC_COMP_2: {
        "func_num": 22,
        "shift": load_composition_shift_matrix(22, dim, n_components=3, dtype=dtype),
        "rot": load_composition_rotation_matrices(22, dim, n_components=3, dtype=dtype),
        "bias": jnp.array(2200.0, dtype=dtype),
    },

    CEC_COMP_4: {
        "func_num": 24,
        "shift": load_composition_shift_matrix(24, dim, n_components=4, dtype=dtype),
        "rot": load_composition_rotation_matrices(24, dim, n_components=4, dtype=dtype),
        "bias": jnp.array(2400.0, dtype=dtype),
    },

    CEC_COMP_7: {
        "func_num": 27,
        "shift": load_composition_shift_matrix(27, dim, n_components=6, dtype=dtype),
        "rot": load_composition_rotation_matrices(27, dim, n_components=6, dtype=dtype),
        "bias": jnp.array(2700.0, dtype=dtype),
    },

    CEC_COMP_8: {
        "func_num": 28,
        "shift": load_composition_shift_matrix(28, dim, n_components=6, dtype=dtype),
        "rot": load_composition_rotation_matrices(28, dim, n_components=6, dtype=dtype),
        "bias": jnp.array(2800.0, dtype=dtype),
    },
    }


def evaluate_objective(x: jnp.ndarray, objective_id: int) -> jnp.ndarray:
    """
    Evaluate a single point x for the given CEC objective_id.

    x: shape (D,)
    objective_id: one of the CEC_* IDs above
    """
    params = get_cec_params(dim=x.shape[0], dtype=x.dtype)

    if objective_id == CEC_ZAKHAROV:
        return cec_zakharov(x, params[CEC_ZAKHAROV])
    elif objective_id == CEC_RASTRIGIN:
        return cec_rastrigin(x, params[CEC_RASTRIGIN])
    elif objective_id == CEC_SCAFFERS_F6:
        return cec_scaffers_f6(x, params[CEC_SCAFFERS_F6])
    elif objective_id == CEC_LEVY:
        return cec_levy(x, params[CEC_LEVY])
    elif objective_id == CEC_HYBRID_1:
        return cec_hybrid_1(x, params[CEC_HYBRID_1])
    elif objective_id == CEC_HYBRID_6:
        return cec_hybrid_6(x, params[CEC_HYBRID_6])
    elif objective_id == CEC_HYBRID_10:
        return cec_hybrid_10(x, params[CEC_HYBRID_10])
    elif objective_id == CEC_COMP_2:
        return cec_composition_2(x, params[CEC_COMP_2])
    elif objective_id == CEC_COMP_4:
        return cec_composition_4(x, params[CEC_COMP_4])
    elif objective_id == CEC_COMP_7:
        return cec_composition_7(x, params[CEC_COMP_7])
    elif objective_id == CEC_COMP_8:
        return cec_composition_8(x, params[CEC_COMP_8])
    else:
        raise ValueError(f"Unknown objective_id: {objective_id}")


### CEC BENCHMARK FUNCTIONS

# UNIMODAL

def bent_cigar(x: jnp.ndarray) -> jnp.ndarray:
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

