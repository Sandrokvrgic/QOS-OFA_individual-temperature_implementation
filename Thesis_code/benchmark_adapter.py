import benchmarks_2D as b2d
import benchmarks_CEC as bcec
import jax.numpy as jnp

def get_backend(benchmark_family: str):
    if benchmark_family == "2d":
        return b2d
    elif benchmark_family == "cec":
        return bcec
    else:
        raise ValueError(f"Unknown benchmark_family: {benchmark_family}")

def get_bounds(benchmark_family: str, objective_id: int, dim: int | None = None):
    backend = get_backend(benchmark_family)
    if benchmark_family == "2d":
        return backend.get_default_bounds(objective_id)
    elif benchmark_family == "cec":
        if dim is None:
            raise ValueError("CEC benchmarks require dim.")
        return backend.get_default_bounds(objective_id, dim)
    
def get_objective_name(benchmark_family: str, objective_id: int):
    backend = get_backend(benchmark_family)

    if hasattr(backend, "get_objective_name"):
        return backend.get_objective_name(objective_id)

    if benchmark_family == "2d":
        names = {
            b2d.SIX_HUMP_WIDE: "Six-Hump Wide",
            b2d.SIX_HUMP_CLASSIC: "Six-Hump Classic",
            b2d.SIX_HUMP_LOW_CONTRAST: "Six-Hump Low Contrast",
            b2d.SIX_HUMP_HIGH_CONTRAST: "Six-Hump High Contrast",
        }
        return names[objective_id]

    return f"objective_{objective_id}"


def make_objective_fn(
    benchmark_family: str,
    objective_id: int,
    dim: int | None = None,
    dtype=jnp.float32,
):
    """
    Return a function objective_fn(x) that evaluates one point x.

    This avoids hardwiring the algorithm to either benchmarks_2D or benchmarks_CEC.
    For CEC, parameters are loaded once here, not inside every objective evaluation.
    """
    if benchmark_family == "2d":
        return lambda x: b2d.evaluate_objective(x, objective_id)

    elif benchmark_family == "cec":
        if dim is None:
            raise ValueError("CEC benchmarks require dim.")

        params = bcec.get_cec_params(dim=dim, dtype=dtype)
        p = params[objective_id]

        if objective_id == bcec.CEC_BENT_CIGAR:
            return lambda x: bcec.cec_bent_cigar(x, p)
        elif objective_id == bcec.CEC_ZAKHAROV:
            return lambda x: bcec.cec_zakharov(x, p)
        elif objective_id == bcec.CEC_RASTRIGIN:
            return lambda x: bcec.cec_rastrigin(x, p)
        elif objective_id == bcec.CEC_SCAFFERS_F6:
            return lambda x: bcec.cec_scaffers_f6(x, p)
        elif objective_id == bcec.CEC_LEVY:
            return lambda x: bcec.cec_levy(x, p)
        elif objective_id == bcec.CEC_HYBRID_1:
            return lambda x: bcec.cec_hybrid_1(x, p)
        elif objective_id == bcec.CEC_HYBRID_4:
            return lambda x: bcec.cec_hybrid_4(x, p)
        else:
            raise ValueError(f"Unknown CEC objective_id: {objective_id}")

    else:
        raise ValueError(f"Unknown benchmark_family: {benchmark_family}")