"""Límite de hilos para mantener la PC estable. Llamar al inicio de cada notebook."""
import os

_LIMITER = None


def limit_threads(cfg: dict) -> int:
    """Limita los hilos de BLAS/OpenMP, scikit-learn y TensorFlow. Devuelve n_threads."""
    global _LIMITER
    n = int(cfg["runtime"]["n_threads"])
    inter = int(cfg["runtime"].get("inter_op_threads", 2))

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)

    try:                                   # NumPy / scikit-learn (la logística usa BLAS)
        from threadpoolctl import threadpool_limits
        _LIMITER = threadpool_limits(limits=n)
    except ImportError:
        pass

    try:                                   # TensorFlow: debe fijarse antes de la primera operación
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
    except RuntimeError:
        print("TensorFlow ya estaba inicializado: reinicia el kernel y llama a limit_threads primero.")
    except ImportError:
        pass
    return n