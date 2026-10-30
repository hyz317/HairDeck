"""Worker processes of the evaluations: started with 'spawn', one thread each.

A forked worker inherits the state of its parent: after the parent has run torch on the CPU (OpenMP), CUDA or EGL, as
scripts/test.py does before it evaluates, forked workers can hang. Spawned workers start a fresh interpreter and import
what they need; they get the parent's environment with OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS set
to 1.
"""
import multiprocessing
import os

THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def worker_pool(workers, initializer=None, initargs=()):
    """multiprocessing Pool of `workers` spawned processes (initializer(*initargs) runs in each)."""
    old = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: '1' for k in THREAD_VARS})
    try:  # the workers start here and read the environment once
        return multiprocessing.get_context('spawn').Pool(workers, initializer=initializer, initargs=initargs)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
