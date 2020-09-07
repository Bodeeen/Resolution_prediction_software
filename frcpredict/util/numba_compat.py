try:
    from numba import jit, prange
except ImportError:
    print("Couldn't import numba; simulations will likely take a very long time.")

    def jit(*_args, **_kwargs):
        def pass_through(f):
            return f
        return pass_through

    prange = range
