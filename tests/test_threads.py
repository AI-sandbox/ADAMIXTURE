import subprocess
import sys
import textwrap


def test_set_cpu_threads_overrides_an_already_started_runtime() -> None:
    script = textwrap.dedent(
        """
        import ctypes
        import adamixture.src.utils_c.cython.tools  # loads the OpenMP runtime
        lib = ctypes.CDLL(None)
        get = lib.omp_get_max_threads
        get.restype = ctypes.c_int
        get()
        from adamixture.src.utils import set_cpu_threads
        set_cpu_threads(1)
        assert get() == 1
        set_cpu_threads(2)
        assert get() == 2
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
