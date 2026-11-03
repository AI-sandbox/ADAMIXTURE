import subprocess
import sys
import textwrap


def test_set_cpu_threads_overrides_an_already_started_runtime() -> None:
    script = textwrap.dedent(
        """
        import ctypes
        import adamixture.src.utils_c.cython.tools  # loads the OpenMP runtime
        from adamixture.src.utils import _loaded_thread_libraries, set_cpu_threads

        def _find_get_max_threads():
            for path in list(_loaded_thread_libraries()) + [None]:
                try:
                    lib = ctypes.CDLL(path) if path else ctypes.CDLL(None)
                    fn = getattr(lib, "omp_get_max_threads", None)
                    if fn is not None:
                        fn.restype = ctypes.c_int
                        return fn
                except OSError:
                    continue
            raise RuntimeError("Could not find omp_get_max_threads in any loaded library")

        get = _find_get_max_threads()
        get()
        set_cpu_threads(1)
        assert get() == 1
        set_cpu_threads(2)
        assert get() == 2
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
