import subprocess
import sys
import textwrap
import pytest


def test_set_cpu_threads_overrides_an_already_started_runtime() -> None:
    script = textwrap.dedent(
        """
        import ctypes
        import sys
        try:
            import adamixture.src.utils_c.cython.sqp  # explicitly uses OpenMP
        except ImportError:
            pass
        try:
            import adamixture.src.utils_c.cython.tools
        except ImportError:
            pass
        from adamixture.src.utils import _loaded_thread_libraries, set_cpu_threads

        def _find_get_max_threads():
            candidates = list(_loaded_thread_libraries())
            if sys.platform != "win32":
                candidates.append(None)
            else:
                candidates.extend(["vcomp140.dll", "vcomp.dll", "libomp.dll", "libgomp-1.dll"])
            for path in candidates:
                try:
                    lib = ctypes.CDLL(path) if path else ctypes.CDLL(None)
                    fn = getattr(lib, "omp_get_max_threads", None)
                    if fn is not None:
                        fn.restype = ctypes.c_int
                        return fn
                except (OSError, TypeError, ValueError):
                    continue
            return None

        get = _find_get_max_threads()
        if get is None:
            print("SKIPPED")
            sys.exit(0)
        get()
        set_cpu_threads(1)
        assert get() == 1
        set_cpu_threads(2)
        assert get() == 2
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    if "SKIPPED" in completed.stdout:
        pytest.skip("OpenMP runtime is not loaded in this environment")
