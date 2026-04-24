import time
import runpy

t0 = time.perf_counter()
runpy.run_path("linear_models_selection.py", run_name="__main__")
t1 = time.perf_counter()

print(f"Elapsed: {t1 - t0:.2f} s")
