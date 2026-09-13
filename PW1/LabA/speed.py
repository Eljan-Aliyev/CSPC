import time
from decay import simulate, simulate_loop

N0 = 200000
rate = 0.4

# Measure pure Python loop speed
start_loop = time.perf_counter()
simulate_loop(N0, rate)
loop_time = time.perf_counter() - start_loop

# Measure NumPy vectorized speed
start_numpy = time.perf_counter()
simulate(N0, rate)
numpy_time = time.perf_counter() - start_numpy

speedup = loop_time / numpy_time

print(f"loop : {loop_time:.4f} s")
print(f"numpy : {numpy_time:.4f} s")
print(f"speed-up : {speedup:.2f} x faster")
