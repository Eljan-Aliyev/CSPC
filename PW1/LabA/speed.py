import time
from decay import simulate, simulate_loop

# Benchmark configuration
N0 = 200000
rate = 0.4

# Measure pure-Python loop implementation
start_loop = time.perf_counter()
simulate_loop(N0, rate)
end_loop = time.perf_counter()
loop_time = end_loop - start_loop

# Measure NumPy implementation
start_numpy = time.perf_counter()
simulate(N0, rate)
end_numpy = time.perf_counter()
numpy_time = end_numpy - start_numpy

# Calculate speedup factor
speedup = loop_time / numpy_time

print(f"Pure-Python loop time: {loop_time:.6f} seconds")
print(f"NumPy vector time:     {numpy_time:.6f} seconds")
print(f"Speed-up factor:       NumPy is {speedup:.2f}x faster")
