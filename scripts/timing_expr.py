import time
import itertools
import numpy as np

# Set the performance counter scale
if hasattr(time, "perf_counter_ns"):
    perf_counter = time.perf_counter_ns
    perf_counter_scale = 1000000000
else:
    perf_counter = time.perf_counter
    perf_counter_scale = 1


def current_timing(
    function_sym,
    min_timing_iterations: int = 100,
    min_time_in_sec: int = 3,
):
    start_time = perf_counter()
    end_time = perf_counter()

    iterations = 0
    while ((end_time - start_time) / perf_counter_scale) < min_time_in_sec:
        for _ in itertools.repeat(None, min_timing_iterations):
            iterations += 1
            function_sym()
        end_time = perf_counter()

    elapsed_time = (end_time - start_time) / perf_counter_scale
    mean_elapsed_time = elapsed_time / iterations
    return mean_elapsed_time


def new_timing(
    function_sym,
    min_time_in_sec: int = 3,
):

    trials = 1
    time = 0
    secs_elapsed = 0

    while True:
      start_time = perf_counter()
      for _ in itertools.repeat(None, trials):
        function_sym()
      end_time = perf_counter()
      time = end_time - start_time
      secs_elapsed += time / perf_counter_scale
      if secs_elapsed > min_time_in_sec:
        break
      trials *= 2
    return time / (perf_counter_scale*trials)


def my_bench_fun(N):
    a = np.random.randn(N)
    b = np.random.randn(N)
    return lambda: a + b


fast_fn = my_bench_fun(2 ** 6)
slow_fn = my_bench_fun(2 ** 20)

current_timing_res = [current_timing(slow_fn, min_time_in_sec=0.25) for i in range(10)]
new_timing_res = [new_timing(slow_fn, min_time_in_sec=0.25) for i in range(10)]

print("== current method ===")
print("variance = ", np.var(current_timing_res), " , mean = ", np.mean(current_timing_res))
print("min = ", np.min(current_timing_res), " , max = ", np.max(current_timing_res))
print(current_timing_res)

print("\n== new method ===")
print("variance = ", np.var(new_timing_res), " , mean = ", np.mean(new_timing_res))
print("min = ", np.min(new_timing_res), " , max = ", np.max(new_timing_res))
print(new_timing_res)

