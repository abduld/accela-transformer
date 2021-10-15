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
    timer = perf_counter
    start_time = timer()
    end_time = timer()

    iterations = 0
    while ((end_time - start_time) / perf_counter_scale) < min_time_in_sec:
        for _ in itertools.repeat(None, min_timing_iterations):
            iterations += 1
            function_sym()
        end_time = timer()

    elapsed_time = (end_time - start_time) / perf_counter_scale
    mean_elapsed_time = elapsed_time / iterations
    return mean_elapsed_time


def new_timing_(
    function_sym,
    min_time_in_sec: int = 3,
):

    timer = perf_counter
    trials = 1
    time = 0
    secs_elapsed = 0

    while True:
        start_time = timer()
        for _ in itertools.repeat(None, trials):
            function_sym()
        end_time = timer()
        time = end_time - start_time
        secs_elapsed += time / perf_counter_scale
        if secs_elapsed > min_time_in_sec:
            break
        trials *= 2
    return time / (perf_counter_scale * trials)


def new_timing(
    function_sym,
    min_time_in_sec: int = 3,
):

    timer = perf_counter
    trials = 1
    prev_durr = 0
    secs_elapsed = 0
    prev_trials = 1
    num_runs = 0
    min_time_in_ns = min_time_in_sec * perf_counter_scale

    while True:
        if secs_elapsed > min_time_in_sec:
            break
        if prev_durr <= 0:
            prev_durr = 1  # do not divide by 0
        trials = (
            prev_trials * min_time_in_ns / prev_durr
        )  # new number of trials is a function of the previous duration
        trials += trials / 5  # perform more trials than we need
        trials = min(trials, 100 * prev_trials)  # don't grow too fast
        trials = max(trials, prev_trials + 1)  # make an advance
        trials = int(min(trials, 10e9))  # don't go into an infinite loop
        prev_trials = trials

        start_time = timer()
        for _ in itertools.repeat(None, trials):
            function_sym()
        end_time = timer()
        prev_durr = end_time - start_time
        num_runs += prev_trials
        secs_elapsed += prev_durr / perf_counter_scale 
    return secs_elapsed / num_runs 


def my_bench_fun(N):
    a = np.random.randn(N)
    b = np.random.randn(N)
    return lambda: a + b


fast_fn = my_bench_fun(2 ** 6)
slow_fn = my_bench_fun(2 ** 20)

current_timing_res = [10e6*current_timing(slow_fn, min_time_in_sec=1) for i in range(20)]
new_timing_res = [10e6*new_timing(slow_fn, min_time_in_sec=1) for i in range(20)]

print("== current method ===")
print(
    "variance = ", np.var(current_timing_res), " , mean = ", np.mean(current_timing_res)
)
print("min = ", np.min(current_timing_res), " , max = ", np.max(current_timing_res))
print(current_timing_res)

print("\n== new method ===")
print("variance = ", np.var(new_timing_res), " , mean = ", np.mean(new_timing_res))
print("min = ", np.min(new_timing_res), " , max = ", np.max(new_timing_res))
print(new_timing_res)


USE_GOOGLE_BENCHMARK = False
if USE_GOOGLE_BENCHMARK:

  import google_benchmark as benchmark

  @benchmark.register
  @benchmark.option.unit(benchmark.kNanosecond)
  def softmax_numpy(state):
      while state:
        fast_fn()


  benchmark.main()