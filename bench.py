from datetime import datetime
import pandas as pd
import numpy as np
import gc
from typing import Callable
import time
from itertools import product


def measure_once(func: Callable, *args, disable_gc: bool = True) -> float:
    # Mide una sola ejecución de func(*args) con perf_counter (segundos)
    was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()
    t0 = time.perf_counter()
    func(*args)
    t1 = time.perf_counter()
    if disable_gc and was_enabled:
        gc.enable()
    return t1 - t0


def measure_block(func: Callable, *args, batch_size: int = 10, disable_gc: bool = True) -> float:
    # Ejecuta 'batch_size' veces y devuelve tiempo promedio por ejecución
    was_enabled = gc.isenabled()
    if disable_gc:
        gc.disable()
    t0 = time.perf_counter()
    for _ in range(batch_size):
        func(*args)
    t1 = time.perf_counter()
    if disable_gc and was_enabled:
        gc.enable()
    return (t1 - t0) / max(1, batch_size)


def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)


output_file = "bench_fib2.json"
batch_sizes = [1]
ns = [2,4,10,20,30]
combinations = product(ns, batch_sizes)
runs = 500

data = []
for n, batch_size in combinations:
    print()
    print(f"Starting n={n}, batch_size={batch_size} at {datetime.now()}")

    if batch_size==1:
        tic = time.perf_counter()
        times = np.array([measure_once(fib, n)
                        for _ in range(runs)])*1000
        toc = time.perf_counter()
        avg_time_bs1 = np.mean(times)

        print(f"Batch size 1 done in {toc - tic:.2f}s")

        for t in times:
            data.append({'n': n, 'batch_size': batch_size, 'time_ms': t})

    else:
        # print estimaeted time
        es_time = (avg_time_bs1 * runs * batch_size)/1000 / 60.0
        print(f"Estimated time for n={n}, batch_size={batch_size}: {es_time:.2f} minutes")
        tic = time.perf_counter()
        times = np.array([measure_block(fib, n, batch_size=batch_size)
                        for _ in range(runs)])*1000
        toc = time.perf_counter()
        print(f"Batch size {batch_size} done in {toc - tic:.2f}s")
        times = np.array(times)
        
        for t in times:
            data.append({'n': n, 'batch_size': batch_size, 'time_ms': t})

    df = pd.DataFrame(data)
    df.to_json(output_file, orient='records')
