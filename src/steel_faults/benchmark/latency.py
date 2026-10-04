"""Inference latency and throughput benchmarking module."""

import logging
import time
from typing import Any

import numpy as np

from steel_faults.data.schema import FEATURE_NAMES
from steel_faults.data.synthetic import generate_synthetic_steel_data

logger = logging.getLogger(__name__)


def benchmark_inference_latency(
    model: Any,
    classes: list[str],
    n_single_requests: int = 500,
    batch_sizes: list[int] | None = None,
    warmup_requests: int = 50,
) -> dict[str, Any]:
    """Measure single-request and batch inference latency and throughput.

    Args:
        model: Fitted model or CalibratedModelWrapper.
        classes: List of target defect class names.
        n_single_requests: Number of individual single-sample inferences to run.
        batch_sizes: List of batch sizes to evaluate.
        warmup_requests: Warmup passes to prime CPU cache and memory.

    Returns:
        dict[str, Any]: Benchmark results including p50, p95, p99 latencies (ms)
                        and throughput (samples/sec).
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100]

    synthetic_df = generate_synthetic_steel_data(
        n_samples=max(max(batch_sizes), 100), random_seed=42
    )
    sample_features = synthetic_df[list(FEATURE_NAMES)]

    # Warmup
    warmup_row = sample_features.iloc[[0]]
    for _ in range(warmup_requests):
        _ = model.predict_proba(warmup_row)

    # 1. Single sample latency benchmark
    single_latencies_ms: list[float] = []

    for i in range(n_single_requests):
        idx = i % len(sample_features)
        row = sample_features.iloc[[idx]]
        t0 = time.perf_counter()
        _ = model.predict_proba(row)
        t1 = time.perf_counter()
        single_latencies_ms.append((t1 - t0) * 1000.0)

    single_latencies_arr = np.array(single_latencies_ms)
    total_single_time_sec = np.sum(single_latencies_arr) / 1000.0
    throughput_single = (
        float(n_single_requests / total_single_time_sec) if total_single_time_sec > 0 else 0.0
    )

    single_benchmark = {
        "num_requests": n_single_requests,
        "mean_ms": round(float(np.mean(single_latencies_arr)), 3),
        "std_ms": round(float(np.std(single_latencies_arr)), 3),
        "min_ms": round(float(np.min(single_latencies_arr)), 3),
        "p50_ms": round(float(np.percentile(single_latencies_arr, 50)), 3),
        "p90_ms": round(float(np.percentile(single_latencies_arr, 90)), 3),
        "p95_ms": round(float(np.percentile(single_latencies_arr, 95)), 3),
        "p99_ms": round(float(np.percentile(single_latencies_arr, 99)), 3),
        "max_ms": round(float(np.max(single_latencies_arr)), 3),
        "throughput_req_per_sec": round(throughput_single, 1),
    }

    # 2. Batch latency benchmark
    batch_benchmarks: dict[str, Any] = {}
    n_batch_repeats = 100

    for b_size in batch_sizes:
        batch_slice = sample_features.iloc[:b_size]
        batch_times: list[float] = []

        for _ in range(n_batch_repeats):
            t0 = time.perf_counter()
            _ = model.predict_proba(batch_slice)
            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000.0)

        batch_times_arr = np.array(batch_times)
        mean_batch_ms = float(np.mean(batch_times_arr))
        per_sample_ms = mean_batch_ms / b_size

        batch_benchmarks[f"batch_size_{b_size}"] = {
            "mean_batch_ms": round(mean_batch_ms, 3),
            "per_sample_ms": round(per_sample_ms, 4),
            "p95_batch_ms": round(float(np.percentile(batch_times_arr, 95)), 3),
            "samples_per_sec": round(float(1000.0 / per_sample_ms), 1)
            if per_sample_ms > 0
            else 0.0,
        }

    return {
        "single_request": single_benchmark,
        "batch_requests": batch_benchmarks,
    }
