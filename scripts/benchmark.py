"""Standalone benchmark script measuring model inference latency and throughput."""

import json
from pathlib import Path

import joblib
import yaml

from steel_faults.benchmark.latency import benchmark_inference_latency
from steel_faults.pipeline.runner import run_pipeline


def main() -> None:
    config_path = Path("configs/config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_path = Path(cfg["serving"]["model_artifact_path"])
    if not model_path.exists():
        print("Trained model artifact not found. Running training pipeline...")
        run_pipeline(config_path=config_path, offline_synthetic=False)

    print(f"Loading deployment bundle from {model_path}...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    classes = bundle["classes"]

    print(
        "Running inference latency benchmark (500 single requests, batch sizes [1, 10, 50, 100])..."
    )
    results = benchmark_inference_latency(
        model=model,
        classes=classes,
        n_single_requests=500,
        batch_sizes=[1, 10, 50, 100],
    )

    print("\nBenchmark Summary:")
    print(json.dumps(results, indent=2))

    out_file = Path("metrics/latency_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
