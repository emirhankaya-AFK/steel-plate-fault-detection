"""Command Line Interface for Steel Plate Fault Detection."""

import argparse
import json
import logging
import sys
from pathlib import Path

import uvicorn
import yaml

from steel_faults.benchmark.latency import benchmark_inference_latency
from steel_faults.data.downloader import DatasetDownloader
from steel_faults.data.synthetic import save_synthetic_fixture
from steel_faults.pipeline.runner import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("steel_faults.cli")


def main() -> None:
    """CLI main entry point."""
    parser = argparse.ArgumentParser(
        description="Steel Plate Fault Detection & Quality Monitoring CLI",
        prog="steel-faults",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run model training and evaluation pipeline")
    train_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic data for offline execution",
    )
    train_parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Custom output path for metrics JSON artifact",
    )

    # Download command
    subparsers.add_parser("download", help="Download raw UCI dataset into cache")

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark", help="Measure inference latency and throughput"
    )
    bench_parser.add_argument(
        "--requests",
        type=int,
        default=500,
        help="Number of single requests for latency benchmark",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run FastAPI serving server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Fixture command
    fix_parser = subparsers.add_parser(
        "generate-fixture", help="Generate offline synthetic CSV fixture"
    )
    fix_parser.add_argument(
        "--output",
        type=str,
        default="tests/fixtures/synthetic_steel_faults.csv",
        help="Target fixture path",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "download":
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        downloader = DatasetDownloader(
            url=cfg["data"]["raw_data_url"],
            cache_dir=cfg["data"]["raw_data_dir"],
        )
        df = downloader.download_and_extract()
        logger.info(
            "Dataset downloaded successfully: %d records, %d columns", len(df), len(df.columns)
        )

    elif args.command == "generate-fixture":
        path = save_synthetic_fixture(args.output)
        logger.info("Synthetic fixture generated at %s", path)

    elif args.command == "train":
        logger.info("Executing training pipeline (synthetic=%s)...", args.synthetic)
        results = run_pipeline(
            config_path=args.config,
            offline_synthetic=args.synthetic,
            output_metrics_path=args.metrics_out,
        )
        test_m = results["test_metrics"]
        print("\n" + "=" * 60)
        print("PIPELINE TEST SET EVALUATION RESULTS:")
        print("=" * 60)
        print(
            f"Baseline (Logistic Reg) Macro-F1: {test_m['baseline_logistic_regression']['macro_f1']:.4f}"
        )
        print(
            f"Random Forest (Uncalibrated) Macro-F1: {test_m['random_forest_uncalibrated']['macro_f1']:.4f}"
        )
        print(
            f"Random Forest (Calibrated) Macro-F1:   {test_m['random_forest_calibrated']['macro_f1']:.4f}"
        )
        print(
            f"Random Forest (Calibrated) Brier Score: {test_m['random_forest_calibrated']['brier_score']:.4f}"
        )
        print(
            f"Random Forest (Calibrated) ECE:         {test_m['random_forest_calibrated']['expected_calibration_error']:.4f}"
        )
        print("=" * 60)

    elif args.command == "benchmark":
        import joblib

        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        m_path = Path(cfg["serving"]["model_artifact_path"])
        if not m_path.exists():
            logger.info("Artifact not found, running quick training pipeline first...")
            _ = run_pipeline(config_path=args.config, offline_synthetic=True)

        bundle = joblib.load(m_path)
        logger.info("Running latency benchmark with %d requests...", args.requests)
        bench_res = benchmark_inference_latency(
            model=bundle["model"],
            classes=bundle["classes"],
            n_single_requests=args.requests,
        )
        print("\n" + "=" * 60)
        print("INFERENCE LATENCY BENCHMARK RESULTS:")
        print("=" * 60)
        print(json.dumps(bench_res, indent=2))
        print("=" * 60)

    elif args.command == "serve":
        uvicorn.run(
            "steel_faults.api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
