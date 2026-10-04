"""Deterministic synthetic fixture generator for offline testing and CI."""

from pathlib import Path

import numpy as np
import pandas as pd

from steel_faults.data.schema import TARGET_NAMES


def generate_synthetic_steel_data(
    n_samples: int = 210,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic steel plate data conforming to UCI schema.

    Args:
        n_samples: Number of samples to generate (default 210, 30 per class).
        random_seed: Seed for reproducible data generation.

    Returns:
        pd.DataFrame with 27 features and 7 one-hot defect targets.
    """
    rng = np.random.RandomState(random_seed)

    x_min = rng.uniform(0, 1500, size=n_samples)
    x_len = rng.uniform(5, 100, size=n_samples)
    x_max = x_min + x_len

    y_min = rng.uniform(10000, 5000000, size=n_samples)
    y_len = rng.uniform(5, 200, size=n_samples)
    y_max = y_min + y_len

    pixels_areas = x_len * y_len * rng.uniform(0.3, 0.9, size=n_samples)
    x_perimeter = 2 * (x_len + y_len) * rng.uniform(0.8, 1.2, size=n_samples)
    y_perimeter = 2 * y_len * rng.uniform(0.8, 1.2, size=n_samples)

    min_lum = rng.uniform(30, 100, size=n_samples)
    max_lum = min_lum + rng.uniform(10, 100, size=n_samples)
    sum_lum = pixels_areas * (min_lum + max_lum) / 2.0

    steel_a300 = rng.binomial(1, 0.4, size=n_samples)
    steel_a400 = 1 - steel_a300

    data = {
        "X_Minimum": x_min,
        "X_Maximum": x_max,
        "Y_Minimum": y_min,
        "Y_Maximum": y_max,
        "Pixels_Areas": pixels_areas,
        "X_Perimeter": x_perimeter,
        "Y_Perimeter": y_perimeter,
        "Sum_of_Luminosity": sum_lum,
        "Minimum_of_Luminosity": min_lum,
        "Maximum_of_Luminosity": max_lum,
        "Length_of_Conveyer": rng.uniform(1200, 1800, size=n_samples),
        "TypeOfSteel_A300": steel_a300,
        "TypeOfSteel_A400": steel_a400,
        "Steel_Plate_Thickness": rng.choice(
            [40, 50, 60, 70, 80, 100, 120, 150, 200], size=n_samples
        ),
        "Edges_Index": rng.uniform(0.01, 0.99, size=n_samples),
        "Empty_Index": rng.uniform(0.05, 0.85, size=n_samples),
        "Square_Index": rng.uniform(0.1, 0.95, size=n_samples),
        "Outside_X_Index": rng.uniform(0.001, 0.5, size=n_samples),
        "Edges_X_Index": rng.uniform(0.1, 0.99, size=n_samples),
        "Edges_Y_Index": rng.uniform(0.1, 0.99, size=n_samples),
        "Outside_Global_Index": rng.choice([0.0, 0.5, 1.0], size=n_samples),
        "LogOfAreas": np.log10(np.maximum(pixels_areas, 1.0)),
        "Log_X_Index": np.log10(np.maximum(x_len, 1.0)),
        "Log_Y_Index": np.log10(np.maximum(y_len, 1.0)),
        "Orientation_Index": rng.uniform(-0.95, 0.95, size=n_samples),
        "Luminosity_Index": rng.uniform(-0.9, 0.6, size=n_samples),
        "SigmoidOfAreas": 1.0 / (1.0 + np.exp(-rng.uniform(-2, 2, size=n_samples))),
    }

    df = pd.DataFrame(data)

    # Generate balanced defect classes
    classes = list(TARGET_NAMES)
    labels = [classes[i % len(classes)] for i in range(n_samples)]
    rng.shuffle(labels)

    for target in classes:
        df[target] = [1 if lbl == target else 0 for lbl in labels]

    return df


def save_synthetic_fixture(
    output_path: str | Path = "tests/fixtures/synthetic_steel_faults.csv",
) -> Path:
    """Generate and save synthetic CSV fixture to disk."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_steel_data(n_samples=210, random_seed=42)
    df.to_csv(out_file, index=False)
    return out_file


if __name__ == "__main__":
    save_synthetic_fixture()
