"""Tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from steel_faults.data.schema import FEATURE_NAMES, SteelPlateFeatures


def get_valid_sample_dict() -> dict:
    """Helper returning valid features dictionary."""
    base = dict.fromkeys(FEATURE_NAMES, 10.0)
    base["X_Minimum"] = 100.0
    base["X_Maximum"] = 150.0
    base["Y_Minimum"] = 50000.0
    base["Y_Maximum"] = 50100.0
    base["Minimum_of_Luminosity"] = 50.0
    base["Maximum_of_Luminosity"] = 120.0
    base["TypeOfSteel_A300"] = 1
    base["TypeOfSteel_A400"] = 0
    base["Orientation_Index"] = 0.2
    base["Luminosity_Index"] = -0.1
    base["SigmoidOfAreas"] = 0.5
    base["Square_Index"] = 0.8
    base["Edges_Index"] = 0.1
    base["Empty_Index"] = 0.3
    base["Outside_X_Index"] = 0.05
    base["Edges_X_Index"] = 0.7
    base["Edges_Y_Index"] = 0.9
    base["Outside_Global_Index"] = 1.0
    base["LogOfAreas"] = 2.5
    base["Log_X_Index"] = 1.2
    base["Log_Y_Index"] = 1.3
    base["Length_of_Conveyer"] = 1600.0
    base["Steel_Plate_Thickness"] = 80.0
    base["Pixels_Areas"] = 300.0
    base["X_Perimeter"] = 25.0
    base["Y_Perimeter"] = 30.0
    base["Sum_of_Luminosity"] = 25000.0
    return base


def test_valid_steel_plate_features():
    """Test that a valid feature payload is accepted without errors."""
    data = get_valid_sample_dict()
    features = SteelPlateFeatures(**data)
    assert features.X_Minimum == 100.0
    assert features.X_Maximum == 150.0
    assert features.TypeOfSteel_A300 == 1


def test_invalid_x_coordinates():
    """Test that X_Minimum > X_Maximum raises ValidationError."""
    data = get_valid_sample_dict()
    data["X_Minimum"] = 200.0
    data["X_Maximum"] = 100.0
    with pytest.raises(ValidationError, match="X_Minimum .* cannot exceed X_Maximum"):
        SteelPlateFeatures(**data)


def test_invalid_y_coordinates():
    """Test that Y_Minimum > Y_Maximum raises ValidationError."""
    data = get_valid_sample_dict()
    data["Y_Minimum"] = 200000.0
    data["Y_Maximum"] = 100000.0
    with pytest.raises(ValidationError, match="Y_Minimum .* cannot exceed Y_Maximum"):
        SteelPlateFeatures(**data)


def test_invalid_luminosity_range():
    """Test that Minimum_of_Luminosity > Maximum_of_Luminosity raises ValidationError."""
    data = get_valid_sample_dict()
    data["Minimum_of_Luminosity"] = 200.0
    data["Maximum_of_Luminosity"] = 50.0
    with pytest.raises(
        ValidationError, match="Minimum_of_Luminosity .* cannot exceed Maximum_of_Luminosity"
    ):
        SteelPlateFeatures(**data)


def test_conflicting_steel_types():
    """Test that both steel types set to 1 raises ValidationError."""
    data = get_valid_sample_dict()
    data["TypeOfSteel_A300"] = 1
    data["TypeOfSteel_A400"] = 1
    with pytest.raises(
        ValidationError, match="cannot be both TypeOfSteel_A300 and TypeOfSteel_A400"
    ):
        SteelPlateFeatures(**data)


def test_extra_forbidden_fields():
    """Test that unexpected extra fields raise ValidationError."""
    data = get_valid_sample_dict()
    data["Extra_Unknown_Feature"] = 999.0
    with pytest.raises(ValidationError):
        SteelPlateFeatures(**data)
