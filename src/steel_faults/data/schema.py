"""Schema definitions and Pydantic models for Steel Plate Fault Detection."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

FEATURE_NAMES: tuple[str, ...] = (
    "X_Minimum",
    "X_Maximum",
    "Y_Minimum",
    "Y_Maximum",
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
    "Sum_of_Luminosity",
    "Minimum_of_Luminosity",
    "Maximum_of_Luminosity",
    "Length_of_Conveyer",
    "TypeOfSteel_A300",
    "TypeOfSteel_A400",
    "Steel_Plate_Thickness",
    "Edges_Index",
    "Empty_Index",
    "Square_Index",
    "Outside_X_Index",
    "Edges_X_Index",
    "Edges_Y_Index",
    "Outside_Global_Index",
    "LogOfAreas",
    "Log_X_Index",
    "Log_Y_Index",
    "Orientation_Index",
    "Luminosity_Index",
    "SigmoidOfAreas",
)

TARGET_NAMES: tuple[str, ...] = (
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
)


class SteelPlateFeatures(BaseModel):
    """Pydantic model representing all 27 geometric and photometric features for a steel plate."""

    model_config = ConfigDict(extra="forbid")

    X_Minimum: float = Field(..., description="Minimum x-coordinate of the fault bounding box")
    X_Maximum: float = Field(..., description="Maximum x-coordinate of the fault bounding box")
    Y_Minimum: float = Field(..., description="Minimum y-coordinate of the fault bounding box")
    Y_Maximum: float = Field(..., description="Maximum y-coordinate of the fault bounding box")
    Pixels_Areas: float = Field(..., ge=0.0, description="Area of the fault in pixels")
    X_Perimeter: float = Field(..., ge=0.0, description="Perimeter in x-direction")
    Y_Perimeter: float = Field(..., ge=0.0, description="Perimeter in y-direction")
    Sum_of_Luminosity: float = Field(..., description="Sum of pixel luminosity values")
    Minimum_of_Luminosity: float = Field(..., ge=0.0, le=255.0, description="Minimum luminosity")
    Maximum_of_Luminosity: float = Field(..., ge=0.0, le=255.0, description="Maximum luminosity")
    Length_of_Conveyer: float = Field(..., gt=0.0, description="Length of conveyer belt")
    TypeOfSteel_A300: int = Field(
        ..., ge=0, le=1, description="Binary indicator for steel type A300"
    )
    TypeOfSteel_A400: int = Field(
        ..., ge=0, le=1, description="Binary indicator for steel type A400"
    )
    Steel_Plate_Thickness: float = Field(..., gt=0.0, description="Thickness of plate in mm")
    Edges_Index: float = Field(..., description="Edges index")
    Empty_Index: float = Field(..., description="Empty area ratio")
    Square_Index: float = Field(..., description="Squareness index")
    Outside_X_Index: float = Field(..., description="Outside X index")
    Edges_X_Index: float = Field(..., description="Edges X index")
    Edges_Y_Index: float = Field(..., description="Edges Y index")
    Outside_Global_Index: float = Field(..., description="Outside global index")
    LogOfAreas: float = Field(..., description="Logarithm of fault area")
    Log_X_Index: float = Field(..., description="Logarithm of X index")
    Log_Y_Index: float = Field(..., description="Logarithm of Y index")
    Orientation_Index: float = Field(..., ge=-1.0, le=1.0, description="Orientation index [-1, 1]")
    Luminosity_Index: float = Field(..., ge=-1.0, le=1.0, description="Luminosity index [-1, 1]")
    SigmoidOfAreas: float = Field(..., ge=0.0, le=1.0, description="Sigmoid transform of area")

    @model_validator(mode="after")
    def validate_coordinates_and_luminosity(self) -> "SteelPlateFeatures":
        """Validate logical coordinate and range consistency."""
        if self.X_Minimum > self.X_Maximum:
            raise ValueError(
                f"X_Minimum ({self.X_Minimum}) cannot exceed X_Maximum ({self.X_Maximum})"
            )
        if self.Y_Minimum > self.Y_Maximum:
            raise ValueError(
                f"Y_Minimum ({self.Y_Minimum}) cannot exceed Y_Maximum ({self.Y_Maximum})"
            )
        if self.Minimum_of_Luminosity > self.Maximum_of_Luminosity:
            raise ValueError(
                f"Minimum_of_Luminosity ({self.Minimum_of_Luminosity}) cannot exceed Maximum_of_Luminosity ({self.Maximum_of_Luminosity})"
            )
        if self.TypeOfSteel_A300 == 1 and self.TypeOfSteel_A400 == 1:
            raise ValueError(
                "Plate cannot be both TypeOfSteel_A300 and TypeOfSteel_A400 simultaneously"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return features in ordered dictionary."""
        return {col: getattr(self, col) for col in FEATURE_NAMES}


class PredictionResult(BaseModel):
    """Prediction result for a single steel plate."""

    predicted_class: str = Field(..., description="Predicted defect class name")
    predicted_class_index: int = Field(..., description="Predicted defect class index (0-6)")
    probabilities: dict[str, float] = Field(..., description="Predicted class probabilities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Top-1 class predicted probability")
    review_recommended: bool = Field(..., description="Whether human review is recommended")
    abstention_threshold: float = Field(
        ..., description="Confidence threshold used for review flag"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    samples: list[SteelPlateFeatures] = Field(
        ..., min_length=1, description="List of plate records"
    )
    abstention_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Optional custom abstention confidence threshold"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response summary."""

    predictions: list[PredictionResult] = Field(..., description="List of individual predictions")
    total_samples: int = Field(..., ge=1, description="Total number of evaluated plates")
    review_count: int = Field(..., ge=0, description="Number of plates flagged for human review")
    review_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of plates flagged for review"
    )


class FeatureDriftReport(BaseModel):
    """Drift evaluation for an individual feature."""

    feature_name: str
    psi: float = Field(..., description="Population Stability Index")
    ks_statistic: float = Field(..., description="Kolmogorov-Smirnov statistic")
    ks_pvalue: float = Field(..., description="Kolmogorov-Smirnov p-value")
    is_drift: bool = Field(..., description="True if feature shows significant drift")
    severity: str = Field(..., description="None, Moderate, or Severe drift")


class PredictionDriftReport(BaseModel):
    """Drift evaluation for the predicted defect distribution."""

    psi: float = Field(..., description="PSI between baseline and production defect frequencies")
    reference_distribution: dict[str, float] = Field(..., description="Baseline class frequencies")
    production_distribution: dict[str, float] = Field(
        ..., description="Production class frequencies"
    )
    is_drift: bool = Field(..., description="True if prediction distribution has drifted")


class DriftCheckRequest(BaseModel):
    """Drift check request containing current production samples and optional custom reference."""

    production_samples: list[SteelPlateFeatures] = Field(
        ..., min_length=10, description="Production feature samples (min 10)"
    )
    reference_samples: list[SteelPlateFeatures] | None = Field(
        None, description="Optional custom reference baseline samples"
    )


class DriftCheckResponse(BaseModel):
    """Comprehensive drift analysis response."""

    features_evaluated: int
    features_drifted_count: int
    overall_drift_detected: bool
    feature_reports: list[FeatureDriftReport]
    prediction_drift: PredictionDriftReport | None = None
