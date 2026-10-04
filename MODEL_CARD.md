# Model Card: Steel Plate Fault Detection & Quality Monitoring

## Model Details
- **Model Name:** Steel Plate Multi-Defect Random Forest Classifier (Calibrated)
- **Version:** 0.1.0
- **Model Type:** Balanced Ensemble Random Forest Classifier (`RandomForestClassifier(n_estimators=300, class_weight='balanced')`) with Post-Hoc Sigmoid Probability Calibration (`CalibratedClassifierCV`)
- **Framework:** `scikit-learn>=1.3.0` (pure CPU-friendly scikit-learn pipeline)
- **License:** MIT License
- **Contact:** ML Engineering Team

---

## Intended Use
### Primary Intended Uses
- **Industrial Defect Classification:** Automated sorting and defect categorization of industrial steel plates into 7 standardized defect classes based on 27 geometric, photometric, and metallurgical features.
- **Selective Classification & Quality Gate:** Flagging low-confidence predictions (confidence $< 0.50$ by default) for metallurgical inspection while allowing confident classifications to proceed automatically.
- **Production Drift Monitoring:** Continuous monitoring of rolling-line feature distributions (via Population Stability Index and Kolmogorov-Smirnov test) and defect category proportions.

### Out-of-Scope & Unintended Uses
- Autonomous safety-critical shutdown of manufacturing lines without human-in-the-loop review.
- Application to non-ferrous alloys (e.g., aluminum, copper sheets) or hot-strip rolling mills without domain retraining and recalibration.
- Real-time video frame object localization (this system classifies tabular bounding-box summaries extracted by upstream optical inspection).

---

## Training Data & Data Splits
- **Source:** Official UCI Machine Learning Repository — Steel Plates Faults dataset (Semeion Research Center).
- **Total Records:** 1,941 samples, 27 continuous/binary input features, 7 mutually exclusive defect labels.
- **Partitioning Strategy (Stratified Split with Fixed Seed `42`):**
  - **Train Set:** 1,358 samples (70%)
  - **Validation Set:** 291 samples (15%) — utilized strictly for model selection, probability calibration fitting, and permutation feature importance.
  - **Test Set:** 292 samples (15%) — held untouched and evaluated exactly once for final reporting.
- **Class Imbalance Mitigation:**
  - Strict absence of pre-split synthetic oversampling (no SMOTE before or after splitting).
  - Inverse class frequency balanced sample weighting (`class_weight='balanced'`) natively integrated into tree learning and logistic baseline solvers.

---

## Performance Evaluation & Measured Results

All metrics below are computed on the **untouched final test set (292 samples)**:

### 1. Comparative Model Summary
| Metric | Baseline (Logistic Regression) | Random Forest (Uncalibrated) | Random Forest (Calibrated - Deployed) |
| :--- | :---: | :---: | :---: |
| **Macro-F1 (Primary)** | **0.6455** | **0.8155** | **0.7903** |
| **Weighted-F1** | 0.6492 | 0.7859 | 0.7807 |
| **Balanced Accuracy** | 0.7239 | 0.8148 | 0.7543 |
| **Accuracy** | 0.6507 | 0.7808 | 0.7774 |
| **Log Loss** | 0.9892 | 0.6212 | **0.6026** *(best calibration)* |
| **Multiclass Brier Score** | 0.4609 | 0.3236 | **0.3044** *(best reliability)* |
| **Expected Calibration Error (ECE)** | 0.0611 | 0.1074 | **0.0689** *(reliable probabilities)* |

*Observation:* While uncalibrated Random Forest achieves a slightly higher discrete macro-F1 (0.8155), the Sigmoid-calibrated model substantially improves probabilistic quality — reducing log loss from 0.6212 to 0.6026, multiclass Brier score from 0.3236 to 0.3044, and Expected Calibration Error (ECE) from 0.1074 to 0.0689, which is crucial for abstention thresholds.

### 2. Per-Class Test Breakdown (Calibrated Random Forest)
| Class Label | Support (N=292) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **K_Scatch** | 59 | 0.9815 | 0.8983 | **0.9381** |
| **Stains** | 11 | 1.0000 | 0.9091 | **0.9524** |
| **Z_Scratch** | 29 | 0.9600 | 0.8276 | **0.8889** |
| **Dirtiness** | 8 | 1.0000 | 0.6250 | **0.7692** |
| **Other_Faults** | 101 | 0.6777 | 0.8119 | **0.7387** |
| **Bumps** | 60 | 0.7692 | 0.6667 | **0.7143** |
| **Pastry** | 24 | 0.5200 | 0.5417 | **0.5306** |

---

## Selective Classification & Abstention Policy
To balance automation with risk mitigation, the pipeline implements an abstention rule: if the top predicted probability $\max_k p_k < \theta$, the sample is flagged for metallurgical review (`review_recommended=True`).

### Coverage vs Selective Accuracy on Untouched Test Set
| Confidence Threshold ($\theta$) | Coverage (%) | Automated Samples | Flagged for Review | Selective Accuracy (%) | Selective Macro-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00** *(No abstention)* | 100.0% | 292 | 0 | 77.74% | 0.7903 |
| **0.30** | 100.0% | 292 | 0 | 77.74% | 0.7903 |
| **0.40** | 98.29% | 287 | 5 | 79.09% | 0.8173 |
| **0.50** *(Default operational gate)* | **90.41%** | **264** | **28** | **82.20%** | **0.8409** |
| **0.60** | 78.77% | 230 | 62 | 85.65% | 0.8643 |
| **0.70** | 64.73% | 189 | 103 | 89.42% | 0.8844 |
| **0.80** | 48.63% | 142 | 150 | 95.77% | 0.9149 |
| **0.90** | 7.19% | 21 | 271 | **100.00%** | **1.0000** |

*Takeaway:* At the default threshold ($\theta = 0.50$), 90.41% of plates are classified autonomously with **82.20% selective accuracy** and **0.8409 selective Macro-F1** (an improvement over the 77.74% baseline accuracy and 0.7903 Macro-F1 at zero abstention). The 9.59% ambiguous plates (28 cases) are safely diverted to manual review. For plants prioritizing higher selective precision, $\theta = 0.60$ achieves **85.65% selective accuracy** and **0.8643 selective Macro-F1** at 78.77% coverage.

---

## Explainability
- **Permutation Importance (Validation Split, scoring=`f1_macro`):**
  1. `Length_of_Conveyer`: $+0.0716$ drop in Macro-F1 when shuffled.
  2. `Pixels_Areas`: $+0.0463$ drop.
  3. `Steel_Plate_Thickness`: $+0.0451$ drop.
  4. `Edges_Index`: $+0.0425$ drop.
  5. `LogOfAreas`: $+0.0389$ drop.
- **Model-Native Gini Importance (Tree MDI):** Top contributors align with fault dimensions and conveyer settings.
- **Methodological Note:** We do not claim external SHAP explanations or make surrogate approximations; scikit-learn's official permutation importance and tree-native MDI provide mathematically verifiable importance directly on validation observations.

---

## Quantitative Latency Benchmark (Intel/AMD x86_64 CPU)
- **Single-Request Inferences ($N=500$):**
  - **Median (p50):** 39.4 ms
  - **95th Percentile (p95):** 54.0 ms
  - **99th Percentile (p99):** 87.9 ms
  - **Throughput:** 22.4 single requests/sec
- **Batch Inferences ($N=100$ items per request):**
  - **Mean Batch Latency:** 53.0 ms
  - **Per-Sample Latency:** 0.53 ms
  - **Throughput:** 1,885.3 samples/sec

---

## Limitations & Operational Boundaries
1. **Severe Imbalance on Rare Defect Classes:** `Dirtiness` (55 total) and `Stains` (72 total) represent $<4\%$ of data. While precision on test is high due to distinct photometric profiles, confidence estimates on small numbers of samples warrant human oversight.
2. **Ambiguity in "Other_Faults":** The largest category (`Other_Faults`, $34.7\%$ of dataset) is a heterogeneous catch-all class created by plant operators. It exhibits the lowest precision ($67.8\%$) and frequently overlaps with `Pastry` and `Bumps`.
3. **Sensor Drift:** Ambient lighting fluctuations or sensor dirtiness can shift luminosity indices. The built-in drift detector monitors both PSI and KS-test to alert operators before degradation occurs.
