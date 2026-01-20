# Credit Default Risk Modeling

A decision-theoretic approach to credit risk modeling, from temporal framing through deployment governance.

## What decision does this system make?

**Approve or reject credit card applications** based on estimated Probability of Default (PD).

The system outputs calibrated PD estimates, which are combined with business parameters (Exposure at Default, Loss Given Default, opportunity cost) to make approval decisions via a cost-based threshold rule:

```
Approve if: PD × EAD × LGD < opportunity_cost
```

This is fundamentally a **constrained optimization problem**: minimize expected financial loss while operating within regulatory constraints (fairness, explainability) and business constraints (approval rates, portfolio composition).

## Why is this hard in the real world?

**Asymmetric costs**: A false negative (approving a default) costs 10-100x more than a false positive (rejecting a creditworthy applicant). Standard classification metrics (accuracy, AUC) don't capture this.

**Label noise**: Default labels are noisy due to delayed defaults, censored outcomes (early payoff), and definitional ambiguity. This degrades PD calibration, which directly impacts Expected Loss calculations.

**Temporal leakage**: Credit models must predict future defaults using only historical information available at decision time. Random cross-validation creates unrealistic optimism by leaking future information.

**Regulatory constraints**: Models must be interpretable (coefficient inspection, SHAP values), fair (no systematic disparities across protected groups), and stable (consistent predictions across time periods and demographic groups).

**Data quality over model complexity**: Performance gains come primarily from addressing data issues (label noise, leakage, missingness) rather than increasing model complexity. A well-calibrated logistic regression often outperforms poorly-calibrated deep learning models.

## What tradeoffs were chosen?

**Model choice**: Logistic regression baseline → LightGBM (constrained). Not deep learning because:
- Tabular credit data doesn't require high-dimensional representations
- Interpretability requirements are strict (regulatory audits, customer explanations)
- Marginal performance gains don't justify added complexity and opacity

**Calibration over discrimination**: Well-calibrated PD estimates are essential for Expected Loss calculation. A model with perfect discrimination but poor calibration will produce biased EL estimates.

**Decision-level evaluation over classification metrics**: Focus on business outcomes (expected loss, approval rates) rather than accuracy, AUC, or other classification metrics.

**Fairness as constraint, not optimization target**: Fairness requirements are enforced as constraints (e.g., 80% rule for approval rate disparities), not optimized as objectives.

**Temporal splits over random CV**: Train/validation/test splits respect temporal ordering (proxy-based when true timestamps unavailable) to prevent leakage and expose distribution shift.

**Data-centric iteration**: Improvements come from data interventions (label refinement, feature exclusion to reduce dominance) rather than model architecture changes.

## When should this model NOT be used?

**Economic regime shifts**: Models trained on historical data may not generalize to fundamentally different economic conditions (recessions, policy changes). Requires retraining or threshold adjustment.

**Thin-file applicants**: Applicants with insufficient credit history may have unreliable PD estimates. Requires manual review or alternative scoring methods.

**Extreme PD values**: Predictions near 0 or 1 may be poorly calibrated. Requires manual review or confidence intervals.

**High-stakes applications**: Large loan amounts (high EAD) amplify Expected Loss. May require stricter thresholds or additional underwriting.

**Protected attribute proxies**: Models may encode bias through proxies (e.g., credit limit as socioeconomic proxy). Requires fairness constraints and group-level monitoring.

**Missing critical features**: Models assume complete feature sets. Missing income, employment status, or debt-to-income ratio may degrade performance.

## Repository Structure

```
credit-default-risk-modeling/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/              # Original dataset (gitignored if large)
└── notebooks/
    ├── 01_temporal_framing.ipynb
    ├── 02_label_construction_and_noise.ipynb
    ├── 03_data_audit_and_leakage.ipynb
    ├── 04_baseline_decision_model.ipynb
    ├── 05_thresholding_and_policy.ipynb
    ├── 06_data_centric_iteration.ipynb
    ├── 07_higher_capacity_model.ipynb
    ├── 08_fairness_and_stability.ipynb
    └── 09_deployment_and_drift_integration.ipynb
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

UCI Credit Card Default dataset (30,000 accounts). See `notebooks/01_temporal_framing.ipynb` for temporal structure and assumptions.

## Core Insight

**"What degrades performance more: model choice or data issues?"**

This project tests the hypothesis that addressing data quality (label noise, leakage, missingness) yields larger performance gains than switching from logistic regression to gradient boosting, or from gradient boosting to neural networks.

Results validate that data-centric interventions (label refinement, feature engineering) often provide larger gains than model complexity increases, while maintaining interpretability and regulatory compliance.
