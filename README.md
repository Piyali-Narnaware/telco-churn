# Customer Churn & Retention Intelligence

Predictive churn analysis and retention strategy for a subscription-based telecom company.

## Overview

An analysis of customer churn using the IBM sample telecom dataset. The repository contains data preparation, segmentation, classification models and exploratory retention scenarios. The retention ideas are proposals, not interventions tested with customers.

## Dataset

**Telco Customer Churn** (IBM Sample — 7,043 customers, 21 features)

Includes customer demographics, account information, service subscriptions, and churn status.

## Project Structure

```
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned data, predictions, Power BI export
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_segmentation.ipynb   # Customer segmentation (K-Means)
│   ├── 03_churn_analysis.ipynb # Churn driver identification
│   ├── 04_risk_model.ipynb     # Predictive churn modeling (Random Forest)
│   └── 05_business_report.ipynb # Business recommendations
├── src/                        # Reusable Python modules
│   ├── preprocessing.py        # Data cleaning & feature engineering
│   ├── features.py             # Segmentation, risk scoring
│   ├── segmentation.py         # K-Means clustering
│   └── modeling.py             # Model training & evaluation
├── dashboard/                  # Power BI files
├── models/                     # Trained model artifacts
├── reports/                    # Generated charts and outputs
└── scripts/                    # Pipeline execution scripts
```

## Key Findings

| Finding | Impact |
|---------|--------|
| Month-to-month contracts are 3x more likely to churn | 45.8% of model importance |
| Churn rate drops 60% after 12 months tenure | Intervene early |
| Customers without online security churn 2x more | Bundle opportunity |
| Electronic check users churn at 45% vs 20% auto-pay | Payment migration |
| 312 high-value, high-risk customers identified | £48K/mo revenue at risk |

## Model Performance

| Model | ROC AUC | Avg Precision |
|-------|---------|---------------|
| **Logistic Regression** | **0.83** | **0.62** |
| Random Forest | 0.82 | 0.59 |

## Business Recommendations

1. **Early Tenure Intervention** — Retention offers within first 90 days
2. **Service Bundles** — Package security + support at discount
3. **Auto-Pay Migration** — £5/mo discount for electronic check users
4. **VIP Retention** — Dedicated program for high-value at-risk customers
5. **Fiber Optic Retention** — Competitive offers for fiber subscribers

The retention proposals have not been tested, so no realised or validated annual savings are claimed.

## Reproducing the analysis

The source CSV is not stored in this repository. Obtain the IBM Telco Customer Churn sample dataset and save it as `data/raw/Telco-Customer-Churn.csv`, then run `python scripts/run_pipeline.py` from the repository root after installing the dependencies in `environment.yml`. The pipeline preserves source customer IDs in the BI export. The rule-based `RiskScore` is a heuristic, not a calibrated model probability; segment labels and retention-value estimates are exploratory. Model results in this README should be rechecked against a fresh run.

## Technologies

- Python (pandas, scikit-learn, SHAP)
- Jupyter Notebook
- Power BI
- Conda
