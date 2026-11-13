# Telecom Customer Churn | Retention Analysis

**Aim:** Which observable customer characteristics could help a telecom provider prioritise retention research?

I used IBM's public Telco Customer Churn sample to explore which account and service characteristics are associated with churn. It combines data preparation, segmentation and classification with a proposed retention workflow. I haven't tested any retention offers or measured recovered revenue.

## Analysis approach

1. Prepare customer, account, contract and service fields
2. Examine churn patterns by tenure, contract type, payment method and services
3. Compare predictive models and review their errors
4. Translate observed patterns into hypotheses for retention testing

The code is organised into [reusable modules](src/) and a [pipeline script](scripts/run_pipeline.py).

## Business questions

| Question | Possible follow-up |
|---|---|
| Does churn vary across contract types and tenure? | Investigate onboarding and renewal experiences |
| Do payment methods coincide with different churn rates? | Test whether payment friction contributes |
| Which customers might warrant outreach? | Compare model scores, contact cost and customer value |
| Which interventions work? | Run a controlled test before claiming retention uplift |

These comparisons are associations, not evidence of cause and effect. An apparent difference between groups may reflect tenure, customer mix or other factors.

## Model and validation status

The repository includes classification code and exploratory risk scoring. The latest automated check verifies Python syntax, **not** end-to-end model performance. Previously reported ROC AUC, average precision and financial exposure figures have therefore been omitted from this summary until reproduced from a documented run.

The rule-based `RiskScore` is a heuristic, **not** a calibrated churn probability. Historical segmentation that uses actual churn status must not be used as if it were a prospective prediction.

## Reproduce

The raw CSV is not included. Obtain the IBM Telco Customer Churn sample and place it at:

```text
data/raw/Telco-Customer-Churn.csv
```

Install the dependencies specified in [`environment.yml`](environment.yml), then run from the repository root:

```bash
python scripts/run_pipeline.py
```

Check the outputs in `data/processed/` and the saved model in `models/`. The pipeline preserves customer IDs in the prediction export for review.

## Limits and next step

This is a public sample-data exercise, not professional telecom retention experience. Before using the model operationally, it would need a reproducible holdout evaluation, leakage checks, probability calibration, a prospective customer-scoring design and a measured intervention test.
