import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")
from src.preprocessing import load_data
from src.features import segment_customers, retention_value, add_risk_score

df = load_data()
df = segment_customers(df)
df = retention_value(df)
df = add_risk_score(df)

preds = pd.read_csv("data/processed/predictions.csv", index_col=0) if pd.io.common.file_exists("data/processed/predictions.csv") else None

if preds is not None and len(preds) == len(df):
    df["ChurnProbability"] = preds["Churn_Probability"].values
    df["RiskBand"] = preds["Risk_Band"].values
else:
    df["ChurnProbability"] = df["RiskScore"]
    df["RiskBand"] = df["RiskBand"]

df["CustomerID"] = df.index
export = df[[
    "CustomerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "InternetService", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
    "Segment", "RiskScore", "RiskBand", "ChurnProbability",
    "RetentionValue", "RetentionPriority", "ServiceCount",
    "HasPartner", "HasDependents", "IsSenior",
    "IsMonthToMonth", "IsElectronicCheck", "IsPaperlessBilling",
    "AvgMonthlySpend", "TenureGroup", "Churn"
]]
export.to_csv("data/processed/powerbi_export.csv", index=False)
print(f"Power BI export saved: data/processed/powerbi_export.csv ({len(export)} rows, {len(export.columns)} cols)")
print(f"Columns: {list(export.columns)}")
