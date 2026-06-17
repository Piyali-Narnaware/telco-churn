import pandas as pd
import numpy as np

RISK_BANDS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
RISK_LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]

RISK_SCORE_MULTIPLIERS = {
    "tenure <= 12": 1.5,
    "tenure <= 6": 2.0,
    "Month-to-month": 2.5,
    "Electronic check": 1.4,
    "No online security": 1.3,
    "No tech support": 1.3,
    "Fiber optic": 1.2,
}

def add_risk_score(df):
    base = 0.5
    df["RiskScore"] = base
    df.loc[df["tenure"] <= 12, "RiskScore"] *= RISK_SCORE_MULTIPLIERS["tenure <= 12"]
    df.loc[df["tenure"] <= 6, "RiskScore"] *= RISK_SCORE_MULTIPLIERS["tenure <= 6"]
    if "IsMonthToMonth" in df.columns:
        df.loc[df["IsMonthToMonth"] == 1, "RiskScore"] *= RISK_SCORE_MULTIPLIERS["Month-to-month"]
    if "IsElectronicCheck" in df.columns:
        df.loc[df["IsElectronicCheck"] == 1, "RiskScore"] *= RISK_SCORE_MULTIPLIERS["Electronic check"]
    if "HasOnlineSecurity" in df.columns:
        df.loc[df["HasOnlineSecurity"] == 0, "RiskScore"] *= RISK_SCORE_MULTIPLIERS["No online security"]
    if "HasTechSupport" in df.columns:
        df.loc[df["HasTechSupport"] == 0, "RiskScore"] *= RISK_SCORE_MULTIPLIERS["No tech support"]
    df["RiskScore"] = df["RiskScore"].clip(0, 1)
    df["RiskBand"] = pd.cut(df["RiskScore"], bins=RISK_BANDS,
                            labels=RISK_LABELS, include_lowest=True)
    return df

def segment_customers(df):
    conditions = [
        (df["tenure"] >= 24) & (df["MonthlyCharges"] >= df["MonthlyCharges"].median()) & (df["Churn"] == "No"),
        (df["tenure"] >= 24) & (df["MonthlyCharges"] >= df["MonthlyCharges"].median()) & (df["Churn"] == "Yes"),
        (df["tenure"] >= 24) & (df["MonthlyCharges"] < df["MonthlyCharges"].median()),
        (df["tenure"] < 24) & (df["tenure"] >= 6),
        (df["tenure"] < 6),
    ]
    labels = [
        "High-Value Loyal",
        "High-Value At-Risk",
        "Low-Value Loyal",
        "Developing",
        "New Customer",
    ]
    df["Segment"] = np.select(conditions, labels, default="Other")
    return df

def retention_value(df):
    df["RetentionValue"] = df["MonthlyCharges"] * 12
    df["RetentionPriority"] = pd.qcut(
        df["RetentionValue"].rank(method="first"),
        q=4, labels=["Low", "Medium", "High", "Very High"]
    )
    return df
