import sys; sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.preprocessing import full_pipeline, load_data
from src.modeling import (train_logistic_regression, train_random_forest,
                          evaluate_model, save_model, get_feature_importance)
from src.features import segment_customers, retention_value, add_risk_score

print("=== Running Full Pipeline ===")

df, df_encoded, X_train, X_test, y_train, y_test, feature_cols = full_pipeline()
print(f"Processed: {len(df)} customers, {len(feature_cols)} features")

print("Training Logistic Regression...")
lr = train_logistic_regression(X_train, y_train)
lr_metrics, _, _ = evaluate_model(lr, X_test, y_test, "Logistic Regression")
print(f"  ROC AUC: {lr_metrics['roc_auc']}, Avg Precision: {lr_metrics['avg_precision']}")

print("Training Random Forest...")
rf = train_random_forest(X_train, y_train)
rf_metrics, _, _ = evaluate_model(rf, X_test, y_test, "Random Forest")
print(f"  ROC AUC: {rf_metrics['roc_auc']}, Avg Precision: {rf_metrics['avg_precision']}")

best_model = lr
save_model(best_model, "lr_churn_model.pkl")
print("Best model (Logistic Regression) saved to models/lr_churn_model.pkl")

y_proba = best_model.predict_proba(X_test)[:, 1]

preds = pd.DataFrame({
    "Churn_Probability": y_proba,
    "Predicted_Churn": (y_proba >= 0.5).astype(int),
    "Actual_Churn": y_test.values,
})
preds["Risk_Band"] = pd.cut(
    preds["Churn_Probability"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["Very Low", "Low", "Medium", "High", "Very High"],
)
preds.to_csv("data/processed/predictions.csv", index=False)
print(f"Predictions saved. ROC AUC: {rf_metrics['roc_auc']}")
print(preds["Risk_Band"].value_counts())

imp = get_feature_importance(best_model, feature_cols, top_n=10)
print("\nTop 10 Features:")
print(imp.to_string(index=False))

print("\n=== Power BI Export ===")
from src.preprocessing import (create_demographic_features, create_service_features,
                                create_contract_features, create_payment_features,
                                create_spend_features)
df_raw = load_data()
df_raw = create_demographic_features(df_raw)
df_raw = create_service_features(df_raw)
df_raw = create_contract_features(df_raw)
df_raw = create_payment_features(df_raw)
df_raw = create_spend_features(df_raw)
df_raw = segment_customers(df_raw)
df_raw = retention_value(df_raw)
df_raw = add_risk_score(df_raw)
df_raw = df_raw.reset_index(drop=False)
df_raw["CustomerID"] = df_raw.index.astype(str)

export = df_raw[[
    "CustomerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "InternetService", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
    "Segment", "RiskScore", "RiskBand", "RetentionValue", "RetentionPriority",
    "ServiceCount", "HasPartner", "HasDependents",
    "IsMonthToMonth", "IsElectronicCheck", "IsPaperlessBilling",
    "AvgMonthlySpend", "TenureGroup", "Churn",
    "HasOnlineSecurity", "HasTechSupport", "HasOnlineBackup",
]]
export.to_csv("data/processed/powerbi_export.csv", index=False)
print(f"Exported {len(export)} rows to data/processed/powerbi_export.csv")
print("Done")
