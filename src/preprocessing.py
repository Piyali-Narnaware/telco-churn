import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Telco-Customer-Churn.csv")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed") + os.sep

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    return df

def create_demographic_features(df):
    df["HasPartner"] = (df["Partner"] == "Yes").astype(int)
    df["HasDependents"] = (df["Dependents"] == "Yes").astype(int)
    df["IsSenior"] = df["SeniorCitizen"].astype(int)
    return df

def create_service_features(df):
    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies"]
    for col in service_cols:
        df[f"Has{col}"] = df[col].isin(["Yes"]).astype(int)
    df["ServiceCount"] = sum(df[f"Has{col}"] for col in service_cols)
    return df

def create_contract_features(df):
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["IsOneYear"] = (df["Contract"] == "One year").astype(int)
    df["IsTwoYear"] = (df["Contract"] == "Two year").astype(int)
    return df

def create_payment_features(df):
    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["IsPaperlessBilling"] = (df["PaperlessBilling"] == "Yes").astype(int)
    return df

def create_spend_features(df):
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["TenureGroup"] = pd.cut(df["tenure"],
                                bins=[0, 6, 12, 24, 48, 72],
                                labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4-6yr"])
    return df

def encode_target(df):
    le = LabelEncoder()
    df["ChurnTarget"] = le.fit_transform(df["Churn"])
    return df, le

DROP_COLS = ["TenureGroup", "ChargeBin"]

def prepare_features(df):
    cat_cols = ["gender", "Partner", "Dependents", "PhoneService",
                "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod"]
    df_clean = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    for col in df_encoded.select_dtypes(include=["category"]).columns:
        df_encoded[col] = df_encoded[col].astype(int)
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        if col not in ["Churn", "ChurnTarget"]:
            df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train_scaled, X_test_scaled, scaler

def full_pipeline():
    df = load_data()
    df = create_demographic_features(df)
    df = create_service_features(df)
    df = create_contract_features(df)
    df = create_payment_features(df)
    df = create_spend_features(df)
    df_encoded = prepare_features(df)
    df_encoded, le = encode_target(df_encoded)
    feature_cols = [c for c in df_encoded.columns if c not in ["Churn", "ChurnTarget"]]
    X = df_encoded[feature_cols]
    y = df_encoded["ChurnTarget"]
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    df.to_csv(PROCESSED_PATH + "telco_processed.csv", index=False)
    df_encoded.to_csv(PROCESSED_PATH + "telco_encoded.csv", index=False)
    X_train_s.to_csv(PROCESSED_PATH + "X_train.csv", index=False)
    X_test_s.to_csv(PROCESSED_PATH + "X_test.csv", index=False)
    pd.Series(y_train, name="ChurnTarget").to_csv(PROCESSED_PATH + "y_train.csv", index=False)
    pd.Series(y_test, name="ChurnTarget").to_csv(PROCESSED_PATH + "y_test.csv", index=False)
    return df, df_encoded, X_train_s, X_test_s, y_train, y_test, feature_cols

if __name__ == "__main__":
    full_pipeline()
