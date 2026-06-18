import os
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             average_precision_score)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models") + os.sep

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=200):
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": model_name,
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics, y_pred, y_proba

def find_optimal_threshold(model, X_val, y_val):
    y_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1

def save_model(model, filename):
    joblib.dump(model, MODEL_PATH + filename)

def load_model(filename):
    return joblib.load(MODEL_PATH + filename)

def get_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df.sort_values("importance", ascending=False, inplace=True)
    return imp_df.head(top_n)
