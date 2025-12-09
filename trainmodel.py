import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model_utils2 import compute_rule_columns


def compute_fee_weight(months_unpaid):
    if months_unpaid >= 3:
        return 1.0
    elif months_unpaid == 2:
        return 0.7
    elif months_unpaid == 1:
        return 0.4
    else:
        return 0.0


def load_merge(att_path, cgpa_path, fees_path):
    att = pd.read_csv(att_path)
    cgpa = pd.read_csv(cgpa_path)     
    fees = pd.read_csv(fees_path)

    
    fees["fee_weight"] = fees["outstanding_months"].apply(compute_fee_weight)

    
    df = att.merge(cgpa, on="student_id").merge(fees, on="student_id")

    
    if "avg_cgpa" not in df.columns:
        df["avg_cgpa"] = df[["cgpa_sem1", "cgpa_sem2", "cgpa_sem3"]].mean(axis=1)

    return df


def create_labels(df):
    df = compute_rule_columns(df)
    df["dropout_label"] = ((df["rule_score"] >= 0.6) | (df["rule_flag"] == 1)).astype(int)
    return df


def train_and_save(df, model_path="dropout_model_xgb.pkl"):
    features = ["attendance", "avg_cgpa", "fee_weight", "rule_score"]
    X = df[features].fillna(0)
    y = df["dropout_label"]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

   
    pos = sum(y_train == 1)
    neg = sum(y_train == 0)
    scale_pos_weight = neg / pos if pos > 0 else 1

    
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )

   
    calib = CalibratedClassifierCV(xgb, cv=5, method="sigmoid")
    calib.fit(X_train, y_train)

   
    preds = calib.predict(X_test)
    print(classification_report(y_test, preds))

   
    joblib.dump(calib, model_path)
    print("Saved model to", model_path)


if __name__ == "__main__":
    df = load_merge("attendance.csv", "cgpa.csv", "fees.csv")
    df = create_labels(df)
    train_and_save(df)
