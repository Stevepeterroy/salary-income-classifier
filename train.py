"""Train an income-band classifier from a local Adult-schema CSV."""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["age", "hours-per-week"]
CATEGORICAL = ["education", "occupation"]
FEATURES = NUMERIC + CATEGORICAL

def train(csv_path, output_path):
    data = pd.read_csv(csv_path, na_values=["?", " ?"], skipinitialspace=True)
    missing = set(FEATURES + ["income"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = data.dropna(subset=["income"]).copy()
    labels = data["income"].astype(str).str.strip().str.rstrip(".")
    if not set(labels).issubset({"<=50K", ">50K"}):
        raise ValueError("income must contain <=50K or >50K labels")
    features = data[FEATURES].copy()
    for name in NUMERIC:
        features[name] = pd.to_numeric(features[name], errors="raise")
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    metrics = {"majority_baseline_accuracy": accuracy_score(y_test, baseline.predict(x_test)),"holdout_rows": len(y_test), "accuracy": accuracy_score(y_test, prediction),
               "classification_report": classification_report(y_test, prediction, output_dict=True)}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"holdout_rows": len(y_test), "accuracy": metrics["accuracy"]}))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Local CSV with Adult-schema features and income labels")
    parser.add_argument("--output", default="data/income-model.joblib")
    args = parser.parse_args()
    train(args.csv, args.output)
