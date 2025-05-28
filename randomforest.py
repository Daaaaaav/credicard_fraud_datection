import os
import numpy as np
import pandas as pd
import joblib
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from preprocessing import get_current_dataset, get_base_paths
from autoencoder_backend import load_dataset, load_models, compute_reconstruction_error, augment_with_error

def train_and_save_model(dataset_filename=None, model_path="models/random_forest.pkl"):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    if not os.path.exists(dataset_filename):
        raise FileNotFoundError(f"Dataset not found: {dataset_filename}")

    paths = get_base_paths(dataset_filename)
    df = pd.read_csv(dataset_filename)

    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    X = df.drop(columns=['Class'], errors='ignore')
    X = X.drop(columns=['id'], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['Class']
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(paths["scaler"]), exist_ok=True)
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(feature_names, paths["features"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    os.makedirs(os.path.dirname(paths["processed"]), exist_ok=True)
    np.savez(paths["processed"], X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    best_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    best_model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "features": feature_names
    }, model_path)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    return {
    "message": f"Model saved to {model_path}",
    "stats": {
        'model': 'Random Forest',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
}

def train_combined_model():
    dataset_path = get_current_dataset()
    if dataset_path is None or not os.path.exists(dataset_path):
        raise ValueError("Dataset not found. Please upload and preprocess a dataset first.")

    X, y, _ = load_dataset(dataset_path)
    encoder, autoencoder, _, scaler = load_models()
    X_scaled = scaler.transform(X)

    bottleneck = encoder.predict(X_scaled, verbose=0)
    reconstructed = autoencoder.predict(X_scaled, verbose=0)
    error = compute_reconstruction_error(X_scaled, reconstructed).reshape(-1, 1)
    auto_features = augment_with_error(bottleneck, error)

    iso_data = np.load("data/processed/iso_scores.npz")

    if iso_data['train'].shape[0] == auto_features.shape[0]:
        iso_scores = iso_data['train']
    elif iso_data['test'].shape[0] == auto_features.shape[0]:
        iso_scores = iso_data['test']
    elif (iso_data['train'].shape[0] + iso_data['test'].shape[0]) == auto_features.shape[0]:
        iso_scores = np.vstack([iso_data['train'], iso_data['test']])
    else:
        raise ValueError(
            f"Mismatch: auto_features has {auto_features.shape[0]} rows, "
            f"but iso_scores has train={iso_data['train'].shape[0]}, test={iso_data['test'].shape[0]}, total={iso_data['train'].shape[0] + iso_data['test'].shape[0]}"
        )

    combined_features = np.hstack((auto_features, iso_scores))

    X_train, X_test, y_train, y_test = train_test_split(
        combined_features, y, test_size=0.3, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_res, y_res)

    joblib.dump(clf, "models/combined_rf.pkl")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report.get("1", {}).get("f1-score", 0.0)

    metrics = {
        "accuracy": report.get("accuracy", 0.0),
        "precision": report.get("1", {}).get("precision", 0.0),
        "recall": report.get("1", {}).get("recall", 0.0),
        "f1_score": f1
    }

    return {"message": "Combined model trained successfully", "metrics": metrics}

def load_and_predict_bulk(dataset_filename=None, model_path="models/random_forest.pkl"):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    if not os.path.exists(dataset_filename):
        return {'error': f'Dataset file not found: {dataset_filename}'}

    model_bundle = joblib.load(model_path)

    if isinstance(model_bundle, dict):
        model = model_bundle["model"]
        scaler = model_bundle["scaler"]
        features = model_bundle["features"]
    else:
        raise ValueError("Model file format invalid. Retrain Random Forest using /train/randomforest.")


    df = pd.read_csv(dataset_filename)
    if not all(f in df.columns for f in features):
        return {'error': f"Input data missing required features: {features}"}

    X = df[features].select_dtypes(include=[np.number])
    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    df["predicted"] = y_pred
    df["is_fraud"] = y_pred == 1

    stats = {
        "model": "Random Forest",
        "fraud_count": int(df["is_fraud"].sum()),
    }

    if "Class" in df.columns:
        y_true = df["Class"]
        stats.update({
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        })
    else:
        stats["warning"] = "No 'Class' column found. Cannot compute evaluation metrics."

    return {
        "predictions": df[df["is_fraud"]].head(100).to_dict(orient="records"),
        "stats": stats
    }

def evaluate_combined_model():
    from autoencoder_backend import load_models, compute_reconstruction_error, augment_with_error
    dataset_path = get_current_dataset()
    if dataset_path is None or not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found.")

    df = pd.read_csv(dataset_path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' column for evaluation.")

    X = df.drop(columns=["Class"])
    y = df["Class"]
    encoder, autoencoder, _, scaler = load_models()

    X_scaled = scaler.transform(X)
    bottleneck = encoder.predict(X_scaled, verbose=0)
    reconstructed = autoencoder.predict(X_scaled, verbose=0)
    error = compute_reconstruction_error(X_scaled, reconstructed).reshape(-1, 1)
    auto_features = augment_with_error(bottleneck, error)

    iso_data = np.load("data/processed/iso_scores.npz")
    total_iso = iso_data['train'].shape[0] + iso_data['test'].shape[0]
    if total_iso == auto_features.shape[0]:
        iso_scores = np.vstack([iso_data['train'], iso_data['test']])
    elif iso_data['train'].shape[0] == auto_features.shape[0]:
        iso_scores = iso_data['train']
    elif iso_data['test'].shape[0] == auto_features.shape[0]:
        iso_scores = iso_data['test']
    else:
        raise ValueError("Mismatch between auto features and ISO scores")

    combined_features = np.hstack((auto_features, iso_scores))
    clf = joblib.load("models/combined_rf.pkl")

    y_pred = clf.predict(combined_features)
    y_proba = clf.predict_proba(combined_features)[:, 1]

    df["predicted"] = y_pred
    df["is_fraud"] = y_pred == 1

    stats = {
        "model": "Combined",
        "fraud_count": int(df["is_fraud"].sum()),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist()
    }

    fraud_df = df[df["is_fraud"]]
    fraud_count = len(fraud_df)
    total_samples = len(df)

    return {
        "predictions": fraud_df.head(100).to_dict(orient="records"),
        "stats": stats,
        "message": f"Combined Model evaluated {total_samples} samples. Detected {fraud_count} fraudulent cases."
    }