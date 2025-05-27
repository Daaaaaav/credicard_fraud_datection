import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_base_paths(dataset_filename):
    base_name = os.path.splitext(os.path.basename(dataset_filename))[0]
    return {
        "processed": f"data/processed/{base_name}_processed.npz",
        "scaler": f"models/{base_name}_scaler.pkl",
        "features": f"models/{base_name}_features.pkl"
    }

def get_current_dataset():
    try:
        with open("current_dataset.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def preprocess_data(dataset_filename=None):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    if dataset_filename is None or not os.path.exists(dataset_filename):
        raise ValueError("Dataset not found")

    # Save current dataset path persistently
    with open("current_dataset.txt", "w") as f:
        f.write(dataset_filename)

    # Force delete stale processed dataset
    processed_path = "data/processed/processed_dataset.npz"
    if os.path.exists(processed_path):
        os.remove(processed_path)

    df = pd.read_csv(dataset_filename)
    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    X = df.drop(columns=['Class'])
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/processed_dataset.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    return {
        "message": "Data preprocessed and saved.",
        "sample": df.head(1).to_dict(orient="records")
    }
