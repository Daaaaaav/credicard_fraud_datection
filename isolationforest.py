import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from preprocessing import get_current_dataset, get_base_paths

def train_isolation_forest(dataset_filename=None, model_path='models/isolation_forest_model.pkl', threshold_path='models/threshold.json'):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    paths = get_base_paths(dataset_filename)

    data = np.load(paths['processed'])
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    model = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
    model.fit(X_train)
    joblib.dump(model, model_path)

    scores_train = -model.decision_function(X_train).reshape(-1, 1)
    scores_test = -model.decision_function(X_test).reshape(-1, 1)

    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/iso_scores.npz", train=scores_train, test=scores_test)

    median_thresh = np.median(scores_test)
    preds = (scores_test > median_thresh).astype(int)

    return {
        'model': 'Isolation Forest',
        'MSE': float(mean_squared_error(y_test, preds)),
        'MAE': float(mean_absolute_error(y_test, preds)),
        'CrossEntropyLoss': float(log_loss(y_test, preds)),
        'message': f'Model and anomaly features saved.'
    }
