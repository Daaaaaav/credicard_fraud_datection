
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError  

from preprocessing import preprocess_data, get_current_dataset

MODEL_PATH = "models/autoencoder.h5"
RF_PATH = "models/random_forest.pkl"
BOTTLENECK_MODEL_PATH = "models/encoder.h5"
SCALER_PATH = "models/scaler.pkl"

def load_dataset(dataset_filename=None):
    if dataset_filename is None:
        dataset_filename = get_current_dataset()
    df = pd.read_csv(dataset_filename)
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    return X, y, df

def preprocess_autoencoder_data(dataset_filename=None):
    X, y, _ = load_dataset(dataset_filename)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    return X_train_scaled, X_test_scaled, y_train, y_test

def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(input_layer)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    bottleneck = layers.Dense(16, activation="relu", name="bottleneck")(x)
    x = layers.Dense(32, activation="relu")(bottleneck)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    output_layer = layers.Dense(input_dim, activation="linear")(x)
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    encoder = tf.keras.Model(inputs=input_layer, outputs=bottleneck)
    
    autoencoder.compile(optimizer=Adam(1e-4), loss=MeanSquaredError())
    return autoencoder, encoder

def compute_reconstruction_error(original, reconstructed):
    return np.mean(np.square(original - reconstructed), axis=1)

def augment_with_error(encoded, error_vector):
    return np.hstack((encoded, error_vector.reshape(-1, 1)))

def load_models():
    encoder = tf.keras.models.load_model(BOTTLENECK_MODEL_PATH, compile=False)
    autoencoder = tf.keras.models.load_model(MODEL_PATH, compile=False)
    rf_clf = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    return encoder, autoencoder, rf_clf, scaler

def train_autoencoder():
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_autoencoder_data()

    X_train_normal = X_train_scaled[y_train == 0]
    autoencoder, encoder = build_autoencoder(X_train_scaled.shape[1])
    autoencoder.fit(X_train_normal, X_train_normal, epochs=50, batch_size=256, verbose=1)

    encoder.save(BOTTLENECK_MODEL_PATH)
    autoencoder.save(MODEL_PATH)

    bottleneck_test = encoder.predict(X_test_scaled)
    reconstructed_test = autoencoder.predict(X_test_scaled)
    error_test = compute_reconstruction_error(X_test_scaled, reconstructed_test)
    features_test = augment_with_error(bottleneck_test, error_test)

    bottleneck_train = encoder.predict(X_train_scaled)
    reconstructed_train = autoencoder.predict(X_train_scaled)
    error_train = compute_reconstruction_error(X_train_scaled, reconstructed_train)
    features_train = augment_with_error(bottleneck_train, error_train)

    smote = SMOTE(random_state=42)
    features_resampled, y_resampled = smote.fit_resample(features_train, y_train)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(features_resampled, y_resampled)
    joblib.dump(rf_clf, RF_PATH)

    y_pred = rf_clf.predict(features_test)
    y_proba = rf_clf.predict_proba(features_test)[:, 1]

    return {
        'model': 'Autoencoders',
        'MSE': float(mean_squared_error(y_test, y_pred)),
        'MAE': float(mean_absolute_error(y_test, y_pred)),
        'CrossEntropyLoss': float(log_loss(y_test, y_pred)),
        'message': 'Model and anomaly features saved.'
    }


def predict_autoencoder():
    X, y, df = load_dataset()
    encoder, autoencoder, rf_clf, scaler = load_models()
    X_scaled = scaler.transform(X)

    bottleneck = encoder.predict(X_scaled, verbose=0)
    reconstructed = autoencoder.predict(X_scaled, verbose=0)
    error = compute_reconstruction_error(X_scaled, reconstructed)
    features = augment_with_error(bottleneck, error)

    preds = rf_clf.predict(features)
    df["Predicted"] = preds

    true_pos = df[(df["Class"] == 1) & (df["Predicted"] == 1)]
    true_neg = df[(df["Class"] == 0) & (df["Predicted"] == 0)]

    sample_tp = true_pos.sample(n=min(3, len(true_pos)), random_state=42)
    sample_tn = true_neg.sample(n=min(2, len(true_neg)), random_state=42)

    sample = pd.concat([sample_tp, sample_tn]).sample(frac=1, random_state=99)
    return sample.to_dict(orient="records")

def predict_autoencoder_manual(user_input):
    try:
        encoder, autoencoder, rf_clf, scaler = load_models()

        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        bottleneck = encoder.predict(scaled_input, verbose=0)
        reconstructed = autoencoder.predict(scaled_input, verbose=0)
        error = compute_reconstruction_error(scaled_input, reconstructed).reshape(-1, 1)

        features = augment_with_error(bottleneck, error)
        probs = rf_clf.predict_proba(features)[0]
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])

        return {"is_fraud": prediction, "confidence": confidence}

    except Exception as e:
        logging.exception("Error in predict_autoencoder_manual")
        return {"error": str(e)}
