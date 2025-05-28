import logging
import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from preprocessing import preprocess_data, get_current_dataset
from randomforest import train_and_save_model, train_combined_model, load_and_predict_bulk, evaluate_combined_model
from isolationforest import train_isolation_forest
from autoencoder_backend import train_autoencoder, predict_autoencoder, predict_autoencoder_manual, load_models, compute_reconstruction_error, augment_with_error

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        result = preprocess_data(dataset_filename=file_path)
        return jsonify(result)

    except Exception as e:
        logging.exception('Error in /preprocess')
        return jsonify({'error': str(e)}), 500

@app.route('/train/randomforest', methods=['POST'])
def train_rf():
    try:
        data = request.get_json(force=True)
        model_name = data.get('name', 'random_forest.pkl')
        model_path = os.path.join('models', model_name if model_name.endswith('.pkl') else f"{model_name}.pkl")
        result = train_and_save_model(model_path=model_path)
        return jsonify(result)
    except Exception as e:
        logging.exception('Error in /train/randomforest')
        return jsonify({'error': str(e)}), 500

@app.route('/train/isolationforest', methods=['POST'])
def train_iso():
    try:
        result = train_isolation_forest()
        return jsonify({
            "message": "Isolation Forest trained successfully",
            "metrics": result
        })
    except Exception as e:
        logging.exception('Error in /train/isolationforest')
        return jsonify({'error': str(e)}), 500

@app.route('/train/autoencoder', methods=['POST'])
def train_autoencoder_route():
    try:
        result = train_autoencoder()
        return jsonify({
            "message": "Autoencoder trained successfully",
            "metrics": result
        })
    except Exception as e:
        logging.exception("Error training Autoencoder")
        return jsonify({"error": str(e)}), 500

@app.route('/train/combined', methods=['POST'])
def train_combined():
    try:
        result = train_combined_model()
        return jsonify({
            "message": "Combined model trained successfully",
            "metrics": result["metrics"]
        })
    except Exception as e:
        logging.exception("Error training combined model")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/combined', methods=['GET'])
def predict_combined():
    try:
        encoder, autoencoder, _, scaler = load_models()

        dataset_path = get_current_dataset()
        if dataset_path is None or not os.path.exists(dataset_path):
            raise FileNotFoundError("No current dataset found. Please upload and preprocess a dataset first.")
        df = pd.read_csv(dataset_path)

        X = df.drop(columns=["Class"])
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

        clf = joblib.load("models/combined_rf.pkl")
        preds = clf.predict(combined_features)
        df["Predicted"] = preds

        top_100 = df[df["Predicted"] == 1].head(100).to_dict(orient="records")
        return jsonify({"predictions": top_100})

    except Exception as e:
        logging.exception("Error in /predict/combined")
        return jsonify({'error': str(e)}), 500
    
@app.route('/predict/randomforest/all', methods=['GET'])
def eval_rf_all():
    try:
        result = load_and_predict_bulk()
        return jsonify({
            "predictions": result["predictions"],
            "stats": result["stats"],
            "message": f"Random Forest evaluated on {len(result['predictions'])} samples."
        })
    except Exception as e:
        logging.exception("Error evaluating random forest")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/combined/all', methods=['GET'])
def eval_combined_all():
    try:
        result = evaluate_combined_model()
        return jsonify({
            "predictions": result["predictions"],
            "stats": result["stats"],
            "message": f"Combined Model evaluated on {len(result['predictions'])} samples."
        })
    except Exception as e:
        logging.exception("Error evaluating combined model")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/randomforest/manual', methods=['POST'])
def predict_rf_manual():
    try:
        model_data = joblib.load('models/random_forest.pkl')

        if not isinstance(model_data, dict):
            raise ValueError("Model file is corrupted or outdated. Retrain using /train/randomforest.")

        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']

        user_data = request.get_json(force=True)

        missing = [f for f in features if f not in user_data]
        if missing:
            return jsonify({'error': f"Missing features: {missing}"}), 400

        input_values = [float(user_data[f]) for f in features]

        df = pd.DataFrame([input_values], columns=features)
        df_scaled = scaler.transform(df)

        prediction = int(model.predict(df_scaled)[0])
        label = 'Fraudulent' if prediction == 1 else 'Not Fraudulent'

        return jsonify({'prediction': prediction, 'label': label})
    except Exception as e:
        logging.exception("Error in /predict/randomforest/manual")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/combined/manual', methods=['POST'])
def predict_combined_manual():
    try:
        user_data = request.get_json(force=True)
        missing = [f for f in FEATURE_ORDER if f not in user_data]
        if missing:
            return jsonify({'error': f"Missing features: {missing}"}), 400

        input_values = [float(user_data[feature]) for feature in FEATURE_ORDER]
        encoder, autoencoder, _, scaler = load_models()

        input_array = np.array(input_values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        bottleneck = encoder.predict(scaled_input, verbose=0)
        reconstructed = autoencoder.predict(scaled_input, verbose=0)
        error = compute_reconstruction_error(scaled_input, reconstructed).reshape(-1, 1)
        auto_features = augment_with_error(bottleneck, error)

        iso_data = np.load("data/processed/iso_scores.npz")
        iso_scores = np.vstack([iso_data['train'], iso_data['test']])
        if iso_scores.shape[0] < 1:
            raise ValueError("Isolation Forest scores file is empty.")
        iso_score = iso_scores[0].reshape(1, -1)

        combined_features = np.hstack((auto_features, iso_score))

        clf = joblib.load("models/combined_rf.pkl")
        probs = clf.predict_proba(combined_features)[0]
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])

        label = "Fraudulent" if prediction == 1 else "Not Fraudulent"
        return jsonify({
            "prediction": prediction,
            "label": label,
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        logging.exception("Error in /predict/combined/manual")
        return jsonify({'error': str(e)}), 500

@app.route('/train/all_models', methods=['POST'])
def train_all_models():
    try:
        rf_result = train_and_save_model()
        combined_result = train_combined_model()

        return jsonify({
        "rf": {
            "accuracy": rf_result["stats"].get("accuracy"),
            "precision": rf_result["stats"].get("precision"),
            "recall": rf_result["stats"].get("recall"),
            "f1_score": rf_result["stats"].get("f1_score"),
        },
        "combined": combined_result["metrics"]
    })

    except Exception as e:
        logging.exception("Error in training all models")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5005)