from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# --- Model Loading ---
MODEL_DIR = 'trained_model'
MODEL_FILE_NAME = 'trade_predictor_model.joblib'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

model = None

def load_trained_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}. Predictions will be placeholders.")
            model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# --- Feature Engineering (must match model_trainer.py) ---
def engineer_features_for_prediction(data):
    """Prepares input data for prediction, mirroring model_trainer.py feature engineering."""
    price_change_1h = data.get('price_change_1h', 0.0)
    rsi_14 = data.get('rsi_14', 50.0)
    feature_df = pd.DataFrame([[price_change_1h, rsi_14]], columns=['price_change_1h', 'rsi_14'])
    return feature_df

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({
            "predicted_outcome": "neutral (model not loaded)",
            "confidence_score": 0.50,
            "notes": "ML model not found or failed to load. Running in placeholder mode."
        }), 503

    request_data = request.get_json()
    if not request_data:
        return jsonify({"error": "No input data provided"}), 400

    entry_price = request_data.get('entry_price')
    if entry_price is None:
        return jsonify({"error": "'entry_price' is required for prediction context."}), 400

    features_df = engineer_features_for_prediction(request_data)

    try:
        prediction = model.predict(features_df)
        proba = model.predict_proba(features_df)
        
        predicted_class = int(prediction[0])
        confidence = float(proba[0][predicted_class])

        outcome_map = {0: "likely_decrease_or_stagnate", 1: "likely_increase_soon"}

        return jsonify({
            "predicted_outcome": outcome_map.get(predicted_class, "unknown_prediction"),
            "confidence_score": confidence,
            "notes": "Prediction from trained (dummy) model.",
            "model_features_used": features_df.to_dict(orient='records')[0]
        })
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    load_trained_model()
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)