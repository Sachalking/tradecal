from flask import Flask, request, jsonify, send_from_directory, make_response
import joblib
import pandas as pd
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
# Enable CORS for all routes and all origins
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

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
    # We expect 'data' to be a dictionary with keys like 'current_price', 'rsi_14_manual', etc.
    # Or, if we decide to pass a mini-dataframe of recent history:
    # For this example, let's assume we need to calculate features based on a single point
    # or a very short history passed in the request.
    # This part is highly dependent on what features the model was trained on.

    # For the dummy model trained in model_trainer.py, it expects 'price_change_1h' and 'rsi_14'.
    # This is tricky for a single API call without historical context.
    # A more robust API would either:
    # 1. Expect the client to send pre-calculated features.
    # 2. Have access to a data source to fetch recent history and calculate features.

    # --- SIMPLIFIED/PLACEHOLDER FEATURE CALCULATION FOR API ---
    # This is a placeholder. In a real system, you'd need a robust way to get/calculate these.
    # For now, let's assume the client might send some of these, or we use defaults.
    price_change_1h = data.get('price_change_1h', 0.0) # Default to 0 if not provided
    rsi_14 = data.get('rsi_14', 50.0) # Default to neutral RSI if not provided

    # Create a DataFrame in the structure the model expects
    feature_df = pd.DataFrame([[price_change_1h, rsi_14]], columns=['price_change_1h', 'rsi_14'])
    return feature_df

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    global model
    print("Received prediction request")

    if model is None:
        print("Warning: Model not loaded, returning placeholder response")
        # Fallback to old placeholder if model isn't loaded
        return jsonify({
            "predicted_outcome": "neutral (model not loaded)",
            "confidence_score": 0.50,
            "notes": "ML model not found or failed to load. Running in placeholder mode."
        }), 503 # Service Unavailable

    try:
        request_data = request.get_json()
        print(f"Request data: {request_data}")

        if not request_data:
            print("Error: No input data provided")
            return jsonify({"error": "No input data provided"}), 400

        # The user's original inputs
        entry_price = request_data.get('entry_price')
        stop_loss = request_data.get('stop_loss')
        take_profit = request_data.get('take_profit')
        position_size = request_data.get('position_size')
        account_size = request_data.get('account_size')
        trade_type = request_data.get('trade_type')

        print(f"Input values - Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}, Type: {trade_type}")

        if entry_price is None: # Basic check
            print("Error: entry_price is required")
            return jsonify({"error": "'entry_price' is required for prediction context (even if not a direct model feature)."}), 400

        # --- Feature Engineering for Prediction ---
        # For our dummy model, we'll calculate some basic features based on the inputs

        # Calculate price_change_1h based on trade_type and entry/stop values
        # This is just a placeholder calculation for demonstration
        if trade_type == 'long':
            # For long trades, use the difference between entry and stop as a proxy for recent price change
            price_change_1h = (entry_price - stop_loss) / entry_price * 100 if stop_loss else 0.5
        else:
            # For short trades, use the difference between stop and entry
            price_change_1h = (stop_loss - entry_price) / entry_price * 100 if stop_loss else -0.5

        # Calculate a simple RSI proxy based on the risk/reward ratio
        # This is just a placeholder calculation for demonstration
        if take_profit and stop_loss and entry_price:
            if trade_type == 'long':
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit

            rr_ratio = reward / risk if risk > 0 else 1.0
            # Map RR ratio to an RSI-like value (higher RR = higher RSI)
            rsi_14 = min(85, max(30, 50 + (rr_ratio - 1) * 20))
        else:
            rsi_14 = 50.0  # Neutral RSI

        # Create request data with calculated features
        request_data['price_change_1h'] = price_change_1h
        request_data['rsi_14'] = rsi_14

        print(f"Calculated features - price_change_1h: {price_change_1h}, rsi_14: {rsi_14}")

        # Get features in the format the model expects
        features_df = engineer_features_for_prediction(request_data)
        print(f"Features for model: {features_df.to_dict(orient='records')[0]}")

        # Make prediction
        prediction = model.predict(features_df)
        proba = model.predict_proba(features_df)

        predicted_class = int(prediction[0])
        confidence = float(proba[0][predicted_class])

        outcome_map = {0: "likely_decrease_or_stagnate", 1: "likely_increase_soon"}
        predicted_outcome = outcome_map.get(predicted_class, "unknown_prediction")

        print(f"Prediction result: {predicted_outcome} with confidence {confidence:.2f}")

        # Prepare response
        response_data = {
            "predicted_outcome": predicted_outcome,
            "confidence_score": confidence,
            "notes": f"Prediction based on calculated features from your inputs. Trade type: {trade_type}.",
            "model_features_used": features_df.to_dict(orient='records')[0]
        }

        print(f"Sending response: {response_data}")

        # Create response with CORS headers
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(f"Exception: {error_msg}")
        import traceback
        traceback.print_exc()

        # Create error response with CORS headers
        response = jsonify({"error": error_msg})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500


if __name__ == '__main__':
    load_trained_model() # Load the model when the app starts
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)