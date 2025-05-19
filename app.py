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
            print(f"Warning: Model file not found at {MODEL_PATH}. Creating a simple placeholder model.")
            # Create a simple placeholder model
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
            # Fit with dummy data
            X = [[0, 50], [5, 80]]  # price_change_1h, rsi_14
            y = [0, 1]  # decrease, increase
            model.fit(X, y)
            print("Created placeholder model for deployment")

            # Save the directory and model for future use
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            print(f"Saved placeholder model to {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading/creating model: {e}")
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
        response = jsonify({
            "predicted_outcome": "neutral (model not loaded)",
            "confidence_score": 0.50,
            "notes": "ML model not found or failed to load. Running in placeholder mode.",
            "model_features_used": {"price_change_1h": 0.0, "rsi_14": 50.0}
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 503

    try:
        request_data = request.get_json()
        print(f"Request data: {request_data}")

        if not request_data:
            print("Error: No input data provided")
            response = jsonify({"error": "No input data provided"})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400

        # Extract input values
        entry_price = float(request_data.get('entry_price', 0))
        stop_loss = float(request_data.get('stop_loss', 0))
        take_profit = float(request_data.get('take_profit', 0))
        position_size = float(request_data.get('position_size', 0))
        account_size = float(request_data.get('account_size', 0))
        trade_type = request_data.get('trade_type', 'long')

        print(f"Input values - Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}, Type: {trade_type}")

        if entry_price <= 0:
            print("Error: Invalid entry price")
            response = jsonify({"error": "Invalid entry price. Must be greater than 0."})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400

        # Calculate features for prediction
        # For long trades: positive price_change means price went up
        # For short trades: positive price_change means price went down
        if trade_type == 'long':
            # Calculate price change as % difference between entry and stop
            price_change_1h = ((entry_price - stop_loss) / entry_price * 100) if stop_loss > 0 else 1.0

            # Calculate risk/reward ratio
            if take_profit > 0 and stop_loss > 0:
                risk = entry_price - stop_loss
                reward = take_profit - entry_price
                rr_ratio = reward / risk if risk > 0 else 1.0
            else:
                rr_ratio = 1.0
        else:  # short trade
            # For shorts, price change is reversed
            price_change_1h = ((stop_loss - entry_price) / entry_price * 100) if stop_loss > 0 else -1.0

            # Calculate risk/reward ratio for shorts
            if take_profit > 0 and stop_loss > 0:
                risk = stop_loss - entry_price
                reward = entry_price - take_profit
                rr_ratio = reward / risk if risk > 0 else 1.0
            else:
                rr_ratio = 1.0

        # Map RR ratio to RSI (higher RR = higher RSI for longs, lower RSI for shorts)
        if trade_type == 'long':
            # For long trades, higher RR ratio suggests higher RSI
            rsi_14 = min(85, max(30, 50 + (rr_ratio - 1) * 15))
        else:
            # For short trades, higher RR ratio suggests lower RSI
            rsi_14 = min(70, max(15, 50 - (rr_ratio - 1) * 15))

        print(f"Calculated features - price_change_1h: {price_change_1h:.2f}, rsi_14: {rsi_14:.2f}")

        # Create feature dataframe
        features = {'price_change_1h': price_change_1h, 'rsi_14': rsi_14}
        features_df = pd.DataFrame([features])

        print(f"Features for model: {features}")

        # Make prediction
        prediction = model.predict(features_df)
        proba = model.predict_proba(features_df)

        predicted_class = int(prediction[0])

        # Get confidence for the predicted class
        confidence = float(proba[0][predicted_class])

        # Map class to outcome
        outcome_map = {
            0: "likely_decrease_or_stagnate",
            1: "likely_increase_soon"
        }

        predicted_outcome = outcome_map.get(predicted_class, "unknown_prediction")

        # Generate analysis notes based on prediction and trade type
        if trade_type == 'long':
            if predicted_class == 1:  # likely increase
                analysis_notes = "Model predicts price is likely to increase. This aligns with your LONG position."
            else:  # likely decrease
                analysis_notes = "Model predicts price may decrease or stagnate. Consider caution with your LONG position."
        else:  # short trade
            if predicted_class == 0:  # likely decrease
                analysis_notes = "Model predicts price is likely to decrease or stagnate. This aligns with your SHORT position."
            else:  # likely increase
                analysis_notes = "Model predicts price may increase. Consider caution with your SHORT position."

        print(f"Prediction result: {predicted_outcome} with confidence {confidence:.2f}")
        print(f"Analysis: {analysis_notes}")

        # Prepare response
        response_data = {
            "predicted_outcome": predicted_outcome,
            "confidence_score": confidence,
            "notes": analysis_notes,
            "model_features_used": features
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

# Load the model when the app is initialized
load_trained_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)