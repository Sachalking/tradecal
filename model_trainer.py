import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Configuration ---
DATA_FILE_PATH = 'dummy_ohlcv_data.csv' # User should replace this with their actual data file path
MODEL_DIR = 'trained_model'
MODEL_FILE_NAME = 'trade_predictor_model.joblib'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE_NAME)

# --- 1. Data Loading and Preprocessing (Placeholder) ---
def create_dummy_data(file_path, num_rows=1000):
    """Creates a dummy CSV file if one doesn't exist."""
    if not os.path.exists(file_path):
        print(f"Creating dummy data at {file_path}...")
        dates = pd.date_range(start='2022-01-01', periods=num_rows, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 200, num_rows).round(2),
            'volume': np.random.randint(1000, 10000, num_rows)
        })
        
        # Calculate high and low based on open price
        data['high'] = data['open'] + np.random.uniform(0, 10, num_rows).round(2)
        data['low'] = data['open'] - np.random.uniform(0, 10, num_rows).round(2)
        data['close'] = (data['open'] + data['high'] + data['low']) / 3 + np.random.uniform(-5, 5, num_rows).round(2)
        
        # Ensure high is highest and low is lowest
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        data['close'] = np.clip(data['close'], data['low'], data['high'])
        
        data.to_csv(file_path, index=False)
        print("Dummy data created.")
    else:
        print(f"Dummy data file {file_path} already exists.")

def load_data(file_path):
    """Loads data from a CSV file."""
    print(f"Loading data from {file_path}...")
    # In a real scenario, you'd load your actual trading data here
    # This expects columns like: timestamp, open, high, low, close, volume
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: Data file {file_path} not found. Please create it or provide the correct path.")
        return None

# --- 2. Feature Engineering (Placeholder) ---
def engineer_features(df):
    """Creates features for the model."""
    if df is None:
        return None, None
    print("Engineering features...")
    # Example features (very basic):
    df['price_change_1h'] = df['close'].diff(1)
    df['rsi_14'] = 100 - (100 / (1 + (df['close'].diff(1).fillna(0).clip(lower=0).rolling(window=14).mean() / 
                                     df['close'].diff(1).fillna(0).clip(upper=0).abs().rolling(window=14).mean())))
    df = df.dropna()

    # --- 3. Target Variable Creation (Placeholder) ---
    # Predict if the price will go up by > 1% in the next hour
    df['target'] = (df['close'].shift(-1) > df['close'] * 1.01).astype(int)
    df = df.iloc[:-1] # Remove last row as it has no future data for target
    
    if df.empty:
        print("Error: DataFrame is empty after feature engineering or target creation.")
        return None, None

    features = ['price_change_1h', 'rsi_14'] # User should select relevant features
    X = df[features]
    y = df['target']
    print("Features and target engineered.")
    return X, y

# --- 4. Model Training ---
def train_model(X, y):
    """Trains a logistic regression model."""
    if X is None or y is None or X.empty or y.empty:
        print("Cannot train model: Input data is missing or empty.")
        return None

    print("Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    
    if X_train.empty or y_train.empty:
        print("Error: Training data is empty after split. Check data and target variable distribution.")
        return None

    model = LogisticRegression(random_state=42, class_weight='balanced') # class_weight for imbalanced datasets
    try:
        model.fit(X_train, y_train)
        print("Model training complete.")

        # --- 5. Model Evaluation (Basic) ---
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy on Test Set: {accuracy:.4f}")
        return model
    except Exception as e:
        print(f"Error during model training or evaluation: {e}")
        return None

# --- 6. Model Saving ---
def save_model(model, path):
    """Saves the trained model to a file."""
    if model is None:
        print("No model to save.")
        return
    print(f"Saving model to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print("Model saved successfully.")

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting model training process...")
    # 1. Create or ensure dummy data exists for demonstration
    create_dummy_data(DATA_FILE_PATH)

    # 2. Load Data
    data_df = load_data(DATA_FILE_PATH)

    if data_df is not None:
        # 3. Engineer Features and Target
        X_features, y_target = engineer_features(data_df.copy()) # Use .copy() to avoid SettingWithCopyWarning

        if X_features is not None and y_target is not None and not X_features.empty:
            # 4. Train Model
            trained_model = train_model(X_features, y_target)

            if trained_model is not None:
                # 5. Save Model
                save_model(trained_model, MODEL_PATH)
            else:
                print("Model training failed. Model not saved.")
        else:
            print("Feature engineering failed or produced no data. Model training skipped.")
    else:
        print("Data loading failed. Model training skipped.")
    print("Model training process finished.")