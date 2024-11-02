#!/usr/bin/env python3
# continuous_learning.py

"""
Continuous Learning Script for Bitcoin Price Prediction Model

This script enables the model to continuously learn from new data,
thereby maintaining its predictive performance over time.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import pickle
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Configure Logging
logging.basicConfig(
    filename='continuous_learning.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define Constants
MODEL_DIR = './model'
MODEL_PATH = os.path.join(MODEL_DIR, 'btc_price_model.keras')
SCALER_FEATURES_PATH = os.path.join(MODEL_DIR, 'scaler_features.pkl')
SCALER_TARGET_PATH = os.path.join(MODEL_DIR, 'scaler_target.pkl')
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, 'feature_columns.pkl')
TARGET_COLUMN_PATH = os.path.join(MODEL_DIR, 'target_column.pkl')
HYPERPARAMS_PATH = os.path.join(MODEL_DIR, 'hyperparams.pkl')

SENTICRYPT_API_URL = "https://api.senticrypt.com/v2/all.json"
DATA_FETCH_PERIOD = '1y'  # Fetch 1 year of new data
WINDOW_SIZE = 500  # Default window size; will be updated from hyperparameters

# Custom Attention Layer Definition (Must match the one used during training)
from tensorflow.keras.layers import Layer
from keras import backend as K

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, training=False):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        context = K.sum(context, axis=1)
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Feature Engineering Functions (Must match the ones used during training)
def add_technical_indicators(data):
    """Add technical indicators to the data."""
    # Moving Averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    # Standard Deviations
    data['STD5'] = data['Close'].rolling(window=5).std()
    data['STD10'] = data['Close'].rolling(window=10).std()
    data['STD20'] = data['Close'].rolling(window=20).std()
    
    # Daily Returns
    data['Return'] = data['Close'].pct_change()
    
    # Volume Change
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # RSI Calculation
    data['RSI'] = compute_rsi(data['Close'], window=14)
    
    # MACD Calculation
    data['MACD'] = compute_macd(data['Close'])
    
    # Handle any remaining missing values
    data.dropna(inplace=True)
    return data

def compute_rsi(series, window=14):
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=1).mean()
    ma_down = down.rolling(window=window, min_periods=1).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    """Compute Moving Average Convergence Divergence (MACD)."""
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    return macd - signal

# Utility Functions
def load_artifacts():
    """Load model and related artifacts."""
    try:
        logging.info("Loading model and artifacts...")
        model = load_model(MODEL_PATH, custom_objects={'AttentionLayer': AttentionLayer})
        
        with open(SCALER_FEATURES_PATH, 'rb') as f:
            scaler_features = pickle.load(f)
        
        with open(SCALER_TARGET_PATH, 'rb') as f:
            scaler_target = pickle.load(f)
        
        with open(FEATURE_COLUMNS_PATH, 'rb') as f:
            feature_columns = pickle.load(f)
        
        with open(TARGET_COLUMN_PATH, 'rb') as f:
            target_column = pickle.load(f)
        
        with open(HYPERPARAMS_PATH, 'rb') as f:
            hyperparams = pickle.load(f)
        
        window_size = hyperparams.get('window_size', WINDOW_SIZE)
        
        logging.info("Model and artifacts loaded successfully.")
        return model, scaler_features, scaler_target, feature_columns, target_column, hyperparams, window_size
    except Exception as e:
        logging.error(f"Error loading artifacts: {e}")
        raise

def fetch_new_data(period='1y', interval='1d'):
    """Fetch new Bitcoin price data from Yahoo Finance."""
    try:
        logging.info(f"Fetching new BTC-USD data for the past {period}...")
        btc_df = yf.download('BTC-USD', period=period, interval=interval)
        btc_df.reset_index(inplace=True)
        btc_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        logging.info("BTC-USD data fetched successfully.")
        return btc_df
    except Exception as e:
        logging.error(f"Error fetching BTC-USD data: {e}")
        raise

def fetch_sentiment_data():
    """Fetch sentiment data from SentiCrypt API."""
    try:
        logging.info("Fetching sentiment data from SentiCrypt API...")
        response = requests.get(SENTICRYPT_API_URL)
        if response.status_code == 200:
            sentiment_data = response.json()
            sentiment_df = pd.DataFrame(sentiment_data)
            logging.info("Sentiment data fetched successfully.")
            return sentiment_df
        else:
            error_msg = f"Failed to fetch sentiment data. Status code: {response.status_code}"
            logging.error(error_msg)
            raise Exception(error_msg)
    except Exception as e:
        logging.error(f"Error fetching sentiment data: {e}")
        raise

def merge_data(btc_df, sentiment_df):
    """Merge BTC price data with sentiment data on the date."""
    try:
        logging.info("Merging BTC price data with sentiment data...")
        btc_df['Date'] = pd.to_datetime(btc_df['Date']).dt.date
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date
        merged_df = pd.merge(btc_df, sentiment_df, left_on='Date', right_on='date', how='inner')
        merged_df.drop(columns=['date'], inplace=True)
        logging.info("Data merged successfully.")
        return merged_df
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise

def preprocess_data(merged_df, feature_columns, scaler_features):
    """Apply feature engineering and scale the features."""
    try:
        logging.info("Applying feature engineering to merged data...")
        merged_df = add_technical_indicators(merged_df)
        
        logging.info("Scaling features...")
        scaled_features = scaler_features.transform(merged_df[feature_columns])
        scaled_features_df = pd.DataFrame(scaled_features, columns=feature_columns)
        
        logging.info("Feature engineering and scaling completed.")
        return scaled_features_df, merged_df
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        raise

def prepare_sequences(scaled_features, window_size, target_column):
    """Create sequences for LSTM input."""
    try:
        logging.info("Preparing sequences for LSTM...")
        X = []
        y = []
        
        for i in range(window_size, len(scaled_features)):
            X.append(scaled_features[i-window_size:i].values)
            y.append(scaled_features.iloc[i][target_column])
        
        X = np.array(X)
        y = np.array(y)
        
        logging.info(f"Sequences prepared: X shape = {X.shape}, y shape = {y.shape}")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing sequences: {e}")
        raise

def split_data(X, y, train_ratio=0.8, validation_ratio=0.1):
    """Split data into training, validation, and testing sets."""
    try:
        total_samples = len(X)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * validation_ratio)
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        logging.info(f"Data split into Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def retrain_model(model, X_train, y_train, X_val, y_val, hyperparams):
    """Retrain the model with new data."""
    try:
        logging.info("Starting model retraining...")
        lstm_units = hyperparams.get('lstm_units', 128)
        dropout_rate = hyperparams.get('dropout_rate', 0.3)
        learning_rate = hyperparams.get('learning_rate', 1e-3)
        
        # Compile the model with updated hyperparameters if necessary
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        
        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Retrain the model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        logging.info("Model retraining completed.")
        return model, history
    except Exception as e:
        logging.error(f"Error during model retraining: {e}")
        raise

def save_model_and_artifacts(model, scaler_features, scaler_target, feature_columns, target_column, hyperparams):
    """Save the retrained model and updated artifacts."""
    try:
        logging.info("Saving retrained model and artifacts...")
        model.save(MODEL_PATH)
        
        with open(SCALER_FEATURES_PATH, 'wb') as f:
            pickle.dump(scaler_features, f)
        
        with open(SCALER_TARGET_PATH, 'wb') as f:
            pickle.dump(scaler_target, f)
        
        with open(FEATURE_COLUMNS_PATH, 'wb') as f:
            pickle.dump(feature_columns, f)
        
        with open(TARGET_COLUMN_PATH, 'wb') as f:
            pickle.dump(target_column, f)
        
        with open(HYPERPARAMS_PATH, 'wb') as f:
            pickle.dump(hyperparams, f)
        
        logging.info("Model and artifacts saved successfully.")
    except Exception as e:
        logging.error(f"Error saving model and artifacts: {e}")
        raise

def backup_existing_model():
    """Backup the existing model before retraining."""
    try:
        backup_dir = os.path.join(MODEL_DIR, 'backup')
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_model_path = os.path.join(backup_dir, f'btc_price_model_{timestamp}.h5')
        backup_scaler_features = os.path.join(backup_dir, f'scaler_features_{timestamp}.pkl')
        backup_scaler_target = os.path.join(backup_dir, f'scaler_target_{timestamp}.pkl')
        backup_feature_columns = os.path.join(backup_dir, f'feature_columns_{timestamp}.pkl')
        backup_target_column = os.path.join(backup_dir, f'target_column_{timestamp}.pkl')
        backup_hyperparams = os.path.join(backup_dir, f'hyperparams_{timestamp}.pkl')
        
        # Copy files
        import shutil
        shutil.copy2(MODEL_PATH, backup_model_path)
        shutil.copy2(SCALER_FEATURES_PATH, backup_scaler_features)
        shutil.copy2(SCALER_TARGET_PATH, backup_scaler_target)
        shutil.copy2(FEATURE_COLUMNS_PATH, backup_feature_columns)
        shutil.copy2(TARGET_COLUMN_PATH, backup_target_column)
        shutil.copy2(HYPERPARAMS_PATH, backup_hyperparams)
        
        logging.info(f"Existing model and artifacts backed up to {backup_dir}.")
    except Exception as e:
        logging.error(f"Error backing up model and artifacts: {e}")
        raise

def main():
    """Main function to orchestrate continuous learning."""
    try:
        logging.info("=== Continuous Learning Process Started ===")
        
        # Step 1: Backup Existing Model and Artifacts
        backup_existing_model()
        
        # Step 2: Load Model and Artifacts
        model, scaler_features, scaler_target, feature_columns, target_column, hyperparams, window_size = load_artifacts()
        
        # Step 3: Fetch New Data
        btc_df = fetch_new_data(period=DATA_FETCH_PERIOD)
        sentiment_df = fetch_sentiment_data()
        
        # Step 4: Merge Data
        merged_df = merge_data(btc_df, sentiment_df)
        
        # Step 5: Preprocess Data
        scaled_features_df, merged_df = preprocess_data(merged_df, feature_columns, scaler_features)
        
        # Step 6: Prepare Sequences
        X, y = prepare_sequences(scaled_features_df, window_size, target_column)
        
        # Step 7: Split Data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
        
        # Step 8: Retrain Model
        model, history = retrain_model(model, X_train, y_train, X_val, y_val, hyperparams)
        
        # Step 9: Save Updated Model and Artifacts
        save_model_and_artifacts(model, scaler_features, scaler_target, feature_columns, target_column, hyperparams)
        
        logging.info("=== Continuous Learning Process Completed Successfully ===")
        
    except Exception as e:
        logging.error(f"An error occurred during the continuous learning process: {e}")
        print(f"An error occurred. Check the log file for details.")

if __name__ == '__main__':
    main()
