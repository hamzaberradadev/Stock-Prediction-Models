# predict.py

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def predict_future_prices(start_date, num_points):
    # Load the model directory
    model_dir = './model'

    # Load the trained model
    model = load_model(os.path.join(model_dir, 'btc_price_model.keras'), custom_objects={'AttentionLayer': AttentionLayer})

    # Load scalers and configurations
    with open(os.path.join(model_dir, 'scaler_features.pkl'), 'rb') as f:
        scaler_features = pickle.load(f)

    with open(os.path.join(model_dir, 'scaler_target.pkl'), 'rb') as f:
        scaler_target = pickle.load(f)

    with open(os.path.join(model_dir, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)

    with open(os.path.join(model_dir, 'target_column.pkl'), 'rb') as f:
        target_column = pickle.load(f)

    with open(os.path.join(model_dir, 'hyperparams.pkl'), 'rb') as f:
        hyperparams = pickle.load(f)

    window_size = hyperparams['window_size']

    # Convert start_date to datetime
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d').date()

    # Calculate the date range needed for prediction
    end_date_dt = start_date_dt + timedelta(days=num_points + window_size)

    # Fetch BTC price data from start_date - window_size to end_date
    data_start_date = start_date_dt - timedelta(days=window_size)
    btc_df = yf.download('BTC-USD', start=data_start_date.strftime('%Y-%m-%d'), end=end_date_dt.strftime('%Y-%m-%d'), interval='1d')
    btc_df.reset_index(inplace=True)
    btc_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Fetch sentiment data
    senticrypt_api_url = "https://api.senticrypt.com/v2/all.json"
    response = requests.get(senticrypt_api_url)
    if response.status_code == 200:
        sentiment_data = response.json()
    else:
        raise Exception(f"Failed to fetch sentiment data. Status code: {response.status_code}")

    sentiment_df = pd.DataFrame(sentiment_data)
    btc_df['Date'] = pd.to_datetime(btc_df['Date']).dt.date
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date

    # Merge BTC price data with sentiment data
    merged_df = pd.merge(btc_df, sentiment_df, left_on='Date', right_on='date', how='inner')
    merged_df.drop(columns=['date'], inplace=True)

    # Apply feature engineering (same as training)
    merged_df = add_technical_indicators(merged_df)

    # Select features
    scaled_features = scaler_features.transform(merged_df[feature_columns])

    # Prepare sequences
    X_pred = []
    for i in range(window_size, len(scaled_features)):
        X_pred.append(scaled_features[i - window_size:i])

    X_pred = np.array(X_pred)

    # Ensure we have enough data
    if len(X_pred) < num_points:
        raise Exception("Not enough data to make predictions. Please adjust the start date or number of points.")

    # Make predictions
    y_pred = model.predict(X_pred[-num_points:])
    y_pred_rescaled = scaler_target.inverse_transform(y_pred)

    # Prepare results
    prediction_dates = merged_df['Date'].iloc[-num_points:].reset_index(drop=True)
    results = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted_Price': y_pred_rescaled.flatten()
    })

    # Get the last actual prices used for prediction (before the start date)
    last_actual_prices = merged_df[['Date', 'Close']].iloc[-(num_points + num_points):-num_points].reset_index(drop=True)

    return last_actual_prices, results

# Required functions from the training script

def add_technical_indicators(data):
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
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=1).mean()
    ma_down = down.rolling(window=window, min_periods=1).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    return macd - signal

# Custom Attention Layer
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
    
    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        context = K.sum(context, axis=1)
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Main function
if __name__ == '__main__':
    start_date = '2023-01-01'  # Adjust as needed
    num_points = 10  # Number of future points to predict

    # Get the last actual prices and predictions
    last_actual_prices, predictions = predict_future_prices(start_date, num_points)

    # Display the last actual prices used for prediction
    print("Last Actual Prices Used for Prediction:")
    print(last_actual_prices)

    # Display the predicted prices
    print("\nPredicted Future Prices:")
    print(predictions)

    # Combine actual and predicted prices for plotting
    combined_df = pd.concat([last_actual_prices, predictions], ignore_index=True)

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(combined_df['Date'], combined_df['Close'], label='Actual Prices', color='blue')
    plt.plot(combined_df['Date'], combined_df['Predicted_Price'], label='Predicted Prices', color='red', linestyle='--')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()