# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras_tuner as kt
from tensorflow.keras.layers import Layer, Input, Dense, LSTM, Dropout, Activation, Multiply, Permute, Flatten, RepeatVector, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import requests
import json
from keras import backend as K

sns.set()
tf.random.set_seed(1234)

# Install yfinance if not already installed

# %%
import yfinance as yf

# Download BTC-USD data for the past 5 years
btc_df = yf.download('BTC-USD', period='5y', interval='1d')

# Reset the index to ensure 'Date' is a column
btc_df.reset_index(inplace=True)

# Rename the columns correctly to avoid duplicates
btc_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Display the BTC DataFrame structure to confirm
print("BTC Price Data:")
print(btc_df.head())
print(btc_df.columns)  # Check the column names for further inspection

# Define the SentiCrypt v2 API endpoint
senticrypt_api_url = "https://api.senticrypt.com/v2/all.json"

# Fetch sentiment data
response = requests.get(senticrypt_api_url)

# Check if the request was successful
if response.status_code == 200:
    sentiment_data = response.json()
    print("Sentiment data fetched successfully!")
else:
    print(f"Failed to fetch sentiment data. Status code: {response.status_code}")

# Convert the JSON data to a DataFrame
sentiment_df = pd.DataFrame(sentiment_data)
print("Sentiment Data:")
print(sentiment_df.head())

# Ensure that the 'Date' column in btc_df is in the same format as the 'date' column in sentiment_df
btc_df['Date'] = pd.to_datetime(btc_df['Date']).dt.date  # Convert to date format
sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date  # Convert to date format

# Merge BTC price data with sentiment data on the date
merged_df = pd.merge(btc_df, sentiment_df, left_on='Date', right_on='date', how='inner')

# Drop the redundant 'date' column from the merged DataFrame
merged_df.drop(columns=['date'], inplace=True)

print("Merged Data:")
print(merged_df.head())

# Now continue with feature engineering



# %%
# Feature Engineering: Adding Technical Indicators
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
    
    # Volume Change (assuming 'Volume' column exists)
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

# Apply feature engineering
merged_df = add_technical_indicators(merged_df)
print("Data after Feature Engineering:")
print(merged_df.head())

# %%
# Selecting features and target
feature_columns = ['Close', 'mean', 'sum', 'MA5', 'MA10', 'MA20',
                   'STD5', 'STD10', 'STD20', 'Return', 'Volume_Change', 'RSI', 'MACD']
target_column = 'Close'

# Initialize scalers
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Fit and transform the features
scaled_features = scaler_features.fit_transform(merged_df[feature_columns])
scaled_target = scaler_target.fit_transform(merged_df[[target_column]])

# Convert to DataFrame for easier handling
scaled_features_df = pd.DataFrame(scaled_features, columns=feature_columns)
scaled_target_df = pd.DataFrame(scaled_target, columns=[target_column])

# Combine scaled features and target
scaled_df = pd.concat([scaled_features_df, scaled_target_df], axis=1)
print("Scaled Data:")
print(scaled_df.head())

# %%
# Define window size (number of past days to consider for prediction)
window_size = 500

# Create sequences
X = []
y = []

for i in range(window_size, len(scaled_df)):
    X.append(scaled_features[i-window_size:i])
    y.append(scaled_target[i][0])

X = np.array(X)
y = np.array(y)

print(f'X shape: {X.shape}, y shape: {y.shape}')

# %%
# Define dataset sizes
test_size = 30
validation_size = 30
train_size = len(X) - test_size - validation_size

# Split the data
X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + validation_size]
y_val = y[train_size:train_size + validation_size]

X_test = X[train_size + validation_size:]
y_test = y[train_size + validation_size:]

print(f'Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}')

# %%
# Define the custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
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
        # Compute attention scores
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        e = K.squeeze(e, axis=-1)  # Shape: (batch_size, time_steps)
        alpha = K.softmax(e)        # Shape: (batch_size, time_steps)
        alpha = K.expand_dims(alpha, axis=-1)  # Shape: (batch_size, time_steps, 1)
        # Compute context vector
        context = inputs * alpha
        context = K.sum(context, axis=1)  # Shape: (batch_size, features)
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# %%
# Define the model-building function with Keras Tuner
def build_model(hp):
    # Hyperparameters to tune
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    # LSTM Layer
    x = LSTM(units=lstm_units, return_sequences=True)(inputs)
    x = Dropout(rate=dropout_rate)(x)
    
    # Attention Layer
    attention = AttentionLayer()(x)
    
    # Fully Connected Layers
    x = Dense(units=64, activation='relu')(attention)
    x = Dropout(rate=dropout_rate)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

# %%
# Initialize Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,  # Adjust based on your computational resources
    executions_per_trial=1,
    directory='my_dir',
    project_name='btc_price_prediction'
)

# Display search space summary
tuner.search_space_summary()

# %%
# Define Early Stopping to prevent overfitting during hyperparameter search
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Run Hyperparameter Search
tuner.search(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# %%
# Display tuner results
tuner.results_summary()

# %%
# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of LSTM units is {best_hps.get('lstm_units')},
the optimal dropout rate is {best_hps.get('dropout_rate')}, and the optimal learning rate is {best_hps.get('learning_rate')}.
""")

# %%
# Build the model with the best hyperparameters
model = build_model(best_hps)

# Combine training and validation data
X_final_train = np.concatenate([X_train, X_val], axis=0)
y_final_train = np.concatenate([y_train, y_val], axis=0)

# Define Early Stopping
early_stopping_final = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the final model
history = model.fit(
    X_final_train, y_final_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,  # Further split for validation during final training
    callbacks=[early_stopping_final]
)

# %%
# Predict on the test set
y_pred = model.predict(X_test)

# Rescale the predictions and actual values
y_pred_rescaled = scaler_target.inverse_transform(y_pred)
y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

print(f'Test MAE: {mae:.2f}')
print(f'Test RMSE: {rmse:.2f}')

# %%
# Create a DataFrame for visualization
test_dates = merged_df['Date'].iloc[-len(y_test_rescaled):].reset_index(drop=True)
results = pd.DataFrame({
    'Date': test_dates,
    'True_Price': y_test_rescaled.flatten(),
    'Predicted_Price': y_pred_rescaled.flatten()
})

# Apply Exponential Moving Average for smoothing (optional)
def exponential_moving_average(signal, alpha=0.3):
    return signal.ewm(alpha=alpha).mean()

results['Predicted_SMA'] = exponential_moving_average(results['Predicted_Price'])
results['True_SMA'] = exponential_moving_average(results['True_Price'])

# Plotting
plt.figure(figsize=(15, 7))
plt.plot(results['Date'], results['True_Price'], label='True Prices', color='black')
plt.plot(results['Date'], results['Predicted_SMA'], label='Predicted Prices (EMA)', color='red')
plt.title('Bitcoin Price Prediction with Optimized LSTM and Attention Mechanism')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.show()

import os
import pickle

# Create the model directory if it doesn't exist
model_dir = './model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model
model.save(os.path.join(model_dir, 'btc_price_model.keras'))

# Save the scalers
with open(os.path.join(model_dir, 'scaler_features.pkl'), 'wb') as f:
    pickle.dump(scaler_features, f)

with open(os.path.join(model_dir, 'scaler_target.pkl'), 'wb') as f:
    pickle.dump(scaler_target, f)

# Save the feature columns
with open(os.path.join(model_dir, 'feature_columns.pkl'), 'wb') as f:
    pickle.dump(feature_columns, f)

# Save the target column
with open(os.path.join(model_dir, 'target_column.pkl'), 'wb') as f:
    pickle.dump(target_column, f)

# Save hyperparameters
hyperparams = {
    'window_size': window_size,
    'lstm_units': best_hps.get('lstm_units'),
    'dropout_rate': best_hps.get('dropout_rate'),
    'learning_rate': best_hps.get('learning_rate')
}

with open(os.path.join(model_dir, 'hyperparams.pkl'), 'wb') as f:
    pickle.dump(hyperparams, f)

print("All necessary data saved in './model/' directory.")