# %%
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

# %%
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Activation, Multiply, Permute, Flatten, RepeatVector, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import keras_tuner as kt
sns.set()
tf.random.set_seed(1234)

# %%
# Downloading BTC-USD data for the past 2 years
df = yf.download('BTC-USD', period='2y', interval='1d')
df.reset_index(inplace=True)
print(df.head())

# %%
# Feature Engineering: Adding Technical Indicators
def add_indicators(data):
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['STD5'] = data['Close'].rolling(window=5).std()
    data['STD10'] = data['Close'].rolling(window=10).std()
    data['STD20'] = data['Close'].rolling(window=20).std()
    data['Return'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['RSI'] = compute_rsi(data['Close'], window=14)
    data['MACD'] = compute_macd(data['Close'])
    data = data.dropna()
    return data

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
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

df = add_indicators(df)
print(df.head())

# %%
# Selecting features and target
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'MA5', 'MA10', 'MA20', 'STD5', 'STD10', 'STD20', 'Return', 'Volume_Change', 'RSI', 'MACD']
target = 'Close'

# Scaling the data
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

scaled_features = scaler_features.fit_transform(df[features])
scaled_target = scaler_target.fit_transform(df[[target]])

# Define window size (number of past days to consider for prediction)
window_size = 60

# Create sequences
X = []
y = []

for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i-window_size:i])
    y.append(scaled_target[i])

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
# Define the model-building function with Keras Tuner
def build_model(hp):
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    # LSTM layers
    x = LSTM(units=lstm_units, return_sequences=True)(inputs)
    x = Dropout(rate=dropout_rate)(x)
    
    # Attention Mechanism
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(lstm_units)(attention)
    attention = Permute([2, 1])(attention)
    attention_output = Multiply()([x, attention])
    attention_output = Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(attention_output)
    
    # Fully connected layers
    x = Dense(units=64, activation='relu')(attention_output)
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
    max_trials=20,  # Adjust based on resources
    executions_per_trial=1,
    directory='my_dir',
    project_name='btc_price_prediction'
)

tuner.search_space_summary()

# %%
# Define Early Stopping
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
# Display the best hyperparameters
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

# Train the final model
history = model.fit(
    X_final_train, y_final_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,  # Further split for validation during final training
    callbacks=[early_stopping]
)

# %%
# Predict on the test set
y_pred = model.predict(X_test)

# Rescale the predictions and actual values
y_pred_rescaled = scaler_target.inverse_transform(y_pred)
y_test_rescaled = scaler_target.inverse_transform(y_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

print(f'Test MAE: {mae:.2f}')
print(f'Test RMSE: {rmse:.2f}')

# %%
# Create a DataFrame for visualization
test_dates = df['Date'].iloc[-len(y_test_rescaled):].reset_index(drop=True)
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
plt.show()
