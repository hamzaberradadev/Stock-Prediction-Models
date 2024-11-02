# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras_tuner as kt
from tensorflow.keras.layers import Layer, Input, Dense, LSTM, Dropout, Activation, Multiply, Permute, Flatten, RepeatVector, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

sns.set()
tf.random.set_seed(1234)

# %%
# Load BTC Sentiment Data
df = pd.read_csv('../dataset/BTC-sentiment.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values('timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)
print("Initial Data:")
print(df.head())

# %%
# Feature Engineering: Adding Technical Indicators
def add_technical_indicators(data):
    # Moving Averages
    data['MA5'] = data['close'].rolling(window=5).mean()
    data['MA10'] = data['close'].rolling(window=10).mean()
    data['MA20'] = data['close'].rolling(window=20).mean()
    
    # Standard Deviations
    data['STD5'] = data['close'].rolling(window=5).std()
    data['STD10'] = data['close'].rolling(window=10).std()
    data['STD20'] = data['close'].rolling(window=20).std()
    
    # Daily Returns
    data['Return'] = data['close'].pct_change()
    
    # Volume Change (if volume data exists)
    if 'volume' in data.columns:
        data['Volume_Change'] = data['volume'].pct_change()
    else:
        data['Volume_Change'] = 0.0  # Placeholder
    
    # RSI Calculation
    data['RSI'] = compute_rsi(data['close'], window=14)
    
    # MACD Calculation
    data['MACD'] = compute_macd(data['close'])
    
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
df = add_technical_indicators(df)
print("\nData after Feature Engineering:")
print(df.head())

# %%
# Selecting features and target
features = ['close', 'positive', 'negative', 'MA5', 'MA10', 'MA20',
            'STD5', 'STD10', 'STD20', 'Return', 'Volume_Change', 'RSI', 'MACD']
target = 'close'

# Initialize scalers
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Fit and transform the features
scaled_features = scaler_features.fit_transform(df[features])
scaled_target = scaler_target.fit_transform(df[[target]])

# Define window size (number of past time steps to consider for prediction)
window_size = 60

# Create sequences
X = []
y = []

for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i-window_size:i])
    y.append(scaled_target[i])

X = np.array(X)
y = np.array(y)

print(f'\nX shape: {X.shape}, y shape: {y.shape}')

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

print(f'\nTrain shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}')

# %%
# Define the custom Attention Layer
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
        # Compute attention scores
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute context vector
        context = inputs * alpha
        context = K.sum(context, axis=1)
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# %%
# Define the model-building function with Keras Tuner
def build_model(hp):
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
    max_trials=20,  # Adjust based on resources
    executions_per_trial=1,
    directory='my_dir',
    project_name='btc_price_prediction'
)

# Display search space summary
tuner.search_space_summary()

# %%
# Define Early Stopping to prevent overfitting
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
