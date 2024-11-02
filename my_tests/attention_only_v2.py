# %%
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

# %%
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Activation, Multiply, Permute, Flatten, RepeatVector
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

# %% [markdown]
# ## Data Loading and Preprocessing

# %%
# Downloading the dataset using yfinance for up-to-date data
df = yf.download('GOOG', period='2y', interval='1d')
df.reset_index(inplace=True)
df.head()

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
    data = data.dropna()
    return data

df = add_indicators(df)
df.head()

# %%
# Selecting features and target
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'MA5', 'MA10', 'MA20', 'STD5', 'STD10', 'STD20', 'Return', 'Volume_Change']
target = 'Close'

# Scaling the data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])
scaled_target = scaler.fit_transform(df[[target]])

# %% [markdown]
# ## Splitting Data into Train, Validation, and Test Sets

# %%
test_size = 30
validation_size = 30
train_size = len(df) - test_size - validation_size

X = []
y = []
window_size = 60

for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i-window_size:i])
    y.append(scaled_target[i])

X = np.array(X)
y = np.array(y)

X_train = X[:train_size - window_size]
y_train = y[:train_size - window_size]

X_val = X[train_size - window_size:train_size - window_size + validation_size]
y_val = y[train_size - window_size:train_size - window_size + validation_size]

X_test = X[train_size - window_size + validation_size:]
y_test = y[train_size - window_size + validation_size:]

print(f'Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}')

# %% [markdown]
# ## Building the Model with Hyperparameter Tuning

# %%
def build_model(hp):
    lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')

    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(units=lstm_units, return_sequences=True)(inputs)
    x = Dropout(rate=dropout_rate)(x)

    # Attention Mechanism
    attention = Dense(1, activation='tanh')(x)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(lstm_units)(attention)
    attention = Permute([2, 1])(attention)
    attention_output = Multiply()([x, attention])
    x = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=1))(attention_output)
    
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(rate=dropout_rate)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model

# %%
import keras_tuner as kt

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='my_dir',
    project_name='stock_price_prediction'
)

tuner.search_space_summary()

# %%
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(X_train, y_train,
             epochs=50,
             batch_size=32,
             validation_data=(X_val, y_val),
             callbacks=[early_stopping])

# %%
tuner.results_summary()

# %%
best_model = tuner.get_best_models(num_models=1)[0]

# %% [markdown]
# ## Evaluating the Model

# %%
# Predicting on the test set
y_pred = best_model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], scaled_features.shape[1]-1)), y_pred), axis=1))[:, -1]
y_test_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], scaled_features.shape[1]-1)), y_test), axis=1))[:, -1]

# Calculating evaluation metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f'Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}')

# %% [markdown]
# ## Visualizing the Results

# %%
# Applying Exponential Moving Average for smoothing
def exponential_moving_average(signal, alpha):
    return signal.ewm(alpha=alpha).mean()

predictions = pd.DataFrame(y_pred_rescaled, columns=['Predicted'])
predictions_smoothed = exponential_moving_average(predictions['Predicted'], alpha=0.3)

plt.figure(figsize=(15, 5))
plt.plot(df['Date'].iloc[-len(y_test_rescaled):], y_test_rescaled, label='True Prices', color='black')
plt.plot(df['Date'].iloc[-len(y_test_rescaled):], predictions_smoothed, label='Predicted Prices', color='red')
plt.title('Stock Price Prediction with LSTM and Attention Mechanism')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# %% [markdown]
# ## Conclusion

# %%
print('The model has been improved by incorporating additional features, using hyperparameter tuning, and combining LSTM with an attention mechanism. The evaluation metrics indicate the model\'s performance on the test set.')
