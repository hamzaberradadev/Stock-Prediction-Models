# rl_trading_agent.py

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
import gym
from gym import spaces
import random
model_dir = './model'
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom Trading Environment
class TradingEnv(gym.Env):
    """A custom trading environment for OpenAI Gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()

        # Historical data
        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data) - 1
        self.current_step = 0

        # Wallet
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.crypto_held = 0

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: [Predicted_Price, Crypto_Held, Balance]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self):
        self.balance = 10000
        self.net_worth = 10000
        self.crypto_held = 0
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.data.loc[self.current_step, 'Predicted_Price'],
            self.crypto_held,
            self.balance
        ])
        return obs

    def step(self, action):
        self.current_step += 1

        done = self.current_step >= self.n_steps

        current_price = self.data.loc[self.current_step, 'Close']
        predicted_price = self.data.loc[self.current_step, 'Predicted_Price']

        # Take action
        if action == 0:  # Hold
            pass
        elif action == 1:  # Buy
            max_crypto_can_buy = self.balance / current_price
            self.crypto_held += max_crypto_can_buy
            self.balance -= max_crypto_can_buy * current_price
        elif action == 2:  # Sell
            self.balance += self.crypto_held * current_price
            self.crypto_held = 0

        # Update net worth
        self.net_worth = self.balance + self.crypto_held * current_price

        # Calculate reward (profit or loss)
        reward = self.net_worth - 10000  # Initial net worth

        # Optional: Implement more complex reward function
        # e.g., reward = (self.net_worth - prev_net_worth)

        obs = self._next_observation()

        return obs, reward, done, {}

    def render(self, mode='human'):
        profit = self.net_worth - 10000
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Crypto Held: {self.crypto_held}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Profit: {profit}')

# RL Agent
class RLTradingAgent:
    def __init__(self, memory_file='memory.pkl'):
        self.env = None
        self.model = None
        self.memory = []
        self.memory_file = memory_file

        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.episodes = 50

        # Attempt to load existing memory
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Loaded existing memory with {len(self.memory)} experiences.")
        else:
            self.memory = []
            print("Initialized new memory buffer.")

    def learn(self, start_date, end_date):
        """Initial training of the RL agent."""
        # Load and prepare data
        data = self._prepare_data(start_date, end_date)

        # Create environment
        self.env = TradingEnv(data)

        # Define the RL model (e.g., DQN)
        self.model = self._build_model()

        # Training loop
        for e in range(1, self.episodes + 1):
            state = self.env.reset()
            state = np.reshape(state, [1, 3])
            total_reward = 0

            for time in range(self.env.n_steps):
                if np.random.rand() <= self.epsilon:
                    action = random.randrange(self.env.action_space.n)
                else:
                    act_values = self.model.predict(state, verbose=0)
                    action = np.argmax(act_values[0])

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, 3])

                self.memory.append((state, action, reward, next_state, done))

                state = next_state

                if done:
                    print(f"Episode: {e}/{self.episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                    break

                if len(self.memory) > self.batch_size:
                    minibatch = random.sample(self.memory, self.batch_size)
                    self._replay(minibatch)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        # Save the trained model and memory
        self.model.save('trading_agent.h5')
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)
        print("Training completed and model saved.")

    def continue_learning(self, additional_episodes=50, start_date=None, end_date=None):
        """
        Continue training the RL agent with additional episodes.
        Optionally, train on new data by specifying start_date and end_date.
        """
        # Load the existing model
        if self.model is None and os.path.exists('trading_agent.h5'):
            self.model = load_model('trading_agent.h5', custom_objects={'AttentionLayer': AttentionLayer})
            print("Loaded existing model from 'trading_agent.h5'.")
        else:
            if self.model is None:
                raise Exception("No existing model found. Please train the model first using the learn() method.")

        # If new data is provided, prepare it
        if start_date and end_date:
            new_data = self._prepare_data(start_date, end_date)
            if self.env is not None:
                # Append new data to existing environment's data
                self.env.data = pd.concat([self.env.data, new_data], ignore_index=True)
                self.env.n_steps = len(self.env.data) - 1
                print(f"Appended new data. Total steps in environment: {self.env.n_steps}")
            else:
                # Create a new environment with the new data
                self.env = TradingEnv(new_data)
                print(f"Created new environment with data from {start_date} to {end_date}.")
        elif self.env is None:
            raise ValueError("No environment available. Provide start_date and end_date to prepare new data.")

        # Continue training loop
        for e in range(1, additional_episodes + 1):
            state = self.env.reset()
            state = np.reshape(state, [1, 3])
            total_reward = 0

            for time in range(self.env.n_steps):
                if np.random.rand() <= self.epsilon:
                    action = random.randrange(self.env.action_space.n)
                else:
                    act_values = self.model.predict(state, verbose=0)
                    action = np.argmax(act_values[0])

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, 3])

                self.memory.append((state, action, reward, next_state, done))

                state = next_state

                if done:
                    print(f"Additional Episode: {e}/{additional_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")
                    break

                if len(self.memory) > self.batch_size:
                    minibatch = random.sample(self.memory, self.batch_size)
                    self._replay(minibatch)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        # Save the updated model and memory
        self.model.save('trading_agent.h5')
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)
        print("Continued training completed and model saved.")

    def work(self, current_state):
        """Use the trained agent to make a trading decision based on the current state."""
        # Load the trained model if not already loaded
        if self.model is None:
            if os.path.exists('trading_agent.h5'):
                self.model = tf.keras.models.load_model('trading_agent.h5', custom_objects={'AttentionLayer': AttentionLayer})
                print("Loaded trained model from 'trading_agent.h5'.")
            else:
                raise Exception("No trained model found. Please train the model first.")

        state = np.reshape(current_state, [1, 3])
        act_values = self.model.predict(state, verbose=0)
        action = np.argmax(act_values[0])
        return action  # 0: Hold, 1: Buy, 2: Sell

    def _prepare_data(self, start_date, end_date):
        """Prepare data by fetching predictions and merging with actual prices."""
        # Calculate the number of days between start and end dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        num_days = (end_dt - start_dt).days

        # Get predictions using the predict_future_prices function
        last_actual_prices, predictions = predict_future_prices(start_date, num_days)

        # Combine actual prices and predictions
        data = pd.merge(last_actual_prices, predictions, on='Date', how='inner')
        data = data.rename(columns={'Close': 'Actual_Close'})
        return data

    def _build_model(self):
        """Builds a simple Deep Q-Network."""
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=3, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
        print("Built new DQN model.")
        return model

    def _replay(self, minibatch):
        """Trains the model on a minibatch of experiences."""
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def save_memory(self):
        """Saves the experience replay buffer to a file."""
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)
        print(f"Memory saved to {self.memory_file}.")

    def load_memory(self):
        """Loads the experience replay buffer from a file."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Memory loaded from {self.memory_file}.")
        else:
            print("No memory file found. Starting with an empty memory buffer.")


# Required functions from predict.py (Assuming they are available)

def predict_future_prices(start_date, num_points):
    # Load the model directory
    

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

    # Get the last actual prices used for prediction
    last_actual_prices = merged_df[['Date', 'Close']].iloc[-(num_points + num_points):-num_points].reset_index(drop=True)

    return last_actual_prices, results

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


# Main function to run the agent
if __name__ == '__main__':
    # Define the initial training time frame
    initial_start_date = '2020-01-01'
    initial_end_date = '2023-01-01'

    # Initialize the RL trading agent
    agent = RLTradingAgent()

    # Train the agent initially
    agent.learn(start_date=initial_start_date, end_date=initial_end_date)

    # Example of continuing learning with additional data
    additional_start_date = '2023-01-02'
    additional_end_date = '2024-01-01'
    agent.continue_learning(additional_episodes=20, start_date=additional_start_date, end_date=additional_end_date)

    # Example of using the agent to make a trading decision
    # Let's assume current_state is obtained from real-time data
    # For demonstration, we'll use a sample state
    current_state = np.array([50000, 0, 10000])  # Predicted_Price, Crypto_Held, Balance
    action = agent.work(current_state)
    action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    print(f'The agent recommends to: {action_dict[action]}')
