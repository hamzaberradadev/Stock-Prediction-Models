# rl_trading_agent.py

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import gym
from gym import spaces
import random
from predicts import predict_future_prices

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
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
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        e = tf.keras.backend.squeeze(e, axis=-1)
        alpha = tf.keras.backend.softmax(e)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

import numpy as np
import gym
from gym import spaces

class TradingEnv(gym.Env):
    """A more robust custom trading environment for OpenAI Gym."""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000, annual_trading_days=252):
        super(TradingEnv, self).__init__()

        required_columns = {'Close', 'RSI', 'MACD', 'MA10', 'MA50'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {missing_columns}")

        self.data = data.reset_index(drop=True)

        # Parameters
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.crypto_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.trades = []
        self.annual_trading_days = annual_trading_days

        # Calculate the total number of steps
        self.n_steps = len(self.data) - 1
        self.current_step = 50  # Start after enough data points for indicators

        # Transaction cost parameters
        self.transaction_cost_percent = 0.001  # 0.1% transaction cost per trade

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        # Historical performance tracking
        self.net_worths = [self.net_worth]

    def reset(self):
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.crypto_held = 0
        self.max_net_worth = self.initial_balance
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.trades = []
        self.current_step = 50  # Reset to the starting step
        self.net_worths = [self.net_worth]

        return self._next_observation()

    def _next_observation(self):
        """Get the data for the current step and assemble the observation."""
        current_row = self.data.iloc[self.current_step]

        obs = np.array([
            current_row['Close'],
            current_row['MA10'],
            current_row['MA50'],
            current_row['RSI'],
            current_row['MACD'],
            self.crypto_held,
            self.balance,
            self.net_worth,
            self.max_net_worth
        ], dtype=np.float32)
        return obs

    def step(self, action):
        """Execute one time step within the environment."""
        # Execute the trade
        self._take_action(action)

        self.current_step += 1

        # Check if we have reached the end of the data
        done = self.current_step >= self.n_steps

        # Calculate the reward
        reward = self._calculate_reward()

        # Get the next observation
        obs = self._next_observation()

        return obs, reward, done, {}

    def _take_action(self, action):
        """Execute the trade based on the action."""
        current_price = self.data.loc[self.current_step, 'Close']
        action_type = np.clip(action[0], -1, 1)

        # Calculate transaction cost
        transaction_cost = 0

        if action_type > 0:  # Buy
            amount_to_invest = self.balance * action_type
            num_shares = amount_to_invest / current_price
            transaction_cost = amount_to_invest * self.transaction_cost_percent
            total_cost = amount_to_invest + transaction_cost

            if total_cost <= self.balance:
                self.balance -= total_cost
                self.crypto_held += num_shares
                self.trades.append({'step': self.current_step, 'shares': num_shares, 'type': 'buy'})

        elif action_type < 0:  # Sell
            num_shares = self.crypto_held * (-action_type)
            transaction_cost = (num_shares * current_price) * self.transaction_cost_percent
            total_revenue = (num_shares * current_price) - transaction_cost

            if num_shares <= self.crypto_held:
                self.balance += total_revenue
                self.crypto_held -= num_shares
                self.total_shares_sold += num_shares
                self.total_sales_value += total_revenue
                self.trades.append({'step': self.current_step, 'shares': num_shares, 'type': 'sell'})

        # Update net worth
        self.net_worth = self.balance + self.crypto_held * current_price - transaction_cost
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.net_worths.append(self.net_worth)

    def _calculate_reward(self):
        """Calculate the reward for the current step."""
        # Reward is the percentage change in net worth
        if len(self.net_worths) > 1:
            reward = (self.net_worths[-1] - self.net_worths[-2]) / self.net_worths[-2]
        else:
            reward = 0

        # Penalize large drawdowns
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if drawdown > 0.2:
            reward -= drawdown  # Penalize for drawdown

        return reward

    def render(self, mode='human', close=False):
        """Render the environment to the screen."""
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Crypto Held: {self.crypto_held:.6f}')
        print(f'Net Worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f}')

    def get_annual_return(self):
        """Calculate the annualized return."""
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        num_days = self.current_step
        annual_return = (1 + total_return) ** (self.annual_trading_days / num_days) - 1
        return annual_return

# RL Trading Agent
class RLTradingAgent:
    def __init__(self, memory_file='memory.pkl'):
        self.env = None
        self.model = None
        self.memory = []
        self.memory_file = memory_file

        self.gamma = 0.99
        self.epsilon = 1.0   # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.episodes = 100

        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Loaded existing memory with {len(self.memory)} experiences.")
        else:
            self.memory = []
            print("Initialized new memory buffer.")

    def learn(self, start_date, end_date):
        """Train the RL agent."""
        data = self._prepare_data(start_date, end_date)
        print("Prepared data for training:")
        print(data.head())
        self.env = TradingEnv(data)

        self.model = self._build_model()

        for e in range(1, self.episodes + 1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            total_reward = 0

            for time in range(self.env.n_steps):
                # Continuous action selection
                if np.random.rand() <= self.epsilon:
                    action = np.random.uniform(-1, 1, size=(1,))
                else:
                    action = self.model.predict(state, verbose=0)[0]
                    action = np.clip(action, -1, 1)

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])

                self.memory.append((state, action, reward, next_state, done))

                state = next_state

                if done:
                    annual_return = self.env.get_annual_return()
                    print(f"Episode: {e}/{self.episodes}, Total Reward: {total_reward:.4f}, Annual Return: {annual_return:.2%}, Epsilon: {self.epsilon:.4f}")
                    break

                if len(self.memory) > self.batch_size:
                    minibatch = random.sample(self.memory, self.batch_size)
                    self._replay(minibatch)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        # Save the trained model and memory
        self.model.save('trading_agent.keras')
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)
        print("Training completed and model saved.")

    def work(self, current_state):
        """Make a trading decision based on the current state."""
        if self.model is None:
            if os.path.exists('trading_agent.keras'):
                self.model = tf.keras.models.load_model('trading_agent.keras')
                print("Loaded trained model from 'trading_agent.keras'.")
            else:
                raise Exception("No trained model found. Please train the model first.")

        state = np.reshape(current_state, [1, self.env.observation_space.shape[0]])
        action = self.model.predict(state, verbose=0)[0]
        action = np.clip(action, -1, 1)
        return action  # Continuous action between -1 and 1

    def _prepare_data(self, start_date, end_date):
        """Prepare data by fetching predictions and concatenating with actual prices."""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        num_days = (end_dt - start_dt).days

        # Fetch actual and predicted data
        last_actual_prices, predictions = predict_future_prices(start_date, num_days)
        predictions = predictions.rename(columns={'Predicted_Price': 'Close'})

        # Concatenate historical and predicted data sequentially
        data = pd.concat([last_actual_prices, predictions], ignore_index=True)

        # Ensure required columns are present
        required_columns = {'Close'}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"Data is missing required columns: {missing_columns}")

        # Apply technical indicators
        data = add_technical_indicators(data)

        # Fill missing values
        data = data.bfill().ffill()
        print("Data after filling NaNs and adding indicators:")
        print(data.head())
        return data

    def _build_model(self):
        """Builds a neural network model for continuous action output."""
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='tanh'))  # Output layer with tanh activation for actions between -1 and 1
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
        print("Built neural network model for continuous action space.")
        return model

    def _replay(self, minibatch):
        """Trains the model on a minibatch of experiences."""
        states = np.array([experience[0] for experience in minibatch]).squeeze()
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch]).squeeze()
        dones = np.array([experience[4] for experience in minibatch])

        # Predict Q-values for next states
        target_actions = self.model.predict(next_states, verbose=0)
        target_actions = np.clip(target_actions, -1, 1)

        # Compute targets
        targets = rewards + self.gamma * (1 - dones) * np.squeeze(target_actions)

        # Update the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
def add_technical_indicators(data):
    # Moving Averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Standard Deviations
    data['STD5'] = data['Close'].rolling(window=5).std()
    data['STD10'] = data['Close'].rolling(window=10).std()
    data['STD20'] = data['Close'].rolling(window=20).std()
    
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

# Main function to run the agent
if __name__ == '__main__':
    initial_start_date = '2020-01-01'
    initial_end_date = '2023-01-01'

    agent = RLTradingAgent()
    agent.learn(start_date=initial_start_date, end_date=initial_end_date)

    current_state = np.array([
        50000,    # Close
        49500,    # MA10
        49000,    # MA50
        55,       # RSI
        0.5,      # MACD
        0,        # Crypto_Held
        10000     # Balance
    ])
    action = agent.work(current_state)
    action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
    print(f'The agent recommends to: {action_dict[action]}')
