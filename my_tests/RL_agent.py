# rl_trading_agent.py

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import gym
from gym import spaces
import random
from predicts import predict_future_prices  # Ensure this is correctly implemented

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom Trading Environment
class TradingEnv(gym.Env):
    """A comprehensive custom trading environment for OpenAI Gym."""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000, annual_trading_days=252, 
                 min_acceptable_return=0.05, max_inaction_steps=5):
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
        self.annual_trading_days = annual_trading_days
        self.min_acceptable_return = min_acceptable_return  # Minimum acceptable annual return
        self.max_inaction_steps = max_inaction_steps  # Max allowed consecutive inaction steps
        self.inaction_steps = 0  # Counter for consecutive inaction steps

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
        self.annual_returns = []
        self.sharpe_ratios = []
        self.trade_count = 0  # Total number of trades in the current episode

    def reset(self):
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.crypto_held = 0
        self.max_net_worth = self.initial_balance
        self.current_step = 50  # Reset to the starting step
        self.net_worths = [self.net_worth]
        self.annual_returns = []
        self.sharpe_ratios = []
        self.trade_count = 0
        self.inaction_steps = 0  # Reset inaction steps

        return self._next_observation()

    def _next_observation(self):
        """Get the data for the current step and assemble the observation."""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1  # Prevent index out of bounds

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

        # Define a threshold to consider an action as significant
        action_threshold = 0.01

        # Check if the action is significant
        if abs(action_type) < action_threshold:
            self.inaction_steps += 1
            action_significant = False
        else:
            self.inaction_steps = 0  # Reset inaction steps when action is taken
            action_significant = True

        # Calculate transaction cost
        transaction_cost = 0

        if action_type > action_threshold:  # Buy
            amount_to_invest = self.balance * action_type
            num_shares = amount_to_invest / current_price
            transaction_cost = amount_to_invest * self.transaction_cost_percent
            total_cost = amount_to_invest + transaction_cost

            if total_cost <= self.balance:
                self.balance -= total_cost
                self.crypto_held += num_shares
                self.trade_count += 1  # Increment trade count

        elif action_type < -action_threshold:  # Sell
            num_shares = self.crypto_held * (-action_type)
            transaction_cost = (num_shares * current_price) * self.transaction_cost_percent
            total_revenue = (num_shares * current_price) - transaction_cost

            if num_shares <= self.crypto_held:
                self.balance += total_revenue
                self.crypto_held -= num_shares
                self.trade_count += 1  # Increment trade count

        # Update net worth
        self.net_worth = self.balance + self.crypto_held * current_price - transaction_cost
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        self.net_worths.append(self.net_worth)

    def _calculate_reward(self):
        """Calculate the reward for the current step."""
        # Calculate current annual return
        current_annual_return = self.get_current_annual_return()
        if len(self.annual_returns) > 0:
            previous_annual_return = self.annual_returns[-1]
            # Reward is the improvement in annual return
            return_improvement = current_annual_return - previous_annual_return
            reward = return_improvement
        else:
            reward = 0
        self.annual_returns.append(current_annual_return)

        # Calculate Sharpe Ratio
        sharpe_ratio = self.calculate_sharpe_ratio()
        self.sharpe_ratios.append(sharpe_ratio)

        # Scale the reward components
        reward *= 100  # Scale return improvement
        sharpe_reward = sharpe_ratio * 10  # Scale Sharpe ratio
        reward += sharpe_reward

        # Penalize large drawdowns
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if drawdown > 0.2:
            reward -= drawdown * 100  # Penalize for drawdown, scaled

        # Penalize inaction
        if self.inaction_steps >= self.max_inaction_steps:
            reward -= 10  # Fixed penalty for prolonged inaction

        # Penalize if annual return is below minimum acceptable return
        if current_annual_return < self.min_acceptable_return:
            reward -= (self.min_acceptable_return - current_annual_return) * 100  # Penalize proportionally

        # Penalize excessive trading to minimize transaction costs
        if self.trade_count > 100:  # Example threshold, adjust as needed
            reward -= (self.trade_count - 100) * 0.1  # Small penalty per trade over threshold

        return reward

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculate the Sharpe Ratio based on annual returns."""
        if len(self.annual_returns) < 2:
            return 0
        returns = np.array(self.annual_returns)
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe_ratio

    def get_current_annual_return(self):
        """Calculate the annualized return up to the current step."""
        total_return = (self.net_worth - self.initial_balance) / self.initial_balance
        num_days = self.current_step if self.current_step > 0 else 1
        annual_return = (1 + total_return) ** (self.annual_trading_days / num_days) - 1
        return annual_return

    def render(self, mode='human', close=False):
        """Render the environment to the screen."""
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Crypto Held: {self.crypto_held:.6f}')
        print(f'Net Worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f}')
        if self.trade_count > 0:
            print(f'Total Trades: {self.trade_count}')

# Ornstein-Uhlenbeck Noise for Exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()
        
    def __call__(self):
        # Generate noise
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        self.x_prev = x
        return x
    
    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

# RL Trading Agent using DDPG
class RLTradingAgent:
    def __init__(self, memory_file='memory.pkl'):
        self.env = None
        self.memory_file = memory_file

        # Hyperparameters
        self.gamma = 0.99          # Discount factor
        self.tau = 0.005           # Soft update parameter
        self.batch_size = 64
        self.episodes = 500        # Increased for better training
        self.memory = []           # Experience replay buffer
        self.memory_capacity = 1000000

        # Exploration noise parameters
        self.std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))

        # Initialize actor and critic networks
        self.actor_model = None
        self.critic_model = None
        self.target_actor = None
        self.target_critic = None

        # Optimizers
        self.critic_optimizer = optimizers.Adam(learning_rate=0.001)
        self.actor_optimizer = optimizers.Adam(learning_rate=0.0001)

        # Try to load existing memory
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Loaded existing memory with {len(self.memory)} experiences.")
        else:
            print("Initialized new memory buffer.")

    def _build_actor(self, state_size, action_size, action_bound):
        """Builds the actor network."""
        inputs = layers.Input(shape=(state_size,))
        out = layers.Dense(128, activation='relu')(inputs)
        out = layers.Dense(128, activation='relu')(out)
        out = layers.Dense(64, activation='relu')(out)
        outputs = layers.Dense(action_size, activation='tanh')(out)
        outputs = outputs * action_bound  # Scale output to action bounds
        model = models.Model(inputs, outputs)
        return model

    def _build_critic(self, state_size, action_size):
        """Builds the critic network."""
        state_input = layers.Input(shape=(state_size))
        action_input = layers.Input(shape=(action_size))
        concat = layers.Concatenate()([state_input, action_input])

        out = layers.Dense(128, activation='relu')(concat)
        out = layers.Dense(128, activation='relu')(out)
        out = layers.Dense(64, activation='relu')(out)
        outputs = layers.Dense(1)(out)
        model = models.Model([state_input, action_input], outputs)
        return model

    def policy(self, state):
        """Returns action for given state as per current policy."""
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.ou_noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise
        # Ensure action is within bounds
        legal_action = np.clip(sampled_actions, -1, 1)
        return [np.squeeze(legal_action)]

    def learn(self, start_date, end_date):
        """Train the RL agent using DDPG algorithm."""
        data = self._prepare_data(start_date, end_date)
        print("Prepared data for training:")
        print(data.head())

        self.env = TradingEnv(data)

        # Initialize actor and critic models
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.shape[0]
        action_bound = 1  # Assuming action space is between -1 and 1

        self.actor_model = self._build_actor(state_size, action_size, action_bound)
        self.critic_model = self._build_critic(state_size, action_size)

        # Initialize target networks
        self.target_actor = self._build_actor(state_size, action_size, action_bound)
        self.target_critic = self._build_critic(state_size, action_size)

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        total_episodes = self.episodes

        for ep in range(total_episodes):
            prev_state = self.env.reset()
            self.ou_noise.reset()  # Reset noise at the start of each episode
            episodic_reward = 0

            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = self.policy(tf_prev_state)
                action = np.array(action)

                state, reward, done, _ = self.env.step(action)
                episodic_reward += reward

                # Store experience
                self.memory.append((prev_state, action, reward, state, done))

                if len(self.memory) > self.memory_capacity:
                    self.memory.pop(0)

                # Update networks
                self.update()

                if done:
                    annual_return = self.env.get_current_annual_return()
                    sharpe = self.env.sharpe_ratios[-1] if self.env.sharpe_ratios else 0
                    print(f"Episode: {ep+1}/{total_episodes}, Total Reward: {episodic_reward:.2f}, "
                          f"Annual Return: {annual_return:.2%}, Sharpe Ratio: {sharpe:.2f}")
                    break

                prev_state = state

        # Save the trained models and memory
        self.actor_model.save('actor_model.keras')
        self.critic_model.save('critic_model.keras')
        with open(self.memory_file, 'wb') as f:
            pickle.dump(self.memory, f)
        print("Training completed and models saved.")

    def update(self):
        """Update actor and critic networks."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch]).reshape(self.batch_size, -1)
        rewards = np.array([experience[2] for experience in minibatch]).reshape(self.batch_size, -1)
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch]).astype(np.float32).reshape(self.batch_size, -1)

        # Convert arrays to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Critic update
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            y = rewards + self.gamma * (1 - dones) * self.target_critic([next_states, target_actions])
            critic_value = self.critic_model([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # Actor update
        with tf.GradientTape() as tape:
            actions_pred = self.actor_model(states)
            critic_value = self.critic_model([states, actions_pred])
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        # Update target networks
        self.update_target(self.target_actor.variables, self.actor_model.variables)
        self.update_target(self.target_critic.variables, self.critic_model.variables)

    def update_target(self, target_weights, weights):
        """Soft update target network parameters."""
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def work(self, current_state):
        """Make a trading decision based on the current state."""
        if self.actor_model is None:
            if os.path.exists('actor_model.keras'):
                self.actor_model = tf.keras.models.load_model('actor_model.keras')
                print("Loaded trained actor model from 'actor_model.keras'.")
            else:
                raise Exception("No trained model found. Please train the model first.")

        state = tf.expand_dims(tf.convert_to_tensor(current_state), 0)
        action = tf.squeeze(self.actor_model(state)).numpy()
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
        data.dropna(inplace=True)  # Ensure no remaining NaNs
        print("Data after filling NaNs and adding indicators:")
        print(data.head())
        return data

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
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)
    return data

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=window - 1, adjust=False).mean()
    ma_down = down.ewm(com=window - 1, adjust=False).mean()
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

    # Example of using the agent to make a trading decision
    # Ensure the current_state has the correct shape and values
    current_state = np.array([
        50000,    # Close
        49500,    # MA10
        49000,    # MA50
        55,       # RSI
        0.5,      # MACD
        0,        # Crypto_Held
        10000,    # Balance
        10000,    # Net Worth
        10000     # Max Net Worth
    ])
    action = agent.work(current_state)
    print(f'The agent recommends an action of: {action:.4f}')
