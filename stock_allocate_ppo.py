"""
    Portfolio optimization using historical returns
    The algorithm relies on Proximal Policy Optimization from stable_baseline3
"""
"""
    Import dependencies
"""
import datetime as dt 
from datetime import timezone
import yfinance as yf
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gym import spaces
import gym
import plotly.graph_objects as go

tickers = ['SPY', '^NSEI', 'NQ=F']  # Stocklist; see previous code
end_date = dt.datetime.now(timezone.utc)
start_date = end_date - dt.timedelta(days=500)

# fetching stock data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
# data = data.Close
returns = data.pct_change().dropna()  # Daily returns of the stocks
#DEBUG:
# print(returns.isnull().values.any())

# Portfolio Environment
class PortfolioEnv(gym.Env):
    metadata = {"render_mode": "human", "render_fps": 4}

    def __init__(self, stock_returns, initial_balance=100000, render_mode="human"):
        super(PortfolioEnv, self).__init__()
        self.stock_returns = stock_returns
        self.initial_balance = initial_balance
        self.n_assets = stock_returns.shape[1]
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets,))
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        # Set render mode
        assert render_mode in self.metadata["render_mode"], f"Unsupported render_mode: {render_mode}"
        self.render_mode = render_mode
        self.reset()
        self.portfolio_values = []
        
    def reset(self):
        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets)  # Equal weights initially
        # self.weights = np.random.rand(self.n_assets)
        # self.weights /= self.weights.sum()
        self.state = self.stock_returns.iloc[self.current_step].values
        self.portfolio_values = [self.cash_balance]
        return self.state

    def step(self, action):
        # Normalize action weights
        if np.sum(action) != 0:
            weights = action / np.sum(action)
        else:
            weights = np.array([1.0 / self.n_assets] * self.n_assets)
        
        # Calculate portfolio return for the current step
        portfolio_return = np.dot(self.stock_returns.iloc[self.current_step], weights)
        #DEBUG:
        # print(portfolio_return)
        # portfolio_return = portfolio_return[~np.isnan(portfolio_return)]
        self.cash_balance *= (1 + portfolio_return)
        # print(self.cash_balance)
        self.portfolio_values.append(self.cash_balance)
        # Proceed to next time step
        self.current_step += 1
        
        done = self.current_step >= len(self.stock_returns) - 1
        
        # Update the state (next time step returns)
        self.state = self.stock_returns.iloc[self.current_step].values
        
        # Reward is the new portfolio value
        reward = self.cash_balance
        
        return self.state, reward, done, {}
    
    def render(self):
        if self.render_mode == "human":
            print(f'Step: {self.current_step}, Portfolio Value: {self.cash_balance:.2f}')

# Creating portfolio environment and wrap it for RL
env = PortfolioEnv(returns)
# env = DummyVecEnv([lambda: env])

# Training the RL agent (PPO)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=20000, progress_bar=True)

# Testing the trained model
obs = env.reset()
for i in range(len(returns)):
    action, _states = model.predict(obs)
    # print(action)
    obs, reward, done, info = env.step(action)
    # DEBUG:
    # print(reward)
    env.render()
    if done:
        break

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(len(env.portfolio_values))),
    y=env.portfolio_values,
    mode='lines',
    name='Portfolio Value'
))

fig.update_layout(
    title="Portfolio Value Over Time",
    xaxis_title="Time Steps",
    yaxis_title="Portfolio Value",
    template="plotly_dark"
)

fig.show()