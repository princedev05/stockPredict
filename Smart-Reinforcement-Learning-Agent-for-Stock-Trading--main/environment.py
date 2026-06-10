import gymnasium as gym
import numpy as np
import pandas as pd
import collections

class BitcoinTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 24,
        fee: float = 0.001,
        initial_balance: float = 10_000,
        technical_indicators: pd.DataFrame = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.fee = fee
        self.initial_balance = initial_balance

        # For reward calculations
        self.returns_buf = collections.deque(maxlen=100)
        self.max_net_worth = initial_balance

        # --- Data preprocessing as before ---
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df = df.sort_index()
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        if technical_indicators is not None:
            ti = technical_indicators.copy()
            if not isinstance(ti.index, pd.DatetimeIndex):
                ti.index = pd.to_datetime(ti.index)
            if ti.index.tz is not None:
                ti.index = ti.index.tz_convert(None)
            df = df.merge(ti, left_index=True, right_index=True, how="left")
        df = df.ffill().bfill().apply(pd.to_numeric, errors="raise")

        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.num_features = self.df.shape[1]

        # Action & observation spaces
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32,
        )

        self.reset()

    def seed(self, seed=None):
        self.reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = float(self.initial_balance)
        self.crypto_held = 0.0
        self.net_worth = float(self.initial_balance)
        self.previous_net_worth = float(self.initial_balance)
        self.returns_buf.clear()
        self.max_net_worth = float(self.initial_balance)
        self.trades = []
        return self._get_observation(), {}

    def _get_observation(self):
        window = self.df.iloc[
            self.current_step - self.window_size : self.current_step
        ].values.astype(np.float32)
        mean = window.mean(axis=0, keepdims=True)
        std  = window.std(axis=0, keepdims=True) + 1e-8
        return ((window - mean) / std).astype(np.float32)

    def _get_price(self) -> float:
        return float(self.df.iloc[self.current_step]["close"])

    def step(self, action: int):
        price = self._get_price()
        prev_nw = self.net_worth

        # --- Execute trade ---
        did_trade = False
        if action == 1 and self.balance > 0:
            spend = self.balance * (1 - self.fee)
            units = spend / price
            self.crypto_held += units
            self.balance -= units * price
            did_trade = True
        elif action == 2 and self.crypto_held > 0:
            proceeds = self.crypto_held * price * (1 - self.fee)
            self.balance += proceeds
            self.crypto_held = 0.0
            did_trade = True

        # --- Update net worth ---
        self.net_worth = self.balance + self.crypto_held * price

        # --- 1) Log-return pnl for scale invariance ---
        pnl = np.log(self.net_worth + 1e-8) - np.log(prev_nw + 1e-8)

        # --- 2) Rolling Sharpe-like ratio ---
        self.returns_buf.append(pnl)
        mean_ret = np.mean(self.returns_buf) if self.returns_buf else 0.0
        std_ret  = np.std(self.returns_buf) + 1e-8
        sharpe   = mean_ret / std_ret

        # --- 3) Drawdown penalty ---
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        drawdown = (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-8)

        # --- 4) Turnover penalty (approximate transaction cost) ---
        turnover = abs(self.net_worth - prev_nw)
        turnover_penalty = self.fee * turnover

        # --- Compose final reward (all components O(1)) ---
        reward = (
            10.0 * pnl          # directional gain
          +  2.0 * sharpe       # consistency bonus
          -  5.0 * drawdown     # drawdown control
          -  1.0 * turnover_penalty
        )

        # --- Advance step ---
        self.current_step += 1
        done = self.current_step >= self.max_steps
        obs = (
            self._get_observation()
            if not done
            else np.zeros_like(self.observation_space.sample())
        )
        info = {
            "net_worth": self.net_worth,
            "pnl": pnl,
            "sharpe": sharpe,
            "drawdown": drawdown
        }

        return obs, reward, done, False, info

    def render(self, mode="human"):
        price = self._get_price()
        print(
            f"Step {self.current_step} | Price: {price:.2f} | "
            f"Balance: {self.balance:.2f} | Holdings: {self.crypto_held:.6f} | "
            f"Net Worth: {self.net_worth:.2f}"
        )
        if self.trades:
            last = self.trades[-1]
            print(f"  Last trade: {last}")


