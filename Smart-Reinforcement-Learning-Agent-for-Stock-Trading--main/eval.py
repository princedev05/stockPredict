

import os
import glob
import pandas as pd
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import ta
from environment import BitcoinTradingEnv

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = "./saved_models"
TEST_CSV   = "past_data.csv"
WINDOW_SIZE = 48
FEE        = 0.001
DEVICE     = "cpu"  # inference: cpu is enough

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_and_add_indicators(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.sort_index()
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df = df[["open", "high", "low", "close", "volume"]]

    ind = pd.DataFrame(index=df.index)
    ind["rsi"]          = ta.momentum.RSIIndicator(df["close"]).rsi()
    macd               = ta.trend.MACD(df["close"])
    ind["macd"]         = macd.macd()
    ind["macd_signal"]  = macd.macd_signal()
    ind["sma20"]        = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()

    df = df.merge(ind, left_index=True, right_index=True, how="left")
    return df.ffill().bfill()

# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ──────────────────────────────────────────────────────────────────────────────
def make_env(df: pd.DataFrame):
    env = BitcoinTradingEnv(df=df, window_size=WINDOW_SIZE, fee=FEE)
    return DummyVecEnv([lambda: env])

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_path: str, df: pd.DataFrame) -> list:
    env = make_env(df)
    model = PPO.load(model_path, env=env, device=DEVICE)

    obs = env.reset()
    net_worths = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        net_worth = env.envs[0].net_worth
        net_worths.append(net_worth)

    return net_worths

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # 1) Load test data
    test_df = load_and_add_indicators(TEST_CSV)

    # 2) Find all models
    model_paths = sorted(
        glob.glob(os.path.join(MODEL_DIR, "*.zip")),
        key=lambda x: (
            int(''.join(filter(str.isdigit, os.path.basename(x)))) 
            if any(c.isdigit() for c in os.path.basename(x)) else float('inf')
        )
    )

    if not model_paths:
        print(f"No models found in {MODEL_DIR}")
        return

    print(f"Found {len(model_paths)} models.")

    # 3) Evaluate each model
    results = {}

    for path in model_paths:
        model_name = os.path.splitext(os.path.basename(path))[0]
        print(f"Evaluating: {model_name}")
        net_worths = evaluate_model(path, test_df)
        results[model_name] = net_worths

    # 4) Plot results
    plt.figure(figsize=(14, 7))
    for model_name, net_worths in results.items():
        plt.plot(net_worths, label=model_name)

    plt.xlabel("Time Step")
    plt.ylabel("Net Worth")
    plt.title("Performance of Models on Test Set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()



