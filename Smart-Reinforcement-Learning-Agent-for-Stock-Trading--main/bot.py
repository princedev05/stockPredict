import os
import random
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment import BitcoinTradingEnv

# ────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ────────────────────────────────────────────────────────────────

SEED = 42
NUM_ENVS = 8
LOG_DIR = "./logs"
EVAL_LOG_DIR = os.path.join(LOG_DIR, "eval")
MODEL_DIR = "./saved_models"
for d in (LOG_DIR, EVAL_LOG_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)

logger = setup_logger()
setup_seed(SEED)

# ────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ────────────────────────────────────────────────────────────────

def load_and_add_indicators(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df = df[["open","high","low","close","volume"]].sort_index()
    import ta  # noqa: E402
    ind = pd.DataFrame(index=df.index)
    ind["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    macd = ta.trend.MACD(df["close"])
    ind["macd"] = macd.macd()
    ind["macd_signal"] = macd.macd_signal()
    ind["sma20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    return df.join(ind).ffill().bfill()

# ────────────────────────────────────────────────────────────────
# CALLBACKS
# ────────────────────────────────────────────────────────────────

class NetWorthLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        if dones[0]:
            env = self.training_env.envs[0]
            base = env
            while hasattr(base, 'env'):
                base = base.env
            self.logger.record("custom/net_worth", float(base.net_worth))
        return True

class EpisodeCounter(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.n_episodes = 0
    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.n_episodes += 1
        return True

# ────────────────────────────────────────────────────────────────
# ENV FACTORY
# ────────────────────────────────────────────────────────────────

def make_env(df: pd.DataFrame, log_dir: str):
    def _init():
        env = BitcoinTradingEnv(df=df, window_size=48, fee=0.001)
        env = Monitor(env, log_dir)
        return env
    return _init

# ────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ────────────────────────────────────────────────────────────────

def main():
    logger.info("Loading and preprocessing train/test datasets")
    train_df = load_and_add_indicators("train.csv")
    test_df  = load_and_add_indicators("test.csv")

    # Vectorized + normalized environments
    train_env = DummyVecEnv([make_env(train_df, LOG_DIR) for _ in range(NUM_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = DummyVecEnv([make_env(test_df, EVAL_LOG_DIR)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    train_env.seed(SEED)
    eval_env.seed(SEED)

    logger.info("Configuring callbacks")
    counter_cb    = EpisodeCounter()
    networth_cb   = NetWorthLogger()
    stop_cb       = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    eval_cb       = EvalCallback(
        eval_env,
        callback_on_new_best=stop_cb,
        best_model_save_path=MODEL_DIR,
        log_path=EVAL_LOG_DIR,
        eval_freq=50_000,         # evaluate every 50k steps
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=250_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_btc",
        verbose=1
    )
    callbacks = [counter_cb, networth_cb, eval_cb, checkpoint_cb]

    logger.info("Initializing PPO with stronger regularization")
    policy_kwargs = {
        "net_arch": [64, 64],                    # smaller network
        "activation_fn": torch.nn.LeakyReLU,
        "optimizer_class": torch.optim.AdamW,
        "optimizer_kwargs": {"weight_decay": 1e-3},  # stronger L2
    }
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,       # lower learning rate
        n_steps=2048,
        batch_size=128,
        n_epochs=10,              # fewer epochs per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,           # tighter clipping
        ent_coef=0.1,             # higher entropy bonus
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,           # stop early if KL >1%
        tensorboard_log=LOG_DIR,
        verbose=1,
        seed=SEED,
        device="cpu",
        policy_kwargs=policy_kwargs
    )

    # Train original 5M steps in 1M chunks
    TOTAL_STEPS = 5_000_000
    CHUNK = 1_000_000
    n_chunks = TOTAL_STEPS // CHUNK

    logger.info(f"Training for {TOTAL_STEPS:,} timesteps")
    for i in range(1, n_chunks + 1):
        logger.info(f"Chunk {i}/{n_chunks}")
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=False, callback=callbacks)
        ckpt = os.path.join(MODEL_DIR, f"ppo_btc_{i}M")
        model.save(ckpt)
        logger.info(f"Saved checkpoint: {ckpt}")

    # Extra 50k timesteps for fine-tuning
    EXTRA = 50_000
    logger.info(f"Fine-tuning for an additional {EXTRA:,} timesteps")
    model.learn(total_timesteps=EXTRA, reset_num_timesteps=False, callback=callbacks)

    final_path = os.path.join(MODEL_DIR, "ppo_btc_final")
    model.save(final_path)
    logger.info(f"Done! Episodes seen: {counter_cb.n_episodes}. Final model: {final_path}")

if __name__ == "__main__":
    main()





