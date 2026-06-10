import os
import glob
import time
import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import yfinance as yf
import ta
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import BitcoinTradingEnv

# ──────────────────────────────────────────────────────────────────────────────
# WORKAROUNDS & ENV SETUP
# ──────────────────────────────────────────────────────────────────────────────
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["STREAMLIT_WATCHER_DISABLE_MODULE_PATHS"]  = "1"
torch.classes.__path__ = []

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR   = "saved_models"
FEE         = 0.001
DEVICE      = "cpu"
WINDOW_SIZE = 48  # warm-up bars
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bitcoin RL Bot Dashboard", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# DATA & INDICATOR FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def download_ohlcv(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    pre = start - datetime.timedelta(days=2)
    post= end   + datetime.timedelta(days=1)
    df = yf.download(
        "BTC-USD",
        start=pre.strftime("%Y-%m-%d"),
        end=post.strftime("%Y-%m-%d"),
        interval="1h",
        progress=False,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    df.columns    = df.columns.str.lower()
    return df[["open", "high", "low", "close", "volume"]]

@st.cache_data(show_spinner=False)
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    ind = pd.DataFrame(index=df.index)
    ind["rsi"]          = ta.momentum.RSIIndicator(df["close"]).rsi()
    macd                = ta.trend.MACD(df["close"])
    ind["macd"]         = macd.macd()
    ind["macd_signal"]  = macd.macd_signal()
    ind["sma20"]        = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    return df.join(ind).ffill().bfill()

def load_data(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    raw = download_ohlcv(start, end)
    return raw if raw.empty else add_indicators(raw)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LISTING & BACKTEST
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def list_models() -> List[str]:
    paths = glob.glob(os.path.join(MODEL_DIR, "*.zip"))
    def key(p: str) -> float:
        name   = os.path.splitext(os.path.basename(p))[0]
        digits = "".join(filter(str.isdigit, name))
        return float(digits) if digits else float("inf")
    return sorted(paths, key=key)

def backtest(model_path: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Load a PPO model with a single-env VecEnv and run it over df.
    Returns traces plus buy/sell markers.
    """
    env   = DummyVecEnv([lambda: BitcoinTradingEnv(df=df, window_size=WINDOW_SIZE, fee=FEE)])
    model = PPO.load(model_path, env=env, device=DEVICE)

    obs = env.reset()
    net, price = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        base = env.envs[0]
        net.append(base.net_worth)
        price.append(base._get_price())

    buys, sells = [], []
    for t in base.trades:
        ts = df.index[t["step"]]
        if t["type"] == "buy":
            buys.append((ts, t["price"]))
        else:
            sells.append((ts, t["price"]))

    times = df.index[WINDOW_SIZE : WINDOW_SIZE + len(net)]
    return {
        "time":  times,
        "price": price,
        "net":   net,
        "sma20": df["sma20"].loc[times],
        "rsi":   df["rsi"].loc[times],
        "buys":  buys,
        "sells": sells,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")

today         = datetime.date.today()
default_start = today - datetime.timedelta(days=7)
default_end   = today - datetime.timedelta(days=1)
date_range    = st.sidebar.date_input(
    "Backtest range",
    value=(default_start, default_end),
    min_value=datetime.date(2010,1,1),
    max_value=today
)
if not isinstance(date_range, tuple) or len(date_range) != 2:
    st.sidebar.error("Select start and end dates.")
    st.stop()
start_date, end_date = date_range
if start_date > end_date:
    st.sidebar.error("Start must be ≤ End.")
    st.stop()

models = list_models()
if not models:
    st.sidebar.error("No models found.")
    st.stop()
choice  = st.sidebar.selectbox("Select model", models)

live     = st.sidebar.checkbox("🔴 Animate live", value=False)
delay    = st.sidebar.slider("Delay (s/bar)", 0.01, 1.0, 0.1, 0.01)
show_raw = st.sidebar.checkbox("Show raw data", value=False)
run      = st.sidebar.button("▶️ Run Backtest")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Loading data…"):
        df = load_data(start_date, end_date)
    if df.empty or len(df) < WINDOW_SIZE + 1:
        st.error(f"Not enough data ({len(df)} bars).")
        st.stop()

    st.markdown(f"**Data:** {df.index.min()} → {df.index.max()} ({len(df)} bars)")
    if show_raw:
        st.dataframe(df)

    with st.spinner("Backtesting…"):
        result = backtest(choice, df)

    times  = result["time"]
    price  = result["price"]
    sma20  = result["sma20"]
    netw   = result["net"]
    rsi    = result["rsi"]
    buys   = result["buys"]
    sells  = result["sells"]

    ph_main = st.empty()
    ph_rsi  = st.empty()

    def plot_frame(idx: int):
        t_i = times[: idx+1]
        p_i = price[: idx+1]
        s_i = sma20.iloc[: idx+1]
        nw_i= netw[: idx+1]
        r_i = rsi.iloc[: idx+1]
        b_i = [(ts,pr) for ts,pr in buys  if ts <= t_i[-1]]
        s_i2= [(ts,pr) for ts,pr in sells if ts <= t_i[-1]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_i, y=p_i, name="Price"))
        fig.add_trace(go.Scatter(x=t_i, y=s_i, name="SMA20", line=dict(dash="dash")))
        if b_i:
            bt,bp = zip(*b_i)
            fig.add_trace(go.Scatter(x=bt, y=bp, mode="markers",
                                     marker_symbol="triangle-up", marker_color="green",
                                     marker_size=12, name="BUY"))
        if s_i2:
            st2,sp = zip(*s_i2)
            fig.add_trace(go.Scatter(x=st2, y=sp, mode="markers",
                                     marker_symbol="triangle-down", marker_color="red",
                                     marker_size=12, name="SELL"))
        fig.add_trace(go.Scatter(x=t_i, y=nw_i, name="Net Worth",
                                 yaxis="y2", line=dict(dash="dot")))
        fig.update_layout(
            xaxis_title="Time",
            yaxis=dict(title="Price/SMA/Signals"),
            yaxis2=dict(title="Net Worth", overlaying="y", side="right"),
            margin=dict(l=50,r=50,t=40,b=40), height=500
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t_i, y=r_i, name="RSI", line=dict(color="purple")))
        fig2.update_layout(xaxis_title="Time", yaxis_title="RSI",
                           margin=dict(l=50,r=50,t=40,b=40), height=250)

        ph_main.plotly_chart(fig, use_container_width=True)
        ph_rsi.plotly_chart(fig2, use_container_width=True)

    if not live:
        plot_frame(len(times)-1)
    else:
        for i in range(len(times)):
            plot_frame(i)
            time.sleep(delay)






