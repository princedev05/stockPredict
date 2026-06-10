# Smart Reinforcement-Learning Agent for Stock Trading

A Python-based deep reinforcement learning (RL) framework for automated stock trading. This project implements a custom OpenAI Gym environment, data processing pipeline, training scripts for RL agents, evaluation utilities, and a simple application interface using streamlit to simulate trading on historical data.

---

---

## Features

- **Custom Gym Environment** (`environment.py`) modeling stock trading as an MDP  
- **Data pipeline** (`data.py`) for loading, splitting, and scaling historical price data  
- **Training script** (`bot.py`) implementing deep RL algorithms (e.g., DQN, DDPG, PPO)  
- **Evaluation script** (`eval.py`) for backtesting and performance metrics  
- **Interactive app** (`app.py`) to simulate trades and visualize results  
- **Logging** of training metrics (in `logs/`) and saving of model checkpoints (`saved_models/`)  

---

## Prerequisites

- Python **3.7+**  
- A terminal / command-line environment  
- (Optional) GPU support for faster training via [PyTorch](https://pytorch.org)  

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Astolsko/Smart-Reinforcement-Learning-Agent-for-Stock-Trading-.git
   cd Smart-Reinforcement-Learning-Agent-for-Stock-Trading-

