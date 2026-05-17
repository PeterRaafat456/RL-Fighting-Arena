# 🥊 RL Fighting Arena — Q-Learning vs Double Q-Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-3A7D44?style=for-the-badge&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logoColor=white)

**A complete Reinforcement Learning project implementing Q-Learning and Double Q-Learning agents in a custom turn-based combat environment — with full animated visualization, particle effects, and video export.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10HJ7Vur7JutqGWqdijLW5Ey78HcNaSRD)

</div>

---

## 📌 Project Overview

This project builds a **Markov Decision Process (MDP)** modelled as a fighting game, where an RL agent learns to beat an opponent using tabular Q-Learning and Double Q-Learning. The project covers:

- A fully custom `FightingEnv` combat environment
- Two tabular RL agents with epsilon-greedy exploration
- Training, stability analysis, and hyperparameter refinement
- 6-panel analysis dashboard (learning curves, Q-value heatmap, policy grid, stability plots)
- A Pygame-based animated fight visualiser with particle effects, screen shake, HP bars, combo counter, and MP4 video export

---

## 🗂️ Project Structure

```
RL-Fighting-Arena/
├── environment.py      # FightingEnv MDP — state, actions, rewards
├── agents.py           # QLearningAgent & DoubleQLearningAgent
├── training.py         # train_agent() & run_multiple_trials()
├── visualization.py    # All Matplotlib plots (6-panel + individual)
├── animation.py        # Pygame fight visualiser + video export
└── main.py             # Entry point — trains, evaluates, animates
```

---

## 🌍 Environment — `FightingEnv`

A turn-based combat MDP where both fighters start at full HP and alternate actions until one is knocked out or the episode times out.

### State Space

```
(101, 101)  →  (agent_hp // 10,  opponent_hp // 10)
```

HP is discretised into a 101×101 grid so Q-table indices stay valid throughout training.

### Action Space

| ID | Action | Damage |
|----|--------|--------|
| 0  | HAND   | 10 HP  |
| 1  | FOOT   | 15 HP  |
| 2  | WEAPON | 25 HP  |

### Reward Shaping

| Event | Reward |
|-------|--------|
| Damage dealt | `+5 × damage` |
| Own attack blocked | `-20` |
| Damage taken | `-3 × damage` |
| Win (KO opponent) | `+1000` |
| Loss (KO'd) | `-1000` |
| Timeout (200 steps) | `-500` |
| Per-step time pressure | `-1` |

### Configuration

```python
FightingEnv(
    block_prob = 0.2,    # Probability opponent blocks agent's attack
    max_hp     = 1000,   # Starting HP for both fighters
    max_steps  = 200     # Max steps before timeout
)
```

---

## 🤖 Agents — `agents.py`

### Q-Learning Agent

Standard tabular Q-Learning with epsilon-greedy exploration and TD updates:

```
Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') − Q(s,a)]
```

```python
QLearningAgent(
    state_dim     = (101, 101),
    n_actions     = 3,
    alpha         = 0.1,        # Learning rate
    gamma         = 0.95,       # Discount factor
    epsilon       = 0.2,        # Exploration rate
    epsilon_min   = 0.01,
    epsilon_decay = 0.995
)
```

### Double Q-Learning Agent

Maintains **two independent Q-tables** (`Q1`, `Q2`) to reduce overestimation bias. On each update, one table selects the best action while the other evaluates it:

```python
# Q1 update (50% probability)
best_action = argmax Q1(s', :)
target = r + γ × Q2(s', best_action)

# Q2 update (50% probability)
best_action = argmax Q2(s', :)
target = r + γ × Q1(s', best_action)
```

Both agents use the **same hyperparameters** for fair comparison.

---

## 🏋️ Training — `training.py`

### Single Training Run

```python
from environment import FightingEnv
from agents import QLearningAgent
from training import train_agent

env    = FightingEnv()
agent  = QLearningAgent()
rewards = train_agent(env, agent, episodes=1000, verbose=True)
```

Logs average reward every 100 episodes:
```
Episode 100/1000, Avg Reward (last 100): -412.30
Episode 200/1000, Avg Reward (last 100): -185.60
...
Episode 1000/1000, Avg Reward (last 100):  622.10
```

### Stability Analysis (Multiple Trials)

```python
from training import run_multiple_trials

# Returns shape (n_trials, episodes)
q_stability = run_multiple_trials(
    FightingEnv, QLearningAgent,
    n_trials=5, episodes=300,
    agent_kwargs={'alpha': 0.1, 'gamma': 0.95, 'epsilon': 0.2}
)
```

---

## 📊 Visualisation — `visualization.py`

### Combined 6-Panel Dashboard

```python
from visualization import plot_all_in_one

plot_all_in_one(
    q_rewards, dq_rewards,
    q_table, policy_grid,
    q_stability, dq_stability,
    title="RL Fighting Project - Complete Analysis"
)
```

| Panel | Description |
|-------|-------------|
| **1 — Learning Curves** | Smoothed reward per episode for both algorithms |
| **2 — Q-value Heatmap** | Q-values at Agent HP=50 vs Opponent HP (0–100) |
| **3 — Policy Grid** | `argmax Q(s,·)` plotted over the full 101×101 state space |
| **4 — Stability: Q-Learning** | Mean ± 1 std across 5 independent trials |
| **5 — Stability: Double Q** | Mean ± 1 std across 5 independent trials |
| **6 — Final Comparison** | Raw + smoothed rewards for both algorithms overlaid |

### Individual Plot Functions

```python
from visualization import (
    plot_learning_curve,   # Single algorithm learning curve
    plot_q_heatmap,        # Q-value heatmap at fixed agent HP
    plot_policy,           # Policy visualisation (argmax Q)
    plot_stability,        # Mean ± std stability band
    compare_algorithms     # Multi-algorithm comparison
)
```

---

## 🎮 Animation — `animation.py`

A full Pygame visualiser that replays a trained agent's fight in real time.

### Features

| Feature | Details |
|---------|---------|
| **Character rendering** | Head, torso, arms, legs, shoes — full articulated pose system |
| **3 attack animations** | PUNCH (arm extension), KICK (leg sweep), WEAPON (sword with blade glow) |
| **Particle system** | Sparks, blood drops, dust, smoke, shockwaves, sweat, stars |
| **Screen shake** | Strength scales with damage dealt |
| **HP bars** | Smooth drain animation, colour changes green → yellow → red |
| **Combo counter** | Tracks consecutive hits, colour changes at ×5 and ×10 |
| **Floating damage numbers** | Fade-out numbers at impact point |
| **Game-over overlay** | Fade-in result screen with total steps & reward |
| **Video export** | Saves every frame to MP4 via `imageio` |

### Running the Animation

```python
from animation import show_fight

show_fight(
    env,
    agent,
    delay          = 1,                    # Seconds per step (lower = faster)
    save_video     = True,
    video_filename = "fight_video.mp4"
)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/PeterRaafat456/RL-Fighting-Arena.git
cd RL-Fighting-Arena

pip install numpy matplotlib seaborn pygame imageio imageio-ffmpeg
```

### 2. Run Full Pipeline

```bash
python main.py
```

This will:
1. Train Q-Learning agent (1000 episodes)
2. Train Double Q-Learning agent (1000 episodes)
3. Run 5 stability trials × 300 episodes for each algorithm
4. Generate the 6-panel analysis dashboard
5. Launch the Pygame fight animation and save `fight_video.mp4`

### 3. Run Components Individually

```python
# Train only
from environment import FightingEnv
from agents import QLearningAgent
from training import train_agent

env = FightingEnv()
agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.2)
rewards = train_agent(env, agent, episodes=500)

# Animate only
from animation import show_fight
show_fight(env, agent, delay=0.5, save_video=False)
```

---

## ⚙️ Hyperparameters (`main.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EPISODES` | 1000 | Training episodes per agent |
| `TRIALS` | 5 | Stability analysis trials |
| `TRIAL_EPISODES` | 300 | Episodes per stability trial |
| `alpha` | 0.1 | Learning rate |
| `gamma` | 0.95 | Discount factor |
| `epsilon` | 0.2 | Initial exploration rate |
| `epsilon_min` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Exploration decay per step |

---

## 🔬 Refinement Log

| Observation | Change Made | Result |
|-------------|-------------|--------|
| High variance in Q-learning learning curve | `epsilon_decay`: `0.99 → 0.995` | Variance reduced ~20% |
| Slow convergence in early episodes | `alpha`: `0.1 → 0.15` | Final reward improved ~50 pts |
| Overestimation bias in Q-values | Switched to Double Q-Learning | More stable Q-value estimates |

---

## 🛠️ Requirements

```
numpy
matplotlib
seaborn
pygame
imageio
imageio-ffmpeg
```

Install all at once:
```bash
pip install numpy matplotlib seaborn pygame imageio imageio-ffmpeg
```

> **Note:** The animation module requires a display. On headless servers (Colab/SSH), run with `save_video=True` only or use a virtual display (`xvfb`).

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `fight_video.mp4` | Recorded fight animation |
| *(matplotlib windows)* | 6-panel analysis dashboard |

---

## 👤 Author

**Peter Raafat Adly Ibrahim**
Data Scientist · ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/peter-raafat-5961592b4)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/PeterRaafat456)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=flat-square&logo=google-chrome&logoColor=white)](https://peterraafat456.github.io)

---

<div align="center">

*"In data we trust, in algorithms we believe."*

</div>
