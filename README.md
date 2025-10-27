# 🧊 FrozenLake Data-Driven Policy Analysis

This project explores how **data-driven learning** can be applied to the classic **FrozenLake-v1** environment from Gymnasium.  
Instead of using traditional Reinforcement Learning algorithms like Q-learning, this project takes a **supervised learning approach** — using data collected from random actions to predict rewards and guide the agent’s policy.

---

## 🚀 Project Overview

The **FrozenLake-v1** environment is a 4x4 grid where an agent must move from the **start (S)** to the **goal (G)** without falling into **holes (H)**.  
The environment is **stochastic (slippery)** — meaning the agent’s actions don’t always result in the intended movement.

**Objective:**  
Develop a **data-driven policy** that increases the agent’s success rate compared to random actions.

---

## 🧩 Key Steps

### 1. Environment Setup
- Used `gymnasium.make("FrozenLake-v1", map_name="4x4", is_slippery=True)`
- Initialized environment and defined reward structure.

### 2. Data Collection
- Ran **10,000 random episodes**.
- Recorded for each step:
  - `State`
  - `Action`
  - `Reward`
  - `GoalProximity` (Manhattan distance to goal)
  - `TotalReward` (episode success indicator)

### 3. State–Action Importance
- Calculated success probability for each `(state, action)` pair.
- Identified which actions contribute more frequently to success.

### 4. Model Training
- Used a **Random Forest Regressor** to predict `TotalReward` from `(state, action)` pairs.
- Features: State, Action  
- Target: TotalReward  
- R² score ≈ **0.09** (limited by stochastic nature of the environment).

### 5. Model-Guided Policy
- Created a **greedy policy** that selects the action with the highest predicted reward.
- Success rate improved to **~19.6%** (vs 17% random baseline).

### 6. ε-Greedy Exploration
- Added exploration (ε = 0.1):  
  - With 10% probability → choose random action  
  - With 90% probability → choose model-predicted best action
- Success rate dropped to **~15.6%**, showing exploration is risky in sparse-reward, slippery environments.

---

## 📊 Results Summary

| Policy Type | Episodes | Success Rate |
|--------------|-----------|---------------|
| Random Policy | 10,000 | 0.1734 |
| Model-Guided (Greedy) | 10,000 | 0.1965 |
| ε-Greedy Policy (ε=0.1) | 10,000 | 0.1566 |

**Conclusion:**  
- Model-guided greedy policy performed best.  
- ε-greedy exploration hurt performance due to the environment’s stochastic nature.  
- Demonstrated how **supervised learning** can approximate a policy even without explicit RL training.

---

## 🧠 Key Learnings

- Combining **reinforcement learning ideas** with **supervised models**.
- Understanding **state-action importance** and sparse reward problems.
- Observing the impact of **exploration vs exploitation** in decision-making.
- Gained experience with **Gymnasium**, **pandas**, **scikit-learn**, and **NumPy**.

---

## 🛠️ Tech Stack

- Python 3.x  
- Gymnasium  
- NumPy, Pandas  
- scikit-learn (RandomForestRegressor)  
- Jupyter / Google Colab

---

## 📂 Project Structure

