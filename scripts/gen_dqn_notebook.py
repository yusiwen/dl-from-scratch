#!/usr/bin/env python3
"""Generate DQN notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},"language_info": {"name": "python", "version": "3.12.0"}}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("# DQN: Deep Q-Network\n\nReinforcement learning with experience replay on CartPole-v1.")

md("""## 背景

DQN（Mnih et al. 2015）将深度学习与 Q-Learning 结合，首次在 Atari 游戏上达到人类水平。
核心创新：

- **经验回放（Experience Replay）**：存储过去经验，随机采样训练，打破数据相关性
- **目标网络（Target Network）**：固定 Q-target，稳定训练

环境：**CartPole-v1** — 控制小车左右移动，保持杆子不倒。
状态：4 维（位置、速度、角度、角速度），动作：2 维（左、右）。
""")

md("""## 数学原理

### Q-Learning

$$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left(r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)\\right)$$

### DQN Loss

$$\\mathcal{L} = \\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{D}} \\left[\\left(r + \\gamma \\max_{a'} Q_{\\theta^-}(s', a') - Q_\\theta(s, a)\\right)^2\\right]$$

- $\\mathcal{D}$: 经验回放缓冲区
- $\\theta$: 在线网络参数
- $\\theta^-$: 目标网络参数（每隔 $C$ 步复制一次）

### ε-greedy 探索

$$a = \\begin{cases} \\text{random}, & \\text{概率 } \\varepsilon \\\\ \\arg\\max_a Q(s, a), & \\text{概率 } 1-\\varepsilon \\end{cases}$$

ε 随时间指数衰减。
""")

code("""\
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.dqn import DQN, ReplayBuffer, train_episode, epsilon_by_episode
from utils.config import load_config

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(f"State: {state_dim}  Action: {action_dim}")
""")

code("""\
model = DQN(state_dim, action_dim, hidden_dim=128).to(device)
target = DQN(state_dim, action_dim, hidden_dim=128).to(device)
target.load_state_dict(model.state_dict())
target.eval()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
""")

md("""## 训练

> ⏱ 预估耗时：**500 episode × ~0.2s ≈ 2 分钟**（M4 Max）
""")

code("""\
NUM_EPISODES = 500
LR = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 50000
TARGET_UPDATE = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500

optimizer = optim.Adam(model.parameters(), lr=LR)
buffer = ReplayBuffer(BUFFER_SIZE)
rewards = []

for episode in range(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    episode_reward = 0
    eps = epsilon_by_episode(episode, EPSILON_START, EPSILON_END, EPSILON_DECAY)

    while True:
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                action = q.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        _ = train_episode(model, target, optimizer, buffer, BATCH_SIZE, GAMMA)
        if done:
            break

    if episode % TARGET_UPDATE == 0:
        target.load_state_dict(model.state_dict())

    rewards.append(episode_reward)
    if episode % 50 == 0:
        avg = np.mean(rewards[-50:])
        print(f"Episode [{episode:3d}/{NUM_EPISODES}]  Reward: {episode_reward:.0f}  Avg(50): {avg:.1f}  ε: {eps:.3f}")

env.close()
""")

md("""## Reward 曲线""")

code("""\
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.title("DQN Training on CartPole")
plt.grid(True)
plt.axhline(y=195, color='r', linestyle='--', label='Solved (195)')
plt.legend()
plt.show()
""")

md("""\
## 思考题

1. 为什么需要经验回放（Experience Replay）？在线学习会有什么问题？
2. 目标网络（Target Network）解决了什么？如果不固定 target，Q 值会发散吗？
3. ϵ-greedy 中的 ϵ 从 1.0 开始衰减有什么含义？
4. 如果把 `hidden_dim` 从 128 改到 32，训练速度会怎样？收敛难度呢？
""")

nb.cells = cells
with open("dqn/dqn.ipynb", "w") as f:
    nbf.write(nb, f)
print("Generated dqn/dqn.ipynb")
