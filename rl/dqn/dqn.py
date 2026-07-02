"""DQN: Deep Q-Network for CartPole-v1."""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Q-network: state → action values."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)


def train_episode(model, target_model, optimizer, replay_buffer, batch_size, gamma):
    """Train DQN on one batch from replay buffer."""
    if len(replay_buffer) < batch_size:
        return 0.0

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Q(s, a)
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target: r + γ * max Q'(s', a')
    with torch.no_grad():
        max_next_q = target_model(next_states).max(dim=1)[0]
        target = rewards + gamma * max_next_q * (1 - dones)

    loss = F.mse_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def epsilon_by_episode(episode, epsilon_start, epsilon_end, epsilon_decay):
    """Annealed ε-greedy."""
    return epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * episode / epsilon_decay)
