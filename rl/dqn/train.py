"""DQN training on CartPole-v1."""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rl.dqn.dqn import DQN, ReplayBuffer, train_episode, epsilon_by_episode
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("rl/dqn/config.yaml")
    set_seed(cfg["seed"])

    try:
        import gymnasium as gym
    except ImportError:
        print("Installing gymnasium...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium"])
        import gymnasium as gym

    device = get_device()
    print(f"Device: {device}")

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"CartPole: state={state_dim}, action={action_dim}")

    model = DQN(state_dim, action_dim, cfg["hidden_dim"]).to(device)
    target_model = DQN(state_dim, action_dim, cfg["hidden_dim"]).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    replay_buffer = ReplayBuffer(cfg["buffer_size"])

    num_episodes = cfg["num_episodes"]
    writer = SummaryWriter(log_dir="runs/dqn")

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        epsilon = epsilon_by_episode(episode, cfg["epsilon_start"], cfg["epsilon_end"], cfg["epsilon_decay"])

        while True:
            # ε-greedy action selection.
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = q.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            loss = train_episode(model, target_model, optimizer, replay_buffer, cfg["batch_size"], cfg["gamma"])

            if done:
                break

        # Update target network.
        if episode % cfg["target_update"] == 0:
            target_model.load_state_dict(model.state_dict())

        episode_rewards.append(episode_reward)
        writer.add_scalar("train/reward", episode_reward, episode)
        writer.add_scalar("train/epsilon", epsilon, episode)

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode [{episode:4d}/{num_episodes}]  Reward: {episode_reward:.0f}  Avg(50): {avg_reward:.1f}  ε: {epsilon:.3f}")

    writer.close()
    env.close()
    torch.save(model.state_dict(), cfg["model_path"])
    save_config(cfg, cfg["model_path"].replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {cfg['model_path']}")


if __name__ == "__main__":
    train()
