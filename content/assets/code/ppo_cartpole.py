"""
Minimal PPO on CartPole-v1 — companion script for blog draft.
Run: pip install gymnasium torch && python drafts/ppo_cartpole.py
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# --- hyperparameters (CleanRL-style defaults, scaled down for CPU demo) ---
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NUM_STEPS = 1024
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
TOTAL_UPDATES = 200
SEED = 42


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE-Lambda; rewards/values/dones shape (T,)."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(rewards.size(0))):
        next_non_terminal = 1.0 - dones[t]
        next_v = next_value if t == rewards.size(0) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def main() -> None:
    torch.manual_seed(SEED)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCritic(obs_dim, act_dim)
    opt = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    global_step = 0
    episode_returns: list[float] = []
    obs, _ = env.reset(seed=SEED)
    ep_ret = 0.0

    for update in range(1, TOTAL_UPDATES + 1):
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []

        for _ in range(NUM_STEPS):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, value = net(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            obs_buf.append(obs_t)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf.append(torch.tensor(reward, dtype=torch.float32))
            done_buf.append(torch.tensor(float(done)))
            val_buf.append(value.detach())

            ep_ret += reward
            global_step += 1
            obs = next_obs
            if done:
                episode_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            next_obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            _, next_val = net(next_obs_t)

        rewards = torch.stack(rew_buf)
        values = torch.cat(val_buf)
        dones = torch.stack(done_buf)
        advantages, returns = compute_gae(rewards, values, dones, next_val.squeeze(), GAMMA, GAE_LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_logp = torch.cat(logp_buf).detach()
        obs_batch = torch.cat(obs_buf)
        act_batch = torch.cat(act_buf)

        for _ in range(UPDATE_EPOCHS):
            idx = torch.randperm(NUM_STEPS)
            for start in range(0, NUM_STEPS, MINIBATCH_SIZE):
                mb = idx[start : start + MINIBATCH_SIZE]
                logits, value = net(obs_batch[mb])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_batch[mb])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp[mb])
                mb_adv = advantages[mb]
                pg1 = mb_adv * ratio
                pg2 = mb_adv * torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                policy_loss = -torch.min(pg1, pg2).mean()

                value_loss = 0.5 * ((returns[mb] - value) ** 2).mean()
                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                opt.step()

        if update % 40 == 0 and episode_returns:
            recent = episode_returns[-10:]
            print(f"update {update} | steps {global_step} | mean return (last 10 ep) {sum(recent)/len(recent):.1f}")

    env.close()
    if episode_returns:
        print(f"final mean return (last 20 ep): {sum(episode_returns[-20:]) / min(20, len(episode_returns)):.1f}")


if __name__ == "__main__":
    main()
