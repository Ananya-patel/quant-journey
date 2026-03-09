"""
ppo.py — PPO Algorithm
──────────────────────
Implements Proximal Policy Optimization.
(Schulman et al., 2017 — one of the most cited ML papers)

TRAINING LOOP:
  1. Collect N steps of experience (rollout)
  2. Compute returns and advantages
  3. Update policy K times on the collected data
  4. Repeat

KEY HYPERPARAMETERS:
  gamma:       0.99  — discount factor (care about future)
  gae_lambda:  0.95  — GAE smoothing (bias-variance tradeoff)
  clip_eps:    0.20  — PPO clipping range (±20% policy change)
  vf_coef:     0.50  — how much to weight critic loss
  ent_coef:    0.01  — entropy bonus (encourages exploration)
  n_epochs:    10    — how many times to reuse each batch
"""
"""
ppo.py — PPO Algorithm
──────────────────────
Implements Proximal Policy Optimization.
(Schulman et al., 2017)

KEY FIX: KL early stopping inside mini-batch loop.
If KL divergence > 0.05, stop updating that epoch.
This prevents the catastrophic KL=4.6 we saw before.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam


class PPOBuffer:
    def __init__(self, size: int, state_dim: int, action_dim: int):
        self.states    = np.zeros((size, state_dim),  dtype=np.float32)
        self.actions   = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards   = np.zeros(size,  dtype=np.float32)
        self.values    = np.zeros(size,  dtype=np.float32)
        self.log_probs = np.zeros(size,  dtype=np.float32)
        self.dones     = np.zeros(size,  dtype=np.float32)
        self.ptr  = 0
        self.size = size

    def store(self, state, action, reward, value, log_prob, done):
        i = self.ptr
        self.states[i]    = state
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.dones[i]     = done
        self.ptr += 1

    def get(self) -> dict:
        return {
            'states':    torch.FloatTensor(self.states[:self.ptr]),
            'actions':   torch.FloatTensor(self.actions[:self.ptr]),
            'rewards':   torch.FloatTensor(self.rewards[:self.ptr]),
            'values':    torch.FloatTensor(self.values[:self.ptr]),
            'log_probs': torch.FloatTensor(self.log_probs[:self.ptr]),
            'dones':     torch.FloatTensor(self.dones[:self.ptr]),
        }

    def reset(self):
        self.ptr = 0


class PPO:
    def __init__(self, model, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.20,
                 vf_coef=0.50, ent_coef=0.01,
                 n_epochs=10, batch_size=64,
                 target_kl=0.05):              # ← new param

        self.model      = model
        self.optimizer  = Adam(model.parameters(), lr=lr, eps=1e-5)
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.target_kl  = target_kl            # ← new param

    def compute_gae(self, rewards, values, dones,
                    last_value: float) -> tuple:
        n          = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val  = last_value
                next_done = 0.0
            else:
                next_val  = values[t + 1]
                next_done = dones[t + 1]

            delta    = (rewards[t] +
                        self.gamma * next_val * (1 - next_done) -
                        values[t])
            last_gae = (delta +
                        self.gamma * self.gae_lambda *
                        (1 - next_done) * last_gae)
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, buffer: PPOBuffer,
               last_value: float) -> dict:

        data = buffer.get()

        adv, returns = self.compute_gae(
            data['rewards'].numpy(),
            data['values'].numpy(),
            data['dones'].numpy(),
            last_value
        )

        adv_t     = torch.FloatTensor(adv)
        returns_t = torch.FloatTensor(returns)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        states  = data['states']
        actions = data['actions']
        old_lp  = data['log_probs']

        n       = len(states)
        metrics = {'policy_loss': [], 'value_loss': [],
                   'entropy': [],     'approx_kl': []}

        kl_exceeded = False                    # ← track early stop

        for epoch in range(self.n_epochs):
            if kl_exceeded:                    # ← stop all epochs
                break

            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]

                b_states  = states[idx]
                b_actions = actions[idx]
                b_old_lp  = old_lp[idx]
                b_adv     = adv_t[idx]
                b_returns = returns_t[idx]

                new_lp, entropy, value = \
                    self.model.evaluate(b_states, b_actions)

                log_ratio = new_lp - b_old_lp
                ratio     = log_ratio.exp()

                approx_kl = ((ratio - 1) - log_ratio).mean()

                # ── KL EARLY STOP ─────────────────────────────
                # If policy changed too much → stop immediately
                # Prevents the KL=4.6 explosion we saw before
                if approx_kl.item() > self.target_kl:
                    kl_exceeded = True
                    break                      # ← exit batch loop

                # ── PPO CLIPPED OBJECTIVE ──────────────────────
                obj1 = ratio * b_adv
                obj2 = ratio.clamp(
                    1 - self.clip_eps,
                    1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(obj1, obj2).mean()

                # ── VALUE LOSS ────────────────────────────────
                value_loss = nn.functional.mse_loss(value, b_returns)

                # ── ENTROPY BONUS ─────────────────────────────
                entropy_loss = -entropy.mean()

                # ── TOTAL LOSS ────────────────────────────────
                loss = (policy_loss +
                        self.vf_coef  * value_loss +
                        self.ent_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 0.5)
                self.optimizer.step()

                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(-entropy_loss.item())
                metrics['approx_kl'].append(approx_kl.item())

        return {k: np.mean(v) if v else 0.0
                for k, v in metrics.items()}