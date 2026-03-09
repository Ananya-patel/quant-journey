"""
model.py — Actor-Critic Networks for PPO
─────────────────────────────────────────
Two separate networks:

ACTOR (Policy network):
  Input:  state (8 features)
  Output: mean + log_std of Gaussian distribution
          over actions [bid_offset, ask_offset]

  Why Gaussian?
  Actions are continuous (any value 0.01-0.50)
  We output a distribution, not a single value
  This allows EXPLORATION — sample different actions
  As training progresses, std shrinks → more confident

CRITIC (Value network):
  Input:  state (8 features)
  Output: single scalar V(s)
          estimated total future reward from this state

  The critic doesn't act — it only evaluates.
  It's the "coach" that tells the actor:
  "That was better/worse than I expected"
"""

"""
model.py — Actor-Critic Networks for PPO (v2)
─────────────────────────────────────────────
Key fixes from debug:
  1. Actor mean initialized to center of action space (0.25)
  2. Log_std starts at -1.0 (std=0.37, not 1.0)
  3. Log_prob computed on SAME action that gets stored
     (no clipping mismatch)
  4. Actions bounded via tanh transform (not hard clip)
     tanh keeps gradients flowing through the boundary
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


# Action bounds
ACT_LOW  = 0.01
ACT_HIGH = 0.50
ACT_MID  = (ACT_HIGH + ACT_LOW) / 2      # 0.255
ACT_HALF = (ACT_HIGH - ACT_LOW) / 2      # 0.245


class Actor(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(hidden, action_dim)

        # log_std = -1.0 → std = 0.37
        # Enough exploration but not exploding
        self.log_std = nn.Parameter(
            torch.full((action_dim,), -1.0))

        # Init weights small
        nn.init.orthogonal_(self.net[0].weight, gain=1.0)
        nn.init.orthogonal_(self.net[2].weight, gain=1.0)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        # Init bias so tanh(0) → action near center (0.255)
        nn.init.zeros_(self.mean_head.bias)

    def _raw_to_action(self, raw):
        """
        Map raw network output → valid action range.
        Uses tanh: output in (-1, 1) → scale to (ACT_LOW, ACT_HIGH)

        WHY TANH instead of clip:
          clip(x, 0.01, 0.50) has zero gradient at boundary
          tanh always has gradient → policy can always learn
        """
        return ACT_MID + ACT_HALF * torch.tanh(raw)

    def forward(self, state):
        features = self.net(state)
        raw_mean = self.mean_head(features)
        mean     = self._raw_to_action(raw_mean)
        std      = self.log_std.exp().clamp(0.005, 0.3)
        std      = std.expand_as(mean)
        return Normal(mean, std), raw_mean

    def get_action(self, state, deterministic=False):
        """
        Returns action AND log_prob computed CONSISTENTLY.
        Both use the same action value — no clipping mismatch.
        """
        dist, raw_mean = self.forward(state)

        if deterministic:
            # Use mean directly (already in valid range)
            action = dist.mean
        else:
            # Sample in raw space, transform
            raw_sample = dist.rsample()
            action     = self._raw_to_action(raw_sample)
            # Recompute dist at this action for correct log_prob
            # (sampling in transformed space)

        # Log prob: use action directly with the distribution
        # Clamp action slightly inside bounds for numerical safety
        action_safe = action.clamp(ACT_LOW + 1e-4,
                                   ACT_HIGH - 1e-4)
        log_prob = dist.log_prob(action_safe).sum(dim=-1)

        return action_safe, log_prob

    def evaluate(self, state, action):
        """
        Evaluate stored actions — MUST match get_action exactly.
        """
        dist, _ = self.forward(state)
        action_safe = action.clamp(ACT_LOW + 1e-4,
                                   ACT_HIGH - 1e-4)
        log_prob = dist.log_prob(action_safe).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, state_dim=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        nn.init.orthogonal_(self.net[0].weight, gain=1.0)
        nn.init.orthogonal_(self.net[2].weight, gain=1.0)
        nn.init.orthogonal_(self.net[4].weight, gain=1.0)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden=64):
        super().__init__()
        self.actor  = Actor(state_dim, action_dim, hidden)
        self.critic = Critic(state_dim, hidden)

    def get_action(self, state, deterministic=False):
        return self.actor.get_action(state, deterministic)

    def get_value(self, state):
        return self.critic(state)

    def evaluate(self, state, action):
        log_prob, entropy = self.actor.evaluate(state, action)
        value             = self.critic(state)
        return log_prob, entropy, value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)