"""
trainer.py
──────────
Runs the full PPO training loop.
"""

import numpy as np
import torch
import json, os
from src.environment import MarketMakingEnv
from src.ppo         import PPO, PPOBuffer


def convert(obj):
    """Convert numpy types to Python native for JSON."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def train(model, n_episodes=1000, log_every=50,
          save_dir='models', lam=15.0):

    os.makedirs(save_dir, exist_ok=True)

    env    = MarketMakingEnv(lam=lam)
    ppo    = PPO(model, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.20,
                 vf_coef=0.5, ent_coef=0.01,
                 n_epochs=10, batch_size=64)
    buffer = PPOBuffer(size=env.T,
                       state_dim=8,
                       action_dim=2)

    history = {
        'episode_reward': [],
        'episode_pnl':    [],
        'episode_trades': [],
        'spread_captured':[],
        'final_inventory':[],
        'policy_loss':    [],
        'value_loss':     [],
        'entropy':        [],
    }

    best_pnl = -np.inf

    print(f"\n{'Episode':>8} {'Reward':>9} {'PnL':>9} "
          f"{'Trades':>7} {'Spread$':>9} {'Inv':>6} "
          f"{'Ent':>7}")
    print("─" * 65)

    for ep in range(1, n_episodes + 1):

        # ── COLLECT EPISODE ───────────────────────────────────
        state, _ = env.reset()
        buffer.reset()
        ep_reward = 0.0

        for step in range(env.T):
            state_t = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob = model.get_action(state_t)
                value            = model.get_value(state_t)

            action_np = action.squeeze(0).numpy()
            lp_np     = log_prob.item()
            val_np    = value.item()

            next_state, reward, done, _, info = \
                env.step(action_np)

            buffer.store(state, action_np, reward,
                         val_np, lp_np, float(done))

            state      = next_state
            ep_reward += reward

            if done:
                break

        # Last value for GAE bootstrap
        with torch.no_grad():
            last_val = model.get_value(
                torch.FloatTensor(state).unsqueeze(0)
            ).item()

        # ── PPO UPDATE ────────────────────────────────────────
        metrics = ppo.update(buffer, last_val)

        # ── LOGGING ───────────────────────────────────────────
        history['episode_reward'].append(ep_reward)
        history['episode_pnl'].append(info['pnl'])
        history['episode_trades'].append(info['trades'])
        history['spread_captured'].append(info['spread_captured'])
        history['final_inventory'].append(info['inventory'])
        history['policy_loss'].append(metrics['policy_loss'])
        history['value_loss'].append(metrics['value_loss'])
        history['entropy'].append(metrics['entropy'])

        if ep % log_every == 0:
            r  = np.mean(history['episode_reward'][-log_every:])
            p  = np.mean(history['episode_pnl'][-log_every:])
            tr = np.mean(history['episode_trades'][-log_every:])
            sc = np.mean(history['spread_captured'][-log_every:])
            iv = np.mean(np.abs(
                     history['final_inventory'][-log_every:]))
            en = np.mean(history['entropy'][-log_every:])

            print(f"{ep:>8} {r:>+9.3f} {p:>+9.3f} "
                  f"{tr:>7.1f} {sc:>9.3f} {iv:>6.2f} "
                  f"{en:>7.3f}")

            if p > best_pnl:
                best_pnl = p
                torch.save(model.state_dict(),
                    os.path.join(save_dir, 'best_ppo.pt'))

    # ── SAVE HISTORY ──────────────────────────────────────────
    with open(os.path.join(save_dir, 'ppo_history.json'), 'w') as f:
        json.dump(history, f, default=convert)

    print(f"\n[trainer] Best episode PnL: ${best_pnl:.3f}")
    print(f"[trainer] Model saved → {save_dir}/best_ppo.pt")
    return history