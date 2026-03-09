"""
debug.py — Step by step diagnosis
"""
import torch
import numpy as np
from src.environment import MarketMakingEnv
from src.model import ActorCritic
from src.ppo import PPO, PPOBuffer

print("="*55)
print("STEP 1: Check Actor output")
print("="*55)
model = ActorCritic(state_dim=8, action_dim=2, hidden=64)
state = torch.FloatTensor([[0.1, 0.0, 0.05, 0.5,
                             0.0, 0.5, 0.5, 0.5]])
dist, _ = model.actor.forward(state)
print(f"  Distribution mean:    {dist.mean.detach()}")
print(f"  Distribution std:     {dist.stddev.detach()}")
print(f"  log_std param:        {model.actor.log_std.detach()}")

print("\n" + "="*55)
print("STEP 2: Check get_action log_prob")
print("="*55)
action, log_prob = model.get_action(state)
print(f"  Action:   {action.detach()}")
print(f"  Log prob: {log_prob.item():.4f}")
print(f"  Prob:     {log_prob.exp().item():.6f}")

print("\n" + "="*55)
print("STEP 3: Check evaluate() log_prob")
print("="*55)
log_prob2, entropy, value = model.evaluate(state, action)
print(f"  Log prob (evaluate): {log_prob2.item():.4f}")
print(f"  Entropy:             {entropy.item():.4f}")
print(f"  Value:               {value.item():.4f}")

print("\n" + "="*55)
print("STEP 4: Check ratio")
print("="*55)
ratio = (log_prob2 - log_prob).exp()
print(f"  Ratio (should be ~1.0): {ratio.item():.6f}")

print("\n" + "="*55)
print("STEP 5: Simulate one PPO update")
print("="*55)
env    = MarketMakingEnv(lam=15.0)
buffer = PPOBuffer(size=env.T, state_dim=8, action_dim=2)
ppo    = PPO(model, lr=3e-4)

state_np, _ = env.reset()
for _ in range(env.T):
    s_t = torch.FloatTensor(state_np).unsqueeze(0)
    with torch.no_grad():
        a, lp  = model.get_action(s_t)
        v      = model.get_value(s_t)
    a_np = a.squeeze(0).numpy()
    ns, r, done, _, info = env.step(a_np)
    buffer.store(state_np, a_np, r,
                 v.item(), lp.item(), float(done))
    state_np = ns
    if done: break

print(f"  Buffer filled: {buffer.ptr} steps")
print(f"  Rewards mean:  {buffer.rewards[:buffer.ptr].mean():.4f}")
print(f"  LogProbs mean: {buffer.log_probs[:buffer.ptr].mean():.4f}")
print(f"  LogProbs std:  {buffer.log_probs[:buffer.ptr].std():.6f}")

# Check if log_probs are all identical (dead giveaway)
unique_lp = np.unique(buffer.log_probs[:buffer.ptr].round(4))
print(f"  Unique log_prob values: {len(unique_lp)}")
if len(unique_lp) < 5:
    print(f"  VALUES: {unique_lp}")
    print(f"   Log probs are nearly identical — this is the bug")
else:
    print(f"   Log probs vary normally")

# Run one update and check if actor weights change
actor_weights_before = model.actor.mean_head.weight.clone()
log_std_before = model.actor.log_std.clone()

with torch.no_grad():
    last_val = model.get_value(
        torch.FloatTensor(state_np).unsqueeze(0)).item()

metrics = ppo.update(buffer, last_val)
print(f"\n  Policy loss: {metrics['policy_loss']:.6f}")
print(f"  Value  loss: {metrics['value_loss']:.6f}")
print(f"  Entropy:     {metrics['entropy']:.6f}")
print(f"  Approx KL:   {metrics['approx_kl']:.6f}")

weight_change = (model.actor.mean_head.weight -
                 actor_weights_before).abs().mean().item()
logstd_change = (model.actor.log_std -
                 log_std_before).abs().mean().item()

print(f"\n  Actor weight change:  {weight_change:.8f}")
print(f"  Log_std change:       {logstd_change:.8f}")

if weight_change < 1e-7:
    print("   ACTOR WEIGHTS DID NOT CHANGE — gradient not flowing")
else:
    print("   Actor weights updated")