import torch
from src.model   import ActorCritic, count_parameters
from src.trainer import train

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  P05 — PPO Market Making Agent")
    print("═"*55)

    model = ActorCritic(state_dim=8, action_dim=2, hidden=64)
    print(f"[main] Parameters: {count_parameters(model):,}")

    history = train(model,
                    n_episodes = 1000,
                    log_every  = 50,
                    lam        = 15.0)    # ← dense rewards

    print("\n[main] Training complete!")
    print("[main] Run evaluate.py for full analysis")