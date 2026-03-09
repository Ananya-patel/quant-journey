"""
environment.py
──────────────
Simulated Limit Order Book for market making.

The environment mimics a real exchange:
  - Price moves randomly (GBM + occasional jumps)
  - Buy and sell orders arrive randomly
  - Our agent posts bid and ask quotes
  - When market price crosses our quote → fill

KEY PARAMETERS:
  S0:        Initial price ($100)
  sigma:     Daily volatility (1%)
  lambda:    Order arrival rate (Poisson)
  T:         Episode length (1 trading day = 390 minutes)
  dt:        Time step (1 minute)
  lot_size:  Units per fill
  inv_limit: Maximum inventory (+/- units)
"""

"""
environment.py
──────────────
Simulated Limit Order Book for market making.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MarketMakingEnv(gym.Env):

    def __init__(self,
                 S0:               float = 100.0,
                 sigma:            float = 0.01,
                 lam:              float = 3.0,
                 T:                int   = 390,
                 dt:               float = 1/390,
                 lot_size:         int   = 1,
                 inv_limit:        int   = 50,
                 inv_penalty:      float = 0.05,      # ← increased
                 terminal_penalty: float = 0.1):
        super().__init__()

        self.S0               = S0
        self.sigma            = sigma
        self.lam              = lam
        self.T                = T
        self.dt               = dt
        self.lot_size         = lot_size
        self.inv_limit        = inv_limit
        self.inv_penalty      = inv_penalty
        self.terminal_penalty = terminal_penalty

        self.action_space = spaces.Box(
            low   = np.array([0.01, 0.01], dtype=np.float32),
            high  = np.array([0.50, 0.50], dtype=np.float32),
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (8,),
            dtype = np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t                = 0
        self.price            = self.S0
        self.inventory        = 0
        self.cash             = 0.0
        self.pnl              = 0.0
        self.bid_fills        = []
        self.ask_fills        = []
        self.price_hist       = [self.S0]
        self.trades           = 0
        self.spread_captured  = 0.0

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        inv_norm  = self.inventory / self.inv_limit

        if len(self.price_hist) > 1:
            price_ret = (self.price_hist[-1] /
                         self.price_hist[-2] - 1)
        else:
            price_ret = 0.0

        if len(self.price_hist) > 2:
            rets = np.diff(np.log(self.price_hist[-20:]))
            vol  = rets.std() if len(rets) > 1 else \
                   self.sigma * np.sqrt(self.dt)
        else:
            vol = self.sigma * np.sqrt(self.dt)

        spread   = getattr(self, '_last_spread', 0.02)
        pnl_norm = np.clip(self.pnl / (self.S0 * 0.1), -1, 1)
        time_rem = 1.0 - (self.t / self.T)
        bid_rate = np.mean(self.bid_fills[-10:]) \
                   if self.bid_fills else 0.5
        ask_rate = np.mean(self.ask_fills[-10:]) \
                   if self.ask_fills else 0.5

        return np.array([
            inv_norm,
            price_ret * 100,
            spread,
            vol * 100,
            pnl_norm,
            time_rem,
            bid_rate,
            ask_rate,
        ], dtype=np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(action, 0.01, 0.50)
        bid_offset, ask_offset = action

        bid_price         = self.price - bid_offset
        ask_price         = self.price + ask_offset
        spread            = bid_offset + ask_offset
        self._last_spread = spread

        # ── PRICE EVOLUTION (GBM + jumps) ─────────────────────
        dW   = np.random.normal(0, np.sqrt(self.dt))
        jump = 0.0
        if np.random.random() < 0.005:
            jump = (np.random.choice([-1, 1]) *
                    np.random.uniform(0.001, 0.005) * self.price)

        self.price = self.price * np.exp(
            -0.5 * self.sigma**2 * self.dt +
            self.sigma * dW
        ) + jump
        self.price = max(self.price, 1.0)
        self.price_hist.append(self.price)

        # ── ORDER ARRIVALS ────────────────────────────────────
        n_buy_orders  = np.random.poisson(self.lam * self.dt)
        n_sell_orders = np.random.poisson(self.lam * self.dt)

        kappa      = 1.5
        p_bid_fill = np.exp(-kappa * bid_offset)
        p_ask_fill = np.exp(-kappa * ask_offset)

        bid_filled = 0
        ask_filled = 0

        # Record inventory BEFORE fills (for reduction bonus)
        inv_before = self.inventory

        for _ in range(n_sell_orders):
            if np.random.random() < p_bid_fill:
                if self.inventory > -self.inv_limit:
                    self.inventory += self.lot_size
                    self.cash      -= bid_price * self.lot_size
                    bid_filled     += 1
                    self.trades    += 1

        for _ in range(n_buy_orders):
            if np.random.random() < p_ask_fill:
                if self.inventory < self.inv_limit:
                    self.inventory -= self.lot_size
                    self.cash      += ask_price * self.lot_size
                    ask_filled     += 1
                    self.trades    += 1

        self.bid_fills.append(1 if bid_filled > 0 else 0)
        self.ask_fills.append(1 if ask_filled > 0 else 0)

        # ── MARK-TO-MARKET PnL ────────────────────────────────
        self.pnl = self.cash + self.inventory * self.price

        # ── REWARD ────────────────────────────────────────────
        spread_income = (bid_filled * bid_offset +
                         ask_filled * ask_offset)
        self.spread_captured += spread_income

        # Quadratic inventory penalty (stronger now: 0.05)
        inv_penalty = self.inv_penalty * (self.inventory ** 2)

        # Bonus for REDUCING inventory toward zero
        inv_reduced     = abs(inv_before) > abs(self.inventory)
        reduction_bonus = (0.05 * abs(inv_before - self.inventory)
                           if inv_reduced else 0.0)

        reward = spread_income - inv_penalty + reduction_bonus

        # ── TERMINAL ──────────────────────────────────────────
        self.t   += 1
        done      = self.t >= self.T
        truncated = False

        if done:
            terminal_pen = (self.terminal_penalty *
                            abs(self.inventory) *
                            self.price * 0.01)
            reward -= terminal_pen

        obs  = self._get_obs()
        info = {
            'pnl':             self.pnl,
            'inventory':       self.inventory,
            'trades':          self.trades,
            'spread_captured': self.spread_captured,
            'price':           self.price,
        }

        return obs, reward, done, truncated, info