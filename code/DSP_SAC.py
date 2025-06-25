import os, time, math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import gymnasium as gym                    # use Gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

plt.style.use("seaborn-v0_8")

DATA_CSV  = Path("ad_bid_dataset.csv")
DAY_BUDGET = 12_000.0          # hard budget cap
TARGET_ROI = 1.6
BATCH_SIZE = 256
TIME_STEPS = 60_000            # a bit longer for demo stability

# =========================================================
# Load and peek at the data
# =========================================================
df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
print(f"Loaded {len(df):,} lines")          # show five rows
print(df.head().to_markdown())

# =========================================================
# Ad-bidding environment with a **learnable λ_b** multiplier
# =========================================================
class AdBiddingEnv(gym.Env):
    """
    State 12-dim ➜ continuous bid [$0, 5].
    Reward = profit − λ_b·overspend  (Lagrange-style).
    λ_b is updated on-line (dual ascent) so that spending converges
    toward the DAY_BUDGET target.
    """
    def __init__(self, log_df: pd.DataFrame, day_budget: float,
                 lam_lr: float = 1e-4):
        super().__init__()
        self.df   = log_df.sort_values("timestamp").reset_index(drop=True)
        self.day_budget = day_budget
        self.lam_lr     = lam_lr
        self.lambda_b   = 0.1           # initial multiplier

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(12,), dtype=np.float32)
        self.action_space       = gym.spaces.Box(low=0., high=5.,
                                                 shape=(1,), dtype=np.float32)
        self._ptr = 0
        self._spent = 0.0
        self._rev   = 0.0
        self._episode = 0

    def _vec(self, row):
        return np.array([
            row.floor_price, row.comp_bid_mean, row.pred_ctr,
            row.hour/23.0, row.weekday/6.0, row.budget_remaining_pct,
            row.prev_win_rate, row.action_bid,
            row.click, row.conversion, row.revenue, row.cost
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._ptr = 0; self._spent = 0.; self._rev = 0.
        self._episode += 1
        return self._vec(self.df.loc[self._ptr]), {}

    def step(self, action):
        bid   = float(action[0])
        row   = self.df.loc[self._ptr]

        win   = bid >= row.comp_bid_mean
        click = np.random.rand() < (row.pred_ctr + 0.1*win)
        conv  = click and (np.random.rand() < (0.1 + 0.2*row.pred_ctr))

        cost  = bid if click else 0.
        rev   = np.random.uniform(10, 50) if conv else 0.

        self._spent += cost
        self._rev   += rev

        overspend   = max(0., self._spent - self.day_budget)
        profit      = rev - cost
        reward      = profit - self.lambda_b * overspend

        # ----- dual ascent: update λ_b -----
        self.lambda_b += self.lam_lr * overspend
        self.lambda_b  = max(0., self.lambda_b)         # keep ≥0

        self._ptr += 1
        done = self._ptr >= len(self.df)

        info = dict(step=self._ptr,
                    spent=self._spent,
                    revenue=self._rev,
                    roi=self._rev/self._spent if self._spent else 0.,
                    lambda_b=self.lambda_b)

        state = self._vec(row)
        return state, reward, done, False, info

# =========================================================
# Build vector env & SAC agent
# =========================================================
env = DummyVecEnv([lambda: AdBiddingEnv(df, DAY_BUDGET)])

log_dir = "./sac_logs"
agent_logger = configure(log_dir, ["stdout", "csv"])

model = SAC(
    "MlpPolicy", env,
    buffer_size=200_000,
    batch_size=BATCH_SIZE,
    learning_rate=3e-4,
    gamma=0.99, tau=0.005,
    ent_coef="auto",
    verbose=1
)
model.set_logger(agent_logger)

# =========================================================
# Train with progress prints
# =========================================================
print("⏳ Training ...")
model.learn(total_timesteps=TIME_STEPS, log_interval=10)
model.save("sac_bid_agent_budgeted")
print("✅ Training done.\n")

# =========================================================
# Evaluate one full "day" & collect traces
# =========================================================
# -----  Evaluation on a new day -----
obs = env.reset()            # single ndarray
spend_trace = []; rev_trace = []; lam_trace = []; rew_trace = []

for _ in range(len(df)):
    act, _ = model.predict(obs, deterministic=True)
    obs, reward, done, infos = env.step(act)   # <-- note the "s"
    info   = infos[0]          # unpack from list
    reward = reward[0]         # scalar
    done   = done[0]

    spend_trace.append(info["spent"])
    rev_trace.append(info["revenue"])
    lam_trace.append(info["lambda_b"])
    rew_trace.append(reward)

    if done:
        break

final_spent  = spend_trace[-1]
final_rev    = rev_trace[-1]
print("=== EVAL SUMMARY ===")
print(f"Spent     : ${final_spent:,.2f}")
print(f"Revenue   : ${final_rev:,.2f}")
print(f"ROI       : {final_rev/final_spent:4.2f}")
print(f"λ_b final : {lam_trace[-1]:.3f}")

# =========================================================
# Plots
# =========================================================
fig, ax = plt.subplots(1, 2, figsize=(10,3))
ax[0].plot(spend_trace, label="Spend")
ax[0].axhline(DAY_BUDGET, color="r", ls="--", label="Budget cap")
ax[0].set_title("Cumulative spend vs cap"); ax[0].legend()

ax[1].plot(lam_trace)
ax[1].set_title("Lagrange multiplier λ_b over time")
plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3))
plt.plot(pd.Series(rew_trace).rolling(500).mean())
plt.title("Smoothed reward during eval"); plt.xlabel("step"); plt.ylabel("reward")
plt.tight_layout(); plt.show()

# =========================================================
#  Export tomorrow's bid list (2 000 rows)
# =========================================================
tomorrow = df.sample(2_000, random_state=123).copy()
tomorrow["rec_bid"] = 0.0
for idx, row in tomorrow.iterrows():
    s = env.envs[0]._vec(row)
    b, _ = model.predict(s, deterministic=True)
    tomorrow.at[idx, "rec_bid"] = float(b)
out_path = "tomorrow_bid_list.csv"
tomorrow[["auction_id", "rec_bid"]].to_csv(out_path, index=False)
print(f"📄 Wrote {out_path}")
