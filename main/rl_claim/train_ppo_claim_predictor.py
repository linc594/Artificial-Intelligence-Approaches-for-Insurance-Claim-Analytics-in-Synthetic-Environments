# ===================================
# train_ppo_claim_predictor.py
# ===================================

import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from rl_claim_env_predictor import ClaimSettlementEnv


# ===== PATH CONFIG =====
TRAIN_CLAIMS = r"C:\Users\24495\Desktop\experiment\data\train data\train_claims.csv"
TRAIN_TX     = r"C:\Users\24495\Desktop\experiment\data\train data\train_transactions.csv"

SEED = 43
# ===== LOAD DATA =====
claims = pd.read_csv(TRAIN_CLAIMS)
tx     = pd.read_csv(TRAIN_TX)

claims["claim_no"] = claims["claim_no"].astype(str)
tx["claim_no"]     = tx["claim_no"].astype(str)

set_random_seed(SEED)
# ===== ENV FACTORY =====
def make_env():
    # randomly sample one claim per episode
    row = claims.sample(1).iloc[0]

    # transactions belonging to this claim
    tx_rows = tx[tx["claim_no"] == row["claim_no"]]

    return ClaimSettlementEnv(row, tx_rows)


# ===== VECTOR ENV =====
env = DummyVecEnv([make_env])


# ===== PPO MODEL =====
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    seed=SEED
)


# ===== TRAIN =====
model.learn(total_timesteps=50_000)


# ===== SAVE =====
model.save("ppo_claim_predictor_model")
print("Saved PPO model → ppo_claim_predictor_model")





