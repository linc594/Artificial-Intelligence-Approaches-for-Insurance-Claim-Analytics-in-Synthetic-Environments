"""
训练 ClaimPaymentEnv 上的 PPO。
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_claim_payment_env import (
    ClaimPaymentEnv, load_claims, compute_state_stats,
    TRAIN_TX, DEBUG, NORMALIZE, MAX_CLAIMS
)

MODEL_OUT = r"C:\Users\24495\Desktop\experiment\data\ppo_claim_payment_model.zip"

TOTAL_TIMESTEPS = 50_000 if not DEBUG else 5_000


def make_env():
    claims = load_claims(TRAIN_TX, max_claims=10 if DEBUG else MAX_CLAIMS)
    state_mean, state_std = compute_state_stats(claims)
    return ClaimPaymentEnv(
        claims,
        normalize=NORMALIZE,
        state_mean=state_mean,
        state_std=state_std,
    )

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        gamma=0.99,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_OUT)
    print("Saved PPO model ->", MODEL_OUT)