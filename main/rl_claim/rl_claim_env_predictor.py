# ===================================
# rl_claim_env_predictor.py
# ===================================
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import joblib



# 加载一次 predictor 和 metadata（避免每次 env 都重复加载）
predictor = joblib.load(os.path.join("claim_predictor.pkl"))
meta = joblib.load(os.path.join("predictor_meta.pkl"))
# sigma = meta["sigma"]   # 残差噪声不用了
feature_cols = meta["feature_cols"]

class ClaimSettlementEnv(gym.Env):

    def __init__(self, claim_row, tx_row,
                 episode_length=30,       # 改为30
                 partial_pay_ratio=0.2,
                 uncertainty_scale=1.0):

        super().__init__()

        self.episode_length = episode_length
        self.partial_pay_ratio = partial_pay_ratio
        self.uncertainty_scale = uncertainty_scale

        # ------- 预测初始估计（而不是暴露真实值） -------
        features = np.array([claim_row[col] for col in feature_cols]).reshape(1, -1)
        pred = float(predictor.predict(features)[0])

        # 加残差噪声（注释掉）
        # noise = np.random.normal(0, sigma * uncertainty_scale)
        # self.claim_size_est = max(0.0, pred + noise)

        # 直接使用预测值
        self.claim_size_est = max(0.0, pred)

        # 真实值仅用于评估
        self.true_claim_size = float(claim_row["claim_size"])

        self.outstanding = self.claim_size_est
        self.cumulative_paid = 0.0

        self.delay_cost_rate = 0.01
        self.t = 0

        self.observation_space = spaces.Box(low=0, high=np.inf,
                                            shape=(4,), dtype=np.float32)
        # 只保留 partial payment 和 settle 两个动作
        self.action_space = spaces.Discrete(2)  # 0=partial, 1=settle

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.cumulative_paid = 0.0
        self.outstanding = self.claim_size_est
        return self._get_state(), {}

    def _get_state(self):
        return np.array([
            self.t,
            self.cumulative_paid,
            self.outstanding,
            self.claim_size_est
        ], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        done = False

        # 动作 0 = partial payment
        if action == 0:
            pay_amt = self.outstanding * self.partial_pay_ratio
            self.cumulative_paid += pay_amt
            self.outstanding -= pay_amt
            reward -= pay_amt

        # 动作 1 = settle fully
        elif action == 1:
            reward -= self.outstanding
            self.cumulative_paid += self.outstanding
            self.outstanding = 0.0
            done = True

        # 延迟成本
        reward -= self.delay_cost_rate * self.outstanding

        self.t += 1
        if self.t >= self.episode_length:
            done = True

        return self._get_state(), reward, done, False, {}





