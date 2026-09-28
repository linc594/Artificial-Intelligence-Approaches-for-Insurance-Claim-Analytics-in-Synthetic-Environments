"""
ClaimDelayEnv
-------------
每个 episode 对应一个 claim。
agent 逐期调整 delay 估计，目标是让最后一期的 current_delay
尽可能接近真实最后一笔 payment_delay。

设计要点：
- 只加载 transactions 文件，不加载 claims 文件；
- setldel 用向上取整到 0.5 模糊化，进入 state 作为参考；
- current_delay 初始化为 0（第一笔），之后每步设为上一笔真实 delay；
- 没有“最后一期自动补齐”；
- 每期 agent 独立输出 pred_delay_t；
- 最终输出 = episode 结束时的 current_delay。
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ---------------- Config ----------------
BASE_TRAIN = r"C:\Users\24495\Desktop\articles\article_3\experiment\data\train data"
BASE_TEST  = r"C:\Users\24495\Desktop\articles\article_3\experiment\data\test data"

TRAIN_TX = os.path.join(BASE_TRAIN, "train_transactions.csv")
TEST_TX  = os.path.join(BASE_TEST,  "test_transactions.csv")

DEBUG = False
NORMALIZE = False
MAX_CLAIMS = None

# 动作空间
N_ACTIONS = 7
#ACTION_DELTAS = np.array([-0.25, -0.10, 0.0, 0.10, 0.25], dtype=np.float32)
ACTION_DELTAS = np.array([-0.50, -0.20, -0.05, 0.0, 0.05, 0.20, 0.50], dtype=np.float32)

# ---------------- setldel 模糊化 ----------------
def fuzz_setldel(setldel, granularity=0.5):
    """
    向上取整到 granularity 的倍数。
    4.2 -> 4.5, 10.7 -> 11.0, 7.9 -> 8.0
    """
    return float(np.ceil(setldel / granularity) * granularity)


# ---------------- 数据加载 ----------------
def load_claims(tx_path, max_claims=None):
    """
    返回一个 list，每个元素是一个 dict：
    {
        "claim_no": str,
        "claim_size": float,
        "notidel": float,
        "setldel": float,
        "setldel_sim": float,
        "total_transactions": int,
        "payments_true": np.array,
        "delays_true": np.array,
        "target_last_delay": float,
    }
    """
    df = pd.read_csv(tx_path)
    df["claim_no"] = df["claim_no"].astype(str)

    claims = []
    for claim_no, g in df.groupby("claim_no", sort=False):
        g = g.sort_values("pmt_no").reset_index(drop=True)

        claim_size = float(g["claim_size"].iloc[0])
        notidel = float(g["notidel"].iloc[0])
        setldel = float(g["setldel"].iloc[0])
        setldel_sim = fuzz_setldel(setldel, granularity=0.5)

        payments_true = g["payment_size"].astype(float).to_numpy()
        delays_true = g["payment_delay"].astype(float).to_numpy()

        claims.append({
            "claim_no": claim_no,
            "claim_size": claim_size,
            "notidel": notidel,
            "setldel": setldel,
            "setldel_sim": setldel_sim,
            "total_transactions": len(g),
            "payments_true": payments_true,
            "delays_true": delays_true,
            "target_last_delay": float(delays_true[-1]),
        })

        if max_claims is not None and len(claims) >= max_claims:
            break

    return claims


# ---------------- 环境 ----------------
class ClaimDelayEnv(gym.Env):

    def __init__(self, claims, normalize=False, state_mean=None, state_std=None):
        super().__init__()
        self.claims = claims
        self.normalize = normalize
        self.state_mean = state_mean
        self.state_std = state_std

        self.action_space = spaces.Discrete(N_ACTIONS)

        # state 维度：
        # t, total_transactions, current_delay, remaining_capacity,
        # true_delay_prev, pred_delay_prev, delay_gap_prev,
        # claim_size, notidel, setldel_sim, last_payment_size
        self.state_dim = 11

        obs_low = np.full(self.state_dim, -1e12, dtype=np.float32)
        obs_high = np.full(self.state_dim, 1e12, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        self._current_claim = None

    # ---------- reset ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx = np.random.randint(len(self.claims))
        self._current_claim = self.claims[idx]
        self._reset_internal()
        return self._get_state(), {}

    def _reset_internal(self):
        c = self._current_claim
        self.t = 0
        self.total_transactions = c["total_transactions"]
        self.claim_size = c["claim_size"]
        self.notidel = c["notidel"]
        self.setldel_sim = c["setldel_sim"]
        self.payments_true = c["payments_true"]
        self.delays_true = c["delays_true"]
        self.target_last_delay = c["target_last_delay"]

        # 方案 B：第一笔从 0 开始，之后每步设为上一笔真实 delay
        if self.total_transactions == 1:
            self.current_delay = float(self.setldel_sim)
        else:
            self.current_delay = 0.0
        self.true_delay_prev = 0.0
        self.pred_delay_prev = 0.0
        self.delay_gap_prev = 0.0
        self.last_payment_size = 0.0

        self.remaining_capacity = self.total_transactions
        self.done = False

    # ---------- state ----------
    def _get_state(self):
        state = np.array([
            float(self.t),
            float(self.total_transactions),
            float(self.current_delay),
            float(self.remaining_capacity),
            float(self.true_delay_prev),
            float(self.pred_delay_prev),
            float(self.delay_gap_prev),
            float(self.claim_size),
            float(self.notidel),
            float(self.setldel_sim),
            float(self.last_payment_size),
        ], dtype=np.float32)

        if self.normalize and self.state_mean is not None and self.state_std is not None:
            state = (state - self.state_mean) / self.state_std

        return state

    # ---------- step ----------
    def step(self, action):
        assert not self.done, "step() called after episode done"
        if self.total_transactions == 1:
            true_delay_t = float(self.delays_true[0])
            pred_delay_t = float(self.setldel_sim)
            denom = max(true_delay_t, 0.1)
            reward = -abs(pred_delay_t - true_delay_t) / (denom + 1e-8)
            self.pred_delay_prev = pred_delay_t
            self.true_delay_prev = true_delay_t
            self.delay_gap_prev = true_delay_t - pred_delay_t
            self.t = 1
            self.remaining_capacity = 0
            self.done = True
            info = {"true_delay_t": true_delay_t, "pred_delay_t": pred_delay_t, "delta": 0.0}
            return self._get_state(), float(reward), True, False, info
        action = int(action)
        delta = float(ACTION_DELTAS[action])

        # 当期真实 delay
        true_delay_t = float(self.delays_true[self.t])

        # 当期预测 delay：当前估计 + 调整量
        pred_delay_t = self.current_delay + delta
        pred_delay_t = max(pred_delay_t, 0.0)

        # ---------- reward（你手动修改） ----------
        # 默认用 setldel_sim 归一化，你可以改成 true_delay_t 或其他
        denom = max(true_delay_t, 0.1)
        reward = -abs(pred_delay_t - true_delay_t) / (denom + 1e-8)
        # ------------------------------------------

        # 更新 prev
        self.true_delay_prev = true_delay_t
        self.pred_delay_prev = pred_delay_t
        self.delay_gap_prev = true_delay_t - pred_delay_t
        self.last_payment_size = float(self.payments_true[self.t])

        # 方案 B：下一笔的起点 = 当前笔真实 delay
        self.current_delay = true_delay_t

        # 推进时间
        self.t += 1
        self.remaining_capacity = self.total_transactions - self.t

        if self.t >= self.total_transactions:
            self.done = True

        info = {
            "true_delay_t": true_delay_t,
            "pred_delay_t": pred_delay_t,
            "delta": delta,
        }

        return self._get_state(), float(reward), self.done, False, info

    # ---------- 最终预测 ----------
    def get_final_prediction(self):
        """
        episode 结束后调用，返回 agent 对最后一笔 delay 的预测。
        """
        return float(self.pred_delay_prev)


# ---------------- 快速测试 ----------------
if __name__ == "__main__":
    claims = load_claims(TRAIN_TX, max_claims=10 if DEBUG else MAX_CLAIMS)
    print(f"Loaded {len(claims)} claims")

    env = ClaimDelayEnv(claims, normalize=NORMALIZE)
    obs, _ = env.reset()
    print("Initial state:", obs)

    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        print(f"t={env.t-1} action={action} reward={reward:.4f} info={info}")

    print("Final prediction:", env.get_final_prediction())
    print("Target:", env.target_last_delay)
    print("Total reward:", total_reward)