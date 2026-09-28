"""
ClaimPaymentEnv
---------------
每个 episode 对应一个 claim。
agent 逐期决定当期支付多少，目标是让最后一期预测支付
尽可能接近真实最后一期支付。

设计要点：
- 只加载 transactions 文件，不加载 claims 文件；
- claim_size / notidel / setldel 从组内第一行取；
- state 里同时包含真实轨迹和预测轨迹的历史值；
- 当期的 true_payment_t 只在 reward 里使用，不进 state；
- 最后一期预测支付 = claim_size - sum(前 t-1 期预测支付)。
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

# 调试开关
DEBUG = False
NORMALIZE = False
MAX_CLAIMS = None   # 例如 1000，用于快速调试

# 动作空间
N_ACTIONS = 10
ACTION_RATIOS = np.linspace(0.0, 1.0, N_ACTIONS)  # [0, 0.111, ..., 1.0]


# ---------------- 数据加载 ----------------
def load_claims(tx_path, max_claims=None):
    """
    返回一个 list，每个元素是一个 dict：
    {
        "claim_no": str,
        "claim_size": float,
        "notidel": float,
        "setldel": float,
        "total_transactions": int,
        "payments_true": np.array,     # 每期真实 payment_size
        "delays_true": np.array,       # 每期真实 payment_delay
        "target_last_payment": float,  # 最后一期真实 payment_size
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

        payments_true = g["payment_size"].astype(float).to_numpy()
        delays_true = g["payment_delay"].astype(float).to_numpy()

        claims.append({
            "claim_no": claim_no,
            "claim_size": claim_size,
            "notidel": notidel,
            "setldel": setldel,
            "total_transactions": len(g),
            "payments_true": payments_true,
            "delays_true": delays_true,
            "target_last_payment": float(payments_true[-1]),
        })

        if max_claims is not None and len(claims) >= max_claims:
            break

    return claims

def compute_state_stats(claims):
    """
    遍历所有 claim，收集每个 step 的 state，计算均值和标准差。
    用于后续 state 标准化。
    """
    states = []
    for c in claims:
        # 模拟一遍 episode，收集 state
        env = ClaimPaymentEnv([c], normalize=False)
        obs, _ = env.reset()
        done = False
        while not done:
            states.append(obs)
            # 用固定 action（比如 1）推进，只为了收集 state
            obs, reward, done, trunc, info = env.step(1)
    states = np.array(states, dtype=np.float32)
    mean = states.mean(axis=0)
    std = states.std(axis=0) + 1e-8
    return mean, std

# ---------------- 环境 ----------------
class ClaimPaymentEnv(gym.Env):
    """
    每个 episode 对应一个 claim。
    """

    def __init__(self, claims, normalize=False, state_mean=None, state_std=None):
        super().__init__()
        self.claims = claims
        self.normalize = normalize
        self.state_mean = state_mean
        self.state_std = state_std

        self.action_space = spaces.Discrete(N_ACTIONS)

        # state 维度：
        # t, total_transactions, claim_size,
        # true_payment_prev, pred_payment_prev,
        # true_remaining_prev, pred_remaining_prev,
        # remaining_gap_prev,
        # current_delay, notidel, setldel
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
        #idx = self.np_random.integers(len(self.claims))
        idx = np.random.randint(len(self.claims))
        self._current_claim = self.claims[idx]
        self._reset_internal()
        return self._get_state(), {}

    def _reset_internal(self):
        c = self._current_claim
        self.t = 0                                  # 当前是第几期（0-indexed）
        self.total_transactions = c["total_transactions"]
        self.claim_size = c["claim_size"]
        self.notidel = c["notidel"]
        self.setldel = c["setldel"]
        self.payments_true = c["payments_true"]
        self.delays_true = c["delays_true"]
        self.target_last_payment = c["target_last_payment"]

        self.true_cumulative_paid = 0.0
        self.pred_cumulative_paid = 0.0

        self.true_remaining_prev = self.claim_size
        self.pred_remaining_prev = self.claim_size

        self.true_payment_prev = 0.0
        self.pred_payment_prev = 0.0

        self.done = False

    # ---------- state ----------
    def _get_state(self):
        c = self._current_claim

        remaining_gap_prev = self.true_remaining_prev - self.pred_remaining_prev

        current_delay = float(self.delays_true[self.t]) \
            if self.t < self.total_transactions else 0.0

        state = np.array([
            float(self.t),
            float(self.total_transactions),
            float(self.claim_size),
            float(self.true_payment_prev),
            float(self.pred_payment_prev),
            float(self.true_remaining_prev),
            float(self.pred_remaining_prev),
            float(remaining_gap_prev),
            float(current_delay),
            float(self.notidel),
            float(self.setldel),
        ], dtype=np.float32)

        if self.normalize and self.state_mean is not None and self.state_std is not None:
            state = (state - self.state_mean) / self.state_std

        return state

    # ---------- step ----------
    def step(self, action):
        assert not self.done, "step() called after episode done"

        action = int(action)
        ratio = float(ACTION_RATIOS[action])

        # 当期真实值
        true_payment_t = float(self.payments_true[self.t])

        # 当期预测支付：基于 pred_remaining 的比例
        #pred_payment_t = float(self.pred_remaining_prev * ratio)
        pred_payment_t = float(self.true_remaining_prev * ratio)
        # 如果这是最后一期，预测支付自动等于 pred_remaining
        is_last = (self.t == self.total_transactions - 1)
        if is_last:
            pred_payment_t = float(self.pred_remaining_prev)

        # 防止 pred_payment_t 超过 remaining
        pred_payment_t = min(pred_payment_t, self.pred_remaining_prev)
        pred_payment_t = max(pred_payment_t, 0.0)

        # 更新轨迹
        self.true_cumulative_paid += true_payment_t
        self.pred_cumulative_paid += pred_payment_t

        new_true_remaining = self.claim_size - self.true_cumulative_paid
        new_pred_remaining = self.claim_size - self.pred_cumulative_paid

        # reward
        payment_error = abs(pred_payment_t - true_payment_t) / (self.claim_size + 1e-8)

        remaining_error = abs(
            new_pred_remaining - new_true_remaining
            ) / (self.claim_size + 1e-8)

        reward = -(
            2 *payment_error
            + 0 * remaining_error
        )
        # 更新 prev 变量
        self.true_payment_prev = true_payment_t
        self.pred_payment_prev = pred_payment_t
        self.true_remaining_prev = new_true_remaining
        self.pred_remaining_prev = new_pred_remaining

        # 推进时间
        self.t += 1

        # termination
        if self.t >= self.total_transactions:
            self.done = True


        info = {
            "true_payment_t": true_payment_t,
            "pred_payment_t": pred_payment_t,
            "true_remaining": new_true_remaining,
            "pred_remaining": new_pred_remaining,
            "is_last": is_last,
        }

        return self._get_state(), float(reward), self.done, False, info

    # ---------- 最终预测 ----------
    def get_final_prediction(self):
        """
        episode 结束后调用。
        返回 agent 对最后一期 payment 的预测。
        """
        return float(self.pred_payment_prev) if self.t >= self.total_transactions else None


# ---------------- 快速测试 ----------------
if __name__ == "__main__":
    claims = load_claims(TRAIN_TX, max_claims=10 if DEBUG else MAX_CLAIMS)
    print(f"Loaded {len(claims)} claims")

    env = ClaimPaymentEnv(claims, normalize=NORMALIZE)
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
    print("Target:", env.target_last_payment)
    print("Total reward:", total_reward)