"""
在 test claims 上评估 ClaimPaymentEnv 的 PPO policy。
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rl_claim_payment_env import ( 
    ClaimPaymentEnv, load_claims,
    TEST_TX, DEBUG, NORMALIZE, MAX_CLAIMS
)

MODEL_PATH = r"C:\Users\24495\Desktop\experiment\data\ppo_claim_payment_model.zip"
OUTPUT_PRED = r"C:\Users\24495\Desktop\experiment\data\rl_claim_payment_predictions.csv"
OUTPUT_METRICS = r"C:\Users\24495\Desktop\experiment\data\rl_claim_payment_results.csv"


def evaluate():
    claims = load_claims(TEST_TX, max_claims=50 if DEBUG else MAX_CLAIMS)
    print(f"Evaluating on {len(claims)} test claims")

    model = PPO.load(MODEL_PATH, device="cpu")

    records = []
    for c in claims:
        env = ClaimPaymentEnv([c], normalize=NORMALIZE)
        obs, _ = env.reset()

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(int(action))

        pred_last = env.get_final_prediction()
        true_last = c["target_last_payment"]

        records.append({
            "claim_no": c["claim_no"],
            "total_transactions": c["total_transactions"],
            "claim_size": c["claim_size"],
            "y_true": true_last,
            "y_pred": pred_last,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PRED, index=False)
    print("Saved predictions ->", OUTPUT_PRED)

    # metrics
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rr = float(y_pred.sum() / (y_true.sum() + 1e-8))

    print("\n===== RL Claim Payment Evaluation =====")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RR : {rr:.6f}")

    # 分组：单笔 vs 多笔
    single = df[df["total_transactions"] == 1]
    multi = df[df["total_transactions"] > 1]

    def _metrics(sub):
        if len(sub) == 0:
            return None
        yt = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        return {
            "n": len(sub),
            "MSE": float(np.mean((yt - yp) ** 2)),
            "MAE": float(np.mean(np.abs(yt - yp))),
            "RR": float(yp.sum() / (yt.sum() + 1e-8)),
        }

    print("\n--- Single-transaction claims ---")
    print(_metrics(single))
    print("\n--- Multi-transaction claims ---")
    print(_metrics(multi))

    # save metrics
    metrics_df = pd.DataFrame([{
        "MSE": mse, "MAE": mae, "RR": rr,
        "MSE_single": _metrics(single)["MSE"] if len(single) else None,
        "MSE_multi": _metrics(multi)["MSE"] if len(multi) else None,
        "MAE_single": _metrics(single)["MAE"] if len(single) else None,
        "MAE_multi": _metrics(multi)["MAE"] if len(multi) else None,
        "RR_single": _metrics(single)["RR"] if len(single) else None,
        "RR_multi": _metrics(multi)["RR"] if len(multi) else None,
    }])
    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print("Saved metrics ->", OUTPUT_METRICS)


if __name__ == "__main__":
    evaluate()