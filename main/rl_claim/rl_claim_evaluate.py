import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rl_claim_env_predictor import ClaimSettlementEnv

# ====== PATHS ======
OUTPUT_METRICS = r"C:\Users\24495\Desktop\experiment\data\rl_claim_results.csv"
MODEL_PATH = r"C:\Users\24495\Desktop\experiment\ppo_claim_predictor_model"
TEST_CLAIMS_FILE = r"C:\Users\24495\Desktop\experiment\data\test data\test_claims.csv"
TEST_TRANSACTIONS_FILE = r"C:\Users\24495\Desktop\experiment\data\test data\test_transactions.csv"

# =========================================================
# STEP 1 — RUN RL POLICY
# =========================================================
def evaluate_policy():
    claims = pd.read_csv(TEST_CLAIMS_FILE)
    claims["claim_no"] = claims["claim_no"].astype(str)

    tx = pd.read_csv(TEST_TRANSACTIONS_FILE)
    tx["claim_no"] = tx["claim_no"].astype(str)

    print("Loading PPO model from:", MODEL_PATH)
    model = PPO.load(MODEL_PATH, device="cpu")

    claim_no_list = []          # ✅ 新增
    rl_paid_list = []
    true_claim_size_list = []

    # Evaluate each claim independently
    for claim_no in claims["claim_no"].unique():
        claim_row = claims[claims["claim_no"] == claim_no].iloc[0]
        tx_rows = tx[tx["claim_no"] == claim_no]

        env = ClaimSettlementEnv(
            claim_row=claim_row,
            tx_row=tx_rows,
            episode_length=10,
            partial_pay_ratio=0.2,
        )

        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        claim_no_list.append(claim_no)                     # ✅ 新增
        rl_paid_list.append(env.cumulative_paid)
        true_claim_size_list.append(claim_row["claim_size"])

    return (
        np.array(claim_no_list, dtype=str),
        np.array(rl_paid_list, dtype=float),
        np.array(true_claim_size_list, dtype=float),
    )


# =========================================================
# STEP 2 — EVALUATE METRICS
# =========================================================
def evaluate_metrics(pred_pay, true_pay):
    mse = mean_squared_error(true_pay, pred_pay)
    mae = mean_absolute_error(true_pay, pred_pay)
    rr = float(np.sum(pred_pay)) / (float(np.sum(true_pay)) + 1e-8)

    print("\n===== RL CLAIM COST EVALUATION =====")
    print(f"Claim_MSE : {mse:.4f}")
    print(f"Claim_MAE : {mae:.4f}")
    print(f"Claim_RR  : {rr:.4f} (<1 成本更低, >1 成本更高)")
    print("====================================")

    # ----------- 保存为 rl_claim_results.csv（追加模式）------------
    df_new = pd.DataFrame([{
        "Claim_MSE": mse,
        "Claim_MAE": mae,
        "Claim_RR": rr
    }])

    if not os.path.exists(OUTPUT_METRICS):
        df_new.to_csv(OUTPUT_METRICS, index=False)
        print("Created →", OUTPUT_METRICS)
    else:
        df_new.to_csv(OUTPUT_METRICS, index=False, mode="a", header=False)
        print("Appended →", OUTPUT_METRICS)


# =========================================================
# MAIN ENTRY
# =========================================================
if __name__ == "__main__":
    claim_nos, pred_pay, true_pay = evaluate_policy()

    # -------- 保存 claim-level 预测结果（small）--------
    rl_claim_pred_df = pd.DataFrame({
        "claim_no": claim_nos,
        "y_true": true_pay,
        "y_pred": pred_pay
    })

    large_pred_path = r"C:\Users\24495\Desktop\experiment\data\rl_claim_predictions.csv"
    rl_claim_pred_df.to_csv(large_pred_path, index=False)
    print("Saved RL claim predictions (large) to:", large_pred_path)

    evaluate_metrics(pred_pay, true_pay)



