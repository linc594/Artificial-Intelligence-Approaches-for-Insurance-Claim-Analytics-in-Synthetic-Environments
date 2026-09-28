import os
import pandas as pd
import numpy as np
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 路径设置 ===
base_train = r"C:\Users\24495\Desktop\experiment\data\train data"
base_test = r"C:\Users\24495\Desktop\experiment\data\test data"
output_dir = r"C:\Users\24495\Desktop\experiment\data"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "large": {
        "train_claims": "train_claims.csv",
        "train_tx": "train_transactions.csv",
        "test_claims": "test_claims.csv",
        "test_tx": "test_transactions.csv"
    }
}

results = []

# === 主循环 ===
for scale, files in datasets.items():
    print(f"\n==================== {scale.upper()} Dataset ====================")

    # ---------- 1. 读取数据 ----------
    train_claims = pd.read_csv(os.path.join(base_train, files["train_claims"]))
    test_claims = pd.read_csv(os.path.join(base_test, files["test_claims"]))

    id_col = "claim_no"

    # ---------- 2. Tweedie GLM baseline ----------
    print("Running Tweedie GLM baseline...")
    feature_cols = [c for c in train_claims.columns if c not in [id_col, "claim_size"]]
    X_train = pd.get_dummies(train_claims[feature_cols], drop_first=True)
    X_test = pd.get_dummies(test_claims[feature_cols], drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    y_train = train_claims["claim_size"]
    y_test = test_claims["claim_size"]

    model = TweedieRegressor(power=1.5, alpha=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse_glm = mean_squared_error(y_test, y_pred)
    mae_glm = mean_absolute_error(y_test, y_pred)
    risk_ratio_glm = y_pred.sum() / (y_test.sum() + 1e-8)

    print(f"Tweedie GLM: MSE={mse_glm:.2f}, MAE={mae_glm:.2f}, Risk Ratio={risk_ratio_glm:.4f}")

    # ---------- 3. Aggregate-level comparison ----------
    print("\nPerforming aggregate-level comparison...")

    # GLM aggregate: 按事故期求和
    agg_tweedie = (
        test_claims.assign(true_claim_size=y_test, pred_claim_size=y_pred)
        .groupby("occurrence_period")[["true_claim_size", "pred_claim_size"]]
        .sum()
        .reset_index()
    )
    agg_mse_glm = mean_squared_error(agg_tweedie["true_claim_size"], agg_tweedie["pred_claim_size"])
    agg_mae_glm = mean_absolute_error(agg_tweedie["true_claim_size"], agg_tweedie["pred_claim_size"])
    agg_risk_glm = agg_tweedie["pred_claim_size"].sum() / (agg_tweedie["true_claim_size"].sum() + 1e-8)

    print(f"Aggregate Comparison Completed: Tweedie MSE={agg_mse_glm:.2f}, "
          f"MAE={agg_mae_glm:.2f}, Risk Ratio={agg_risk_glm:.4f}")

    # ---------- 4. 汇总结果 ----------
    results.append({
        "Dataset": scale,
        "GLM_MSE": mse_glm,
        "GLM_MAE": mae_glm,
        "GLM_RiskRatio": risk_ratio_glm,
        "GLM_MSE_Agg": agg_mse_glm,
        "GLM_MAE_Agg": agg_mae_glm,
        "GLM_RiskRatio_Agg": agg_risk_glm,
    })


    # ---------- 保存 individual-level 预测结果 ----------
    pred_df = pd.DataFrame({
    "claim_no": test_claims[id_col],
    "y_true": y_test.values,
    "y_pred": y_pred
    })

    pred_save_path = os.path.join(
    output_dir,
    f"glm_predictions_{scale}.csv"
    )

    pred_df.to_csv(pred_save_path, index=False)
    print(f"Saved GLM predictions to: {pred_save_path}")

#=== 5. 保存所有结果 ===
results_df = pd.DataFrame(results)
results_path = os.path.join(output_dir, "baseline_glm_results.csv")
results_df.to_csv(results_path, index=False)

print("\n✅ Claim baseline completed (Chain-Ladder removed). Results saved to:")
print(results_path)
print(results_df)
