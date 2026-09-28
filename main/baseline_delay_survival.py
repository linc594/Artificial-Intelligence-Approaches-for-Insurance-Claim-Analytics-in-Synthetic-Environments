import os
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 路径设置 ===
base_train = r"C:\Users\24495\Desktop\experiment\data\train data"
base_test = r"C:\Users\24495\Desktop\experiment\data\test data"
output_dir = r"C:\Users\24495\Desktop\experiment\data"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "large": {
        "train_tx": "train_transactions.csv",
        "test_tx": "test_transactions.csv"
    }
}

results = []

for scale, files in datasets.items():
    print(f"\n==================== {scale.upper()} Dataset ====================")

    train_tx = pd.read_csv(os.path.join(base_train, files["train_tx"]))
    test_tx = pd.read_csv(os.path.join(base_test, files["test_tx"]))

    train_tx["event"] = 1
    test_tx["event"] = 1

    y_true = test_tx["payment_delay"].values

    # ==============================================================
    # ================= Kaplan–Meier ===============================
    # ==============================================================
    print("Running Kaplan–Meier baseline...")
    km = KaplanMeierFitter()
    km.fit(train_tx["payment_delay"], event_observed=train_tx["event"])

    km_median = km.median_survival_time_
    y_pred_km = np.full_like(y_true, km_median, dtype=float)

    mse_km = mean_squared_error(y_true, y_pred_km)
    mae_km = mean_absolute_error(y_true, y_pred_km)
    rr_km = y_pred_km.sum() / (y_true.sum() + 1e-8)

    km_out = test_tx[["claim_no", "pmt_no", "payment_period"]].copy()
    km_out["payment_delay_true"] = y_true
    km_out["payment_delay_pred"] = y_pred_km
    km_out.to_csv(
        os.path.join(output_dir, "baseline_km_predictions.csv"),
        index=False
    )

    # ==============================================================
    # ================= Cox PH =====================================
    # ==============================================================
    print("Running Cox PH baseline...")
    feature_cols = ["occurrence_period"]

    cph_df_train = train_tx[["payment_delay", "event"] + feature_cols]
    cph_df_test = test_tx[["payment_delay", "event"] + feature_cols]

    cph = CoxPHFitter()
    cph.fit(cph_df_train, duration_col="payment_delay", event_col="event")

    baseline_ch = cph.baseline_cumulative_hazard_
    baseline_survival = np.exp(-baseline_ch)
    baseline_time = baseline_ch.index.values
    baseline_expected = np.trapz(
        baseline_survival.values.flatten(),
        baseline_time
    )

    lin_pred = cph.predict_partial_hazard(cph_df_test).values.flatten()
    y_pred_cox = baseline_expected / lin_pred

    mse_cox = mean_squared_error(y_true, y_pred_cox)
    mae_cox = mean_absolute_error(y_true, y_pred_cox)
    rr_cox = y_pred_cox.sum() / (y_true.sum() + 1e-8)

    cox_out = test_tx[["claim_no", "pmt_no", "payment_period"]].copy()
    cox_out["payment_delay_true"] = y_true
    cox_out["payment_delay_pred"] = y_pred_cox
    cox_out.to_csv(
        os.path.join(output_dir, "baseline_cox_predictions.csv"),
        index=False
    )

    # ==============================================================
    # ================= Weibull AFT ================================
    # ==============================================================
    print("Running Weibull AFT baseline...")
    aft = WeibullAFTFitter()
    aft.fit(cph_df_train, duration_col="payment_delay", event_col="event")

    y_pred_weibull = aft.predict_expectation(cph_df_test).values

    mse_weibull = mean_squared_error(y_true, y_pred_weibull)
    mae_weibull = mean_absolute_error(y_true, y_pred_weibull)
    rr_weibull = y_pred_weibull.sum() / (y_true.sum() + 1e-8)

    weibull_out = test_tx[["claim_no", "pmt_no", "payment_period"]].copy()
    weibull_out["payment_delay_true"] = y_true
    weibull_out["payment_delay_pred"] = y_pred_weibull
    weibull_out.to_csv(
        os.path.join(output_dir, "baseline_weibull_predictions.csv"),
        index=False
    )

    # ==============================================================
    # ================= 汇总指标 ==================================
    # ==============================================================
    results.append({
        "Dataset": scale,

        "KM_MSE": mse_km,
        "KM_MAE": mae_km,
        "KM_RR": rr_km,

        "Cox_MSE": mse_cox,
        "Cox_MAE": mae_cox,
        "Cox_RR": rr_cox,

        "Weibull_MSE": mse_weibull,
        "Weibull_MAE": mae_weibull,
        "Weibull_RR": rr_weibull
    })

# ===== 保存汇总结果 =====
results_df = pd.DataFrame(results)
save_path = os.path.join(output_dir, "baseline_delay_survival_results.csv")
results_df.to_csv(save_path, index=False)

print("\n✅ Delay Modeling Baseline completed.")
print(results_df)
print("Saved summary ->", save_path)

