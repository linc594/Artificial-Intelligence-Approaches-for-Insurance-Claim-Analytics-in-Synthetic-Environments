import os
import pandas as pd
import numpy as np
import chainladder as cl
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === 数据路径 ===
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

def build_triangle(df: pd.DataFrame):
    """从 transactions 原始数据生成链梯三角形数据"""
    # 确定 development_period
    if "development_period" not in df.columns:
        delay_col = "payment_delay" if "payment_delay" in df.columns else "payment_time"
        df["development_period"] = np.floor(df[delay_col]).astype(int)

    # 聚合为 origin × development 表
    tri = (
        df.groupby(["occurrence_period", "development_period"])["payment_size"]
        .sum()
        .reset_index()
        .rename(columns={
            "occurrence_period": "origin",
            "development_period": "development",
            "payment_size": "values"
        })
    )

    # 生成日期以满足 chainladder 的 development_date 要求
    base_date = pd.Timestamp("2000-01-01")
    tri["development_date"] = [
        base_date + pd.DateOffset(months=int(x)) for x in tri["development"]
    ]

    return tri


for scale, files in datasets.items():
    print(f"\n==================== {scale.upper()} Dataset ====================")

    # 读取 transactions
    train_tx = pd.read_csv(os.path.join(base_train, files["train_tx"]))
    test_tx = pd.read_csv(os.path.join(base_test, files["test_tx"]))

    # 构建三角形数据
    tri_train = build_triangle(train_tx)
    tri_test = build_triangle(test_tx)

    # 转为 Chainladder.Triangle 对象
    train_triangle = cl.Triangle(
        data=tri_train,
        origin="origin",
        development="development_date",
        columns="values"
    )

    test_triangle = cl.Triangle(
        data=tri_test,
        origin="origin",
        development="development_date",
        columns="values"
    )

    # 训练 Chain-Ladder
    cl_model = cl.Chainladder().fit(train_triangle)

    # 预测完整三角
    full_triangle = cl_model.full_triangle_

    # === 支付过程建模评估：看预测 vs 实际累计支付 ===
    paid_actual = np.nansum(train_triangle.values) + np.nansum(test_triangle.values)
    paid_pred = np.nansum(full_triangle.values)

    mse = (paid_actual - paid_pred) ** 2
    mae = np.abs(paid_actual - paid_pred)
    risk_ratio = paid_pred / (paid_actual + 1e-8)

    print(f"Chain-Ladder Payment Modeling → "
          f"MSE={mse:,.2f}, MAE={mae:,.2f}, Risk Ratio={risk_ratio:.4f}")

    results.append({
        "Dataset": scale,
        "CL_MSE": mse,
        "CL_MAE": mae,
        "CL_RiskRatio": risk_ratio
    })

# 保存结果
results_df = pd.DataFrame(results)
save_path = os.path.join(output_dir, "baseline_payment_chainladder_results.csv")
results_df.to_csv(save_path, index=False)

print("\n✅ Payment process baseline completed. Results saved to:")
print(save_path)
print(results_df)
