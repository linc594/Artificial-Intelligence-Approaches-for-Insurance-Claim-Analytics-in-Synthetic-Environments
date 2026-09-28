# ==============================
# train_claim_predictor.py
# ==============================

import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

TRAIN_CLAIMS = r"C:\Users\24495\Desktop\experiment\data\train data\train_claims.csv"
SEED = 46
claims = pd.read_csv(TRAIN_CLAIMS)

# ==== 你实际使用的特征列（按你的数据情况修改）====
feature_cols = [c for c in claims.columns if c not in ["claim_no", "claim_size"]]

X = claims[feature_cols]
y = claims["claim_size"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED
)
model.fit(X_train, y_train)

# ==== 保存预测器 ====
joblib.dump(model, "claim_predictor.pkl")
print("Saved claim_predictor.pkl")

# ==== 计算残差标准差（用于 RL 环境的不确定性）====
pred_train = model.predict(X_train)
residuals = y_train - pred_train
sigma = float(np.std(residuals))

joblib.dump({"sigma": sigma, "feature_cols": feature_cols}, "predictor_meta.pkl")
print("Saved predictor_meta.pkl (sigma, feature_cols)")


