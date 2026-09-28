# modified_lstm_full_fixed_mlp.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------- CONFIG --------------------
BASE_TRAIN = r"C:\Users\24495\Desktop\experiment\data\train data"
BASE_TEST  = r"C:\Users\24495\Desktop\experiment\data\test data"
OUTPUT_CSV = r"C:\Users\24495\Desktop\experiment\data\baseline_mlp_results.csv"
OUTPUT_DIR = r"C:\Users\24495\Desktop\experiment\data"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CLAIM_WINDOW = 1
K_CLAIMS_PAYMENT = 1
MAX_STEPS = 20
LOG_CLIP_MIN, LOG_CLIP_MAX = -8, 12

# -------------------- RUN SWITCHES --------------------
# 控制是否运行各部分（True = 运行 / False = 跳过）
RUN_PARTS = {
    "claim": True,     # Claim 部分开关
    "payment": False,  # Payment 部分开关
    "delay":    False     # Delay 部分开关
}

# -------------------- LSTM/MLP PARAMETERS --------------------
# You can keep using the same dict names; this code will interpret them for MLP.
CLAIM_MLP_PARAMS = {
    "input_size": None,                  # 自动在训练时确定
    "hidden_layers": [[128,64,64]],  # list of list 支持调参
    "dropout": 0.0,
    "batch_size": 256,
    "epochs": 200,
    "lr": [5e-4],                 # list 支持调参
    "patience": 10,
}

PAYMENT_MLP_PARAMS = {
    "input_size": None,
    "hidden_layers": [[64]],        # list of list
    "dropout": 0.0,
    "batch_size": 256,
    "epochs": 200,
    "lr": 1e-3,
    "patience": 5,
}

DELAY_MLP_PARAMS = {
    "input_size": None,
    "hidden_layers": [[64]],      # list of list
    "dropout": 0.0,
    "batch_size": 256,
    "epochs": 200,
    "lr": 1e-3,
    "patience": 5,
}
# -------------------- UTIL --------------------
def safe_expm1_clip(x, min_log=-5.0, max_log=15.0):
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, min_log, max_log)
    out = np.expm1(arr)
    med = float(np.nanmedian(out)) if np.isfinite(out).any() else 0.0
    mask_finite = np.isfinite(out)
    out = np.where(mask_finite, out, med)
    return np.maximum(out, 0.0)

# -------------------- MODEL --------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layers=[128, 64], dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # 输出层
        self.model = nn.Sequential(*layers)

    def forward(self, x, lengths=None):
        # lengths 参数保留接口一致，但不使用
        x_flat = x.view(x.size(0), -1)  # flatten 所有特征
        return self.model(x_flat).squeeze(-1)

# -------------------- DATALOADER --------------------
def build_dataloader_from_arrays(X, lengths, y, batch_size=256, shuffle=True):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_seq_model(model, optimizer, loss_fn, train_loader, val_loader, epochs=40, patience=6):
    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        tr_losses = []
        for Xb, lb, yb in train_loader:
            Xb, lb, yb = Xb.to(DEVICE), lb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(Xb, lb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xv, lv, yv in val_loader:
                Xv, lv, yv = Xv.to(DEVICE), lv.to(DEVICE), yv.to(DEVICE)
                val_losses.append(loss_fn(model(Xv, lv), yv).item())
        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        print(f"Epoch {ep+1}/{epochs} train_loss={np.mean(tr_losses):.6f} val_loss={mean_val:.6f}")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# --- 打印结果到终端（紧凑表格版） ---
def print_results_compact(results):
    print("\nFinal Results Summary:")
    header = f"{'Dataset':<8} | {'Claim-MSE':>12} | {'Claim-MAE':>12} | {'Claim-RR':>8} | " \
             f"{'Pay-MSE':>12} | {'Pay-MAE':>12} | {'Pay-RR':>8} | " \
             f"{'Delay-MSE':>12} | {'Delay-MAE':>12} | {'Delay-RR':>8}"
    print(header)
    print("-"*len(header))
    for r in results:
        def fmt(x): return f"{x:12.4f}" if pd.notna(x) else f"{'-':>12}"
        print(f"{r['Dataset']:<8} | {fmt(r['Claim_MSE'])} | {fmt(r['Claim_MAE'])} | {fmt(r['Claim_RR'])} | "
              f"{fmt(r['Pay_MSE'])} | {fmt(r['Pay_MAE'])} | {fmt(r['Pay_RR'])} | "
              f"{fmt(r['Delay_MSE'])} | {fmt(r['Delay_MAE'])} | {fmt(r['Delay_RR'])}")
    print("-"*len(header))

# -------------------- CLAIM SEQUENCES --------------------
def build_claim_sequences(claims_df, window=CLAIM_WINDOW):
    num_cols = claims_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["claim_no","claim_size"]]
    X_all = claims_df[num_cols].astype(float)
    y_all = claims_df["claim_size"].astype(float).values
    X_seqs, y_targets = [], []
    N = len(claims_df)
    for i in range(0, N - window):
        X_seqs.append(X_all.iloc[i:i+window].values)
        y_targets.append(y_all[i+window])
    if len(X_seqs)==0:
        return np.zeros((0, window, X_all.shape[1])), np.array([], dtype=float), X_all.columns.tolist()
    return np.stack(X_seqs, axis=0).astype(float), np.array(y_targets, dtype=float), X_all.columns.tolist()

# -------------------- PAYMENT/DELAY WINDOWS --------------------
def build_claim_to_tx_map(tx_df):
    tx_df = tx_df.copy()
    tx_df["claim_no"] = tx_df["claim_no"].astype(str)
    if "payment_time" in tx_df.columns:
        tx_df = tx_df.sort_values(["claim_no","payment_time"])
    else:
        tx_df = tx_df.sort_values(["claim_no","occurrence_period","payment_period"])
    return {k: g.reset_index(drop=True) for k,g in tx_df.groupby("claim_no", sort=False)}

def build_payment_delay_dataset_all_rows(df):
    """
    对每个 claim_no：
        - 所有记录都作为训练样本生成一条输入
        - 但它们的目标都等于该 claim_no 最后一条记录的 payment_size / payment_delay

    输入：df（transaction CSV）
    输出：X（N × D 特征矩阵）
          y_payment（最终支付目标）
          y_delay（最终延迟目标）
    """
    # 所有特征列（移除目标列）
    feat_cols = [c for c in df.columns if c not in ["payment_size", "payment_delay"]]

    X_list = []
    y_pay_list = []
    y_delay_list = []

    for claim_no, grp in df.groupby("claim_no"):

        # 按时间排序（确保最后一条是正确的）
        grp_sorted = grp.sort_values("pmt_no")

        # 最后一条记录是唯一的 target
        last_row = grp_sorted.iloc[-1]
        last_pay = last_row["payment_size"]
        last_delay = last_row["payment_delay"]

        # 所有记录 → 输入
        for _, row in grp_sorted.iterrows():
            X_list.append(row[feat_cols].values)
            y_pay_list.append(last_pay)
            y_delay_list.append(last_delay)

    X = np.array(X_list)
    y_payment = np.array(y_pay_list)
    y_delay = np.array(y_delay_list)

    return X, y_payment, y_delay


def construct_windows_by_claims(claims_df, tx_df, k_claims=K_CLAIMS_PAYMENT, max_steps=MAX_STEPS):
    claim_order = claims_df["claim_no"].astype(str).tolist()
    tx_map = build_claim_to_tx_map(tx_df)
    tx_global = tx_df.sort_values(["payment_time"] if "payment_time" in tx_df.columns else ["occurrence_period","payment_period"]).reset_index(drop=True)
    tx_global["_global_idx"] = tx_global.index
    for claim_no, g in tx_map.items():
        mask = (tx_global["claim_no"].astype(str) == str(claim_no))
        idxs = tx_global.loc[mask, "_global_idx"].values
        local = g.reset_index(drop=True).copy()
        local["_global_idx"] = list(idxs[:len(local)])
        tx_map[claim_no] = local

    X_seqs, lengths, y_dicts = [], [], []
    for i in range(0, len(claim_order)-k_claims+1):
        win_claims = claim_order[i:i+k_claims]
        parts = [tx_map[c] for c in win_claims if c in tx_map and len(tx_map[c])>0]
        if len(parts)==0: continue
        concat = pd.concat(parts, ignore_index=True).sort_values("_global_idx").reset_index(drop=True)
        last_idx = concat["_global_idx"].values[-1]
        next_idx = last_idx + 1
        if next_idx >= len(tx_global): continue
        target_row = tx_global.iloc[next_idx]
        seq_features = concat.select_dtypes(include=[np.number]).columns.tolist()
        seq_features = [f for f in seq_features if f!="_global_idx"]
        seq_mat = concat[seq_features].astype(float).values
        L = seq_mat.shape[0]
        if L>max_steps: seq_mat = seq_mat[-max_steps:]; L=max_steps
        pad_len = max_steps - L
        seq_padded = np.vstack([np.zeros((pad_len, seq_mat.shape[1])), seq_mat]) if pad_len>0 else seq_mat
        y_payment_size = float(target_row.get("payment_size", np.nan))
        y_payment_delay = float(target_row.get("payment_delay", np.nan))
        X_seqs.append(seq_padded.astype(float))
        lengths.append(L)
        y_dicts.append({"payment_size": y_payment_size, "payment_delay": y_payment_delay})
    if len(X_seqs)==0:
        return np.zeros((0,max_steps,len(seq_features))), np.array([],dtype=int), []
    return np.stack(X_seqs, axis=0), np.array(lengths, dtype=int), y_dicts

# -------------------- SCALER --------------------
def fit_scaler_on_seq(X_seq):
    N,T,D = X_seq.shape
    scaler = StandardScaler()
    scaler.fit(X_seq.reshape(-1,D))
    return scaler

def apply_scaler_seq(X_seq, scaler):
    N,T,D = X_seq.shape
    return scaler.transform(X_seq.reshape(-1,D)).reshape(N,T,D)

# -------------------- PIPELINE --------------------
def run_pipeline():
    datasets = {
        "large": {"claims":"train_claims.csv","tx":"train_transactions.csv"}
    }

    import itertools
    def ensure_list(x):
        return x if isinstance(x, list) else [x]

    # ---------------------- 构建参数组合 ----------------------
    claim_hidden_list = ensure_list(CLAIM_MLP_PARAMS["hidden_layers"])
    claim_lr_list     = ensure_list(CLAIM_MLP_PARAMS["lr"])
    claim_batch_list  = ensure_list(CLAIM_MLP_PARAMS["batch_size"])

    payment_hidden_list = ensure_list(PAYMENT_MLP_PARAMS["hidden_layers"])
    payment_lr_list     = ensure_list(PAYMENT_MLP_PARAMS["lr"])
    payment_batch_list  = ensure_list(PAYMENT_MLP_PARAMS["batch_size"])

    delay_hidden_list = ensure_list(DELAY_MLP_PARAMS["hidden_layers"])
    delay_lr_list     = ensure_list(DELAY_MLP_PARAMS["lr"])
    delay_batch_list  = ensure_list(DELAY_MLP_PARAMS["batch_size"])

    param_combinations = list(itertools.product(
        claim_hidden_list, claim_lr_list, claim_batch_list,
        payment_hidden_list, payment_lr_list, payment_batch_list,
        delay_hidden_list, delay_lr_list, delay_batch_list
    ))

    # ---------------------- 循环实验 ----------------------
    for (ch, clr, cb,
         ph, plr, pb,
         dh, dlr, db) in param_combinations:

        # 更新参数字典
        CLAIM_MLP_PARAMS.update({"hidden_layers": ch, "lr": clr, "batch_size": cb})
        PAYMENT_MLP_PARAMS.update({"hidden_layers": ph, "lr": plr, "batch_size": pb})
        DELAY_MLP_PARAMS.update({"hidden_layers": dh, "lr": dlr, "batch_size": db})

        results = []

        for scale, files in datasets.items():
            print(f"===== Running dataset: {scale} =====")

            mse_pay = mae_pay = rr_pay = np.nan
            mse_delay = mae_delay = rr_delay = np.nan

            train_claims = pd.read_csv(os.path.join(BASE_TRAIN, files["claims"]))
            test_claims  = pd.read_csv(os.path.join(BASE_TEST, files["claims"].replace("train","test")))
            train_tx = pd.read_csv(os.path.join(BASE_TRAIN, files["tx"]))
            test_tx  = pd.read_csv(os.path.join(BASE_TEST, files["tx"].replace("train","test")))

            for df in [train_claims, test_claims, train_tx, test_tx]:
                if "claim_no" in df.columns: df["claim_no"] = df["claim_no"].astype(str)

            # --- Claim-MLP ---
            print(f"Building claim sequences (window={CLAIM_WINDOW}) ...")
            Xc_train, yc_train, _ = build_claim_sequences(train_claims, CLAIM_WINDOW)
            Xc_test, yc_test = build_claim_sequences(test_claims, CLAIM_WINDOW)[:2]
            print(f"Claim seqs shapes: {Xc_train.shape} {yc_train.shape}")

            print("Training Claim-MLP...")
            if Xc_train.shape[0] > 0:
                scaler_claim = StandardScaler()
                scaler_claim.fit(Xc_train.reshape(-1, Xc_train.shape[2]))
                Xc_train_s = scaler_claim.transform(Xc_train.reshape(-1, Xc_train.shape[2])).reshape(Xc_train.shape)
                Xc_test_s  = scaler_claim.transform(Xc_test.reshape(-1, Xc_test.shape[2])).reshape(Xc_test.shape)
                yc_train_log = np.log1p(yc_train)
                yc_test_log  = np.log1p(yc_test)

                split = max(int(len(Xc_train_s)*0.9),1)
                X_tr, X_val = Xc_train_s[:split], Xc_train_s[split:]
                y_tr, y_val = yc_train_log[:split], yc_train_log[split:]

                tr_loader = build_dataloader_from_arrays(X_tr, np.ones(len(X_tr)), y_tr, batch_size=CLAIM_MLP_PARAMS["batch_size"])
                val_loader = build_dataloader_from_arrays(X_val, np.ones(len(X_val)), y_val, batch_size=max(8, CLAIM_MLP_PARAMS["batch_size"]//4), shuffle=False)
                test_loader = build_dataloader_from_arrays(Xc_test_s, np.ones(Xc_test_s.shape[0]), yc_test_log)

                model_claim = MLPRegressor(
                    input_dim=Xc_train_s.shape[1]*Xc_train_s.shape[2],
                    hidden_layers=CLAIM_MLP_PARAMS["hidden_layers"],
                    dropout=CLAIM_MLP_PARAMS["dropout"]
                ).to(DEVICE)
                opt_claim = torch.optim.Adam(model_claim.parameters(), lr=CLAIM_MLP_PARAMS["lr"])
                model_claim = train_seq_model(model_claim, opt_claim, nn.MSELoss(), tr_loader, val_loader,
                                              epochs=CLAIM_MLP_PARAMS["epochs"], patience=CLAIM_MLP_PARAMS["patience"])

                model_claim.eval()
                preds_claim_log = np.concatenate([model_claim(Xb.to(DEVICE), lb.to(DEVICE)).detach().cpu().numpy()
                                                  for Xb, lb, _ in test_loader]) if Xc_test_s.shape[0]>0 else np.array([])
                preds_claim = safe_expm1_clip(preds_claim_log)
                trues_claim = yc_test
                offset = CLAIM_WINDOW  # 当前是 1
                aligned_claim_nos = test_claims["claim_no"].iloc[offset : offset + len(trues_claim)].values
                claim_pred_df = pd.DataFrame({
                "claim_no": aligned_claim_nos,
                "y_true": trues_claim,
                "y_pred": preds_claim
                })

                claim_pred_path = os.path.join(
                OUTPUT_DIR,
                f"claim_mlp_predictions_{scale}.csv"
                )

                claim_pred_df.to_csv(claim_pred_path, index=False)
                print(f"[Claim] Saved aligned predictions to: {claim_pred_path}")
                mse_claim = mean_squared_error(trues_claim, preds_claim)
                mae_claim = mean_absolute_error(trues_claim, preds_claim)
                rr_claim  = preds_claim.sum()/(trues_claim.sum()+1e-8)
            else:
                mse_claim = mae_claim = rr_claim = np.nan

            # --- Payment/Delay MLP ---
            if scale == "large":
                print("Constructing payment/delay windows ...")

                # 如果两个部分都被禁用，直接跳过并设 nan（保持结果结构一致）
                if (not RUN_PARTS.get("payment", True)) and (not RUN_PARTS.get("delay", True)):
                    print("Payment and Delay parts are OFF by RUN_PARTS. Skipping.")
                    mse_pay = mae_pay = rr_pay = np.nan
                    mse_delay = mae_delay = rr_delay = np.nan
                else:
                    # Build datasets: every row is an input, but target is the last row's payment_size/delay
                    Xp_train, ypay_train, ydelay_train = build_payment_delay_dataset_all_rows(train_tx)
                    Xp_test,  ypay_test,  ydelay_test  = build_payment_delay_dataset_all_rows(test_tx)

                    print(f"Payment: train rows={Xp_train.shape[0]} test rows={Xp_test.shape[0]}")
                    print(f"Payment: ypay_train NaN ratio={(np.isnan(ypay_train).mean() if ypay_train.size>0 else 1.0):.3f}, "
                          f"ydelay_train NaN ratio={(np.isnan(ydelay_train).mean() if ydelay_train.size>0 else 1.0):.3f}")

                    # Remove NaN-label rows (always do)
                    mask_pay = ~np.isnan(ypay_train)
                    mask_test_pay = ~np.isnan(ypay_test)
                    Xp_train_pay = Xp_train[mask_pay]
                    ypay_train_valid = ypay_train[mask_pay]
                    Xp_test_pay = Xp_test[mask_test_pay]
                    ypay_test_valid = ypay_test[mask_test_pay]

                    mask_delay = ~np.isnan(ydelay_train)
                    mask_test_delay = ~np.isnan(ydelay_test)
                    Xp_train_delay = Xp_train[mask_delay]
                    ydelay_train_valid = ydelay_train[mask_delay]
                    Xp_test_delay = Xp_test[mask_test_delay]
                    ydelay_test_valid = ydelay_test[mask_test_delay]

                    # Payment
                    if RUN_PARTS.get("payment", True) and Xp_train_pay.shape[0] > 0:
                        scaler_pay = StandardScaler().fit(Xp_train_pay)
                        Xp_train_pay_s = scaler_pay.transform(Xp_train_pay)
                        Xp_test_pay_s  = scaler_pay.transform(Xp_test_pay) if Xp_test_pay.shape[0]>0 else np.zeros((0, Xp_train_pay_s.shape[1]))

                        split = max(int(len(Xp_train_pay_s)*0.9), 1)
                        X_tr, X_val = Xp_train_pay_s[:split], Xp_train_pay_s[split:]
                        y_tr, y_val = np.log1p(ypay_train_valid[:split]), np.log1p(ypay_train_valid[split:])

                        D = X_tr.shape[1]
                        X_tr_seq = X_tr.reshape(-1, 1, D)
                        X_val_seq = X_val.reshape(-1, 1, D)
                        X_test_seq = Xp_test_pay_s.reshape(-1, 1, D) if Xp_test_pay_s.shape[0]>0 else np.zeros((0,1,D))

                        tr_loader = build_dataloader_from_arrays(X_tr_seq, np.ones(X_tr_seq.shape[0], dtype=int), y_tr,
                                                                batch_size=PAYMENT_MLP_PARAMS["batch_size"])
                        val_loader = build_dataloader_from_arrays(X_val_seq, np.ones(X_val_seq.shape[0], dtype=int), y_val,
                                                                 batch_size=max(8, PAYMENT_MLP_PARAMS["batch_size"]//4), shuffle=False)

                        model_pay = MLPRegressor(input_dim=D, hidden_layers=PAYMENT_MLP_PARAMS["hidden_layers"],
                                                 dropout=PAYMENT_MLP_PARAMS["dropout"]).to(DEVICE)
                        opt_pay = torch.optim.Adam(model_pay.parameters(), lr=PAYMENT_MLP_PARAMS["lr"])
                        model_pay = train_seq_model(model_pay, opt_pay, nn.MSELoss(), tr_loader, val_loader,
                                                    epochs=PAYMENT_MLP_PARAMS["epochs"], patience=PAYMENT_MLP_PARAMS["patience"])

                        # predict on test
                        model_pay.eval()
                        if X_test_seq.shape[0] > 0:
                            preds_pay_log = np.concatenate([model_pay(Xb.to(DEVICE), lb.to(DEVICE)).detach().cpu().numpy()
                                                            for Xb, lb, _ in build_dataloader_from_arrays(X_test_seq, np.ones(X_test_seq.shape[0], dtype=int),
                                                                                                         np.zeros(X_test_seq.shape[0]), batch_size=PAYMENT_MLP_PARAMS["batch_size"], shuffle=False)]) \
                                            if X_test_seq.shape[0]>0 else np.array([])
                            preds_pay = safe_expm1_clip(preds_pay_log)
                            trues_pay = ypay_test_valid
                            result_payment_df = pd.DataFrame({
                                "claim_no": test_tx.loc[mask_test_pay, "claim_no"].values,
                                "pmt_no": test_tx.loc[mask_test_pay, "pmt_no"].values,
                                "payment_period": test_tx.loc[mask_test_pay, "payment_period"].values,
                                "y_true": ypay_test,
                                "y_pred": preds_pay
                            })
                            result_payment_df.to_csv(
                            os.path.join(OUTPUT_DIR, "payment_mlp_predictions.csv"),
                                index=False
                            )
                            mse_pay = mean_squared_error(trues_pay, preds_pay) if preds_pay.size>0 else np.nan
                            mae_pay = mean_absolute_error(trues_pay, preds_pay) if preds_pay.size>0 else np.nan
                            rr_pay  = preds_pay.sum()/(trues_pay.sum()+1e-8) if preds_pay.size>0 else np.nan
                        else:
                            mse_pay = mae_pay = rr_pay = np.nan
                    else:
                        mse_pay = mae_pay = rr_pay = np.nan

                    # Delay
                    if RUN_PARTS.get("delay", True) and Xp_train_delay.shape[0] > 0:
                        scaler_delay = StandardScaler().fit(Xp_train_delay)
                        Xp_train_delay_s = scaler_delay.transform(Xp_train_delay)
                        Xp_test_delay_s  = scaler_delay.transform(Xp_test_delay) if Xp_test_delay.shape[0]>0 else np.zeros((0, Xp_train_delay_s.shape[1]))

                        split = max(int(len(Xp_train_delay_s)*0.9), 1)
                        X_tr, X_val = Xp_train_delay_s[:split], Xp_train_delay_s[split:]
                        y_tr, y_val = np.log1p(ydelay_train_valid[:split]), np.log1p(ydelay_train_valid[split:])

                        D = X_tr.shape[1]
                        X_tr_seq = X_tr.reshape(-1, 1, D)
                        X_val_seq = X_val.reshape(-1, 1, D)
                        X_test_seq = Xp_test_delay_s.reshape(-1, 1, D) if Xp_test_delay_s.shape[0]>0 else np.zeros((0,1,D))

                        tr_loader = build_dataloader_from_arrays(X_tr_seq, np.ones(X_tr_seq.shape[0], dtype=int), y_tr,
                                                                batch_size=DELAY_MLP_PARAMS["batch_size"])
                        val_loader = build_dataloader_from_arrays(X_val_seq, np.ones(X_val_seq.shape[0], dtype=int), y_val,
                                                                 batch_size=max(8, DELAY_MLP_PARAMS["batch_size"]//4), shuffle=False)

                        model_delay = MLPRegressor(input_dim=D, hidden_layers=DELAY_MLP_PARAMS["hidden_layers"],
                                                   dropout=DELAY_MLP_PARAMS["dropout"]).to(DEVICE)
                        opt_delay = torch.optim.Adam(model_delay.parameters(), lr=DELAY_MLP_PARAMS["lr"])
                        model_delay = train_seq_model(model_delay, opt_delay, nn.MSELoss(), tr_loader, val_loader,
                                                      epochs=DELAY_MLP_PARAMS["epochs"], patience=DELAY_MLP_PARAMS["patience"])

                        model_delay.eval()
                        if X_test_seq.shape[0] > 0:
                            preds_delay_log = np.concatenate([model_delay(Xb.to(DEVICE), lb.to(DEVICE)).detach().cpu().numpy()
                                                              for Xb, lb, _ in build_dataloader_from_arrays(X_test_seq, np.ones(X_test_seq.shape[0], dtype=int),
                                                                                                           np.zeros(X_test_seq.shape[0]), batch_size=DELAY_MLP_PARAMS["batch_size"], shuffle=False)]) \
                                              if X_test_seq.shape[0]>0 else np.array([])
                            preds_delay = safe_expm1_clip(preds_delay_log)
                            trues_delay = ydelay_test_valid
                            result_delay_df = pd.DataFrame({
                                "claim_no": test_tx.loc[mask_test_delay, "claim_no"].values,
                                "pmt_no": test_tx.loc[mask_test_delay, "pmt_no"].values,
                                "payment_period": test_tx.loc[mask_test_delay, "payment_period"].values,
                                "y_true": ydelay_test,
                                "y_pred": preds_delay
                            })
                            result_delay_df.to_csv(
                                os.path.join(OUTPUT_DIR, "delay_mlp_predictions.csv"),
                                index=False
                            )


                            mse_delay = mean_squared_error(trues_delay, preds_delay) if preds_delay.size>0 else np.nan
                            mae_delay = mean_absolute_error(trues_delay, preds_delay) if preds_delay.size>0 else np.nan
                            rr_delay  = preds_delay.sum()/(trues_delay.sum()+1e-8) if preds_delay.size>0 else np.nan
                        else:
                            mse_delay = mae_delay = rr_delay = np.nan
                    else:
                        mse_delay = mae_delay = rr_delay = np.nan
            else:
                mse_pay = mae_pay = rr_pay = np.nan
                mse_delay = mae_delay = rr_delay = np.nan


            # --- 保存结果 ---
            results.append({
                "Dataset": scale,
                "Claim_MSE": mse_claim, "Claim_MAE": mae_claim, "Claim_RR": rr_claim,
                "Claim_hidden": CLAIM_MLP_PARAMS["hidden_layers"],
                "Pay_MSE": mse_pay, "Pay_MAE": mae_pay, "Pay_RR": rr_pay,
                "Pay_hidden": PAYMENT_MLP_PARAMS["hidden_layers"],
                "Delay_MSE": mse_delay, "Delay_MAE": mae_delay, "Delay_RR": rr_delay,
                "Delay_hidden": DELAY_MLP_PARAMS["hidden_layers"],
                "CLAIM_WINDOW": CLAIM_WINDOW, "K_CLAIMS_PAYMENT": K_CLAIMS_PAYMENT, "MAX_STEPS": MAX_STEPS
            })

        # ---------------------- 写入 CSV 并打印当前实验 ----------------------
        df_res = pd.DataFrame(results)
        if os.path.exists(OUTPUT_CSV):
            df_existing = pd.read_csv(OUTPUT_CSV)
            df_final = pd.concat([df_existing, df_res], ignore_index=True)
        else:
            df_final = df_res
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"Results saved to {OUTPUT_CSV}")
        print_results_compact(results)



if __name__=="__main__":
    run_pipeline()