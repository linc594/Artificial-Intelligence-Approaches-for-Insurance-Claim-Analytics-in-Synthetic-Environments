"""
gan_payment_seq_full.py
-----------------------
上游：序列 GAN 生成 claim 的完整 transaction 序列
下游：MLP 预测最后一笔 payment_size
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ks_2samp
# ---------------- Config ----------------
BASE_TRAIN = r"C:\Users\24495\Desktop\experiment\data\train data"
BASE_TEST  = r"C:\Users\24495\Desktop\experiment\data\test data"
OUTPUT_DIR = r"C:\Users\24495\Desktop\experiment\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_TX = os.path.join(BASE_TRAIN, "train_transactions.csv")
TEST_TX  = os.path.join(BASE_TEST,  "test_transactions.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---------------- 上游 GAN 超参 ----------------
MAX_N = 30
NOISE_DIM = 64
GAN_EPOCHS = 20 #200
GAN_BATCH = 64
GAN_LR = 2e-4

SYNTH_RATIO = 1.0

# ---------------- 下游 MLP 超参 ----------------
MLP_HIDDEN = [64]
MLP_LR = 1e-3
MLP_BATCH = 256
MLP_EPOCHS = 200
MLP_PATIENCE = 5

LOG_CLIP_MIN, LOG_CLIP_MAX = -8, 12
SEED = 46

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------- 工具 ----------------
def safe_expm1_clip(x):
    x = np.clip(x, LOG_CLIP_MIN, LOG_CLIP_MAX)
    y = np.expm1(x)
    med = np.nanmedian(y)
    y = np.where(np.isfinite(y), y, med)
    return np.maximum(y, 0.0)


# ============================================================
# ================== 上游：序列 GAN ==========================
# ============================================================

def build_sequences(tx_df, max_n=MAX_N):
    tx = tx_df.copy()
    tx["claim_no"] = tx["claim_no"].astype(str)
    tx = tx.sort_values(["claim_no", "pmt_no"])

    seqs, lengths, metas = [], [], []
    for claim_no, g in tx.groupby("claim_no", sort=False):
        claim_size = float(g["claim_size"].iloc[0])
        n = len(g)
        ratio = (g["payment_size"].astype(float) / (claim_size + 1e-8)).to_numpy()
        delay = g["payment_delay"].astype(float).to_numpy()
        time_ = g["payment_time"].astype(float).to_numpy()
        period = g["payment_period"].astype(float).to_numpy()

        seq = np.stack([ratio, delay, time_, period], axis=1)
        if n < max_n:
            seq = np.vstack([seq, np.zeros((max_n - n, 4))])
        else:
            seq = seq[:max_n]
            n = max_n

        seqs.append(seq)
        lengths.append(n)
        metas.append({
            "claim_size": claim_size,
            "notidel": float(g["notidel"].iloc[0]),
            "setldel": float(g["setldel"].iloc[0]),
            "occurrence_period": float(g["occurrence_period"].iloc[0]),
            "occurrence_time": float(g["occurrence_time"].iloc[0]), 
        })

    return np.array(seqs, dtype=np.float32), np.array(lengths), metas


class SeqGenerator(nn.Module):
    def __init__(self, noise_dim, out_dim, max_n, hidden=128):
        super().__init__()
        self.max_n = max_n
        self.fc = nn.Linear(noise_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, z):
        h = torch.relu(self.fc(z)).unsqueeze(1).repeat(1, self.max_n, 1)
        out, _ = self.lstm(h)
        return self.out(out)


class SeqDiscriminator(nn.Module):

    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x, lengths):
        out, _ = self.lstm(x)

        # lengths = 真实 sequence 长度
        # 取每条 sequence 最后一个有效 timestep
        idx = (lengths - 1).clamp(min=0)

        last_out = out[
            torch.arange(x.size(0), device=x.device),
            idx
        ]

        return self.fc(last_out).squeeze(-1)


def train_gan(seqs, lengths):
    X = torch.tensor(seqs, dtype=torch.float32).to(DEVICE)
    L = torch.tensor(lengths, dtype=torch.long).to(DEVICE)
    N, T, D = X.shape
    G = SeqGenerator(NOISE_DIM, D, T).to(DEVICE)
    Dnet = SeqDiscriminator(D).to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=GAN_LR, betas=(0.5, 0.9))
    optD = torch.optim.Adam(Dnet.parameters(), lr=GAN_LR, betas=(0.5, 0.9))
    bce = nn.BCELoss()
    loader = DataLoader(TensorDataset(X, L), batch_size=GAN_BATCH, shuffle=True, drop_last=True)



    for ep in range(GAN_EPOCHS):
        for real, lengths_batch in loader:
            bs = real.size(0)
            real_lab = torch.ones(bs, device=DEVICE)
            fake_lab = torch.zeros(bs, device=DEVICE)

            lengths_batch = lengths_batch.to(DEVICE)

            z = torch.randn(bs, NOISE_DIM, device=DEVICE)
            fake = G(z).detach()
            lossD = bce(Dnet(real, lengths_batch), real_lab) + bce(Dnet(fake, lengths_batch), fake_lab)
            optD.zero_grad(); lossD.backward(); optD.step()

            z2 = torch.randn(bs, NOISE_DIM, device=DEVICE)
            fake2 = G(z2)
            lossG = bce(Dnet(fake2, lengths_batch), real_lab)
            optG.zero_grad(); lossG.backward(); optG.step()

        if (ep + 1) % 50 == 0:
            print(f"[SeqGAN payment] ep {ep+1}/{GAN_EPOCHS}")
    return G


def generate_synthetic(G, lengths, metas, n_synth):
    G.eval()
    N_real = len(lengths)
    idx = np.random.randint(0, N_real, size=n_synth)
    rows = []
    with torch.no_grad():
        for i, j in enumerate(idx):
            n = lengths[j]
            meta = metas[j]
            z = torch.randn(1, NOISE_DIM, device=DEVICE)
            seq = G(z).cpu().numpy()[0][:n]

            ratio = np.abs(seq[:, 0])
            ratio = ratio / (ratio.sum() + 1e-8)
            delay = np.abs(seq[:, 1])
            time_ = np.abs(seq[:, 2])
            period = np.abs(seq[:, 3])
            time_ = np.minimum(time_, period - 1e-3)
            payment_size = ratio * meta["claim_size"]

            for k in range(n):
                rows.append({
                    "claim_no": f"syn_{i}",
                    "pmt_no": k + 1,
                    "occurrence_period": meta["occurrence_period"],
                    "claim_size": meta["claim_size"],
                    "notidel": meta["notidel"],
                    "setldel": meta["setldel"],
                    "occurrence_time": meta["occurrence_time"],
                    "payment_time": time_[k],
                    "payment_period": period[k],
                    "payment_size": payment_size[k],
                    "payment_delay": delay[k],
                    "payment_inflated": payment_size[k],  
                })
    return pd.DataFrame(rows)

# ============================================================
# ================== Fidelity ================================
# ============================================================

def summarize_fidelity(real_df, synth_df):
    """
    Compare real and synthetic payment transaction distributions.

    Note:
    - sequence length is NOT evaluated here because it is inherited
      from real claims during synthetic generation.
    - static claim-level variables are NOT evaluated as GAN fidelity
      because they are copied from real claims.
    """

    features = [
        "payment_size",
        "payment_delay",
        "payment_time",
        "payment_period",
    ]

    # Payment ratio
    real_ratio = (
        real_df["payment_size"].astype(float)
        / (real_df["claim_size"].astype(float) + 1e-8)
    )

    synth_ratio = (
        synth_df["payment_size"].astype(float)
        / (synth_df["claim_size"].astype(float) + 1e-8)
    )

    real_data = {
        "payment_ratio": real_ratio,
        "payment_size": real_df["payment_size"].astype(float),
        "payment_delay": real_df["payment_delay"].astype(float),
        "payment_time": real_df["payment_time"].astype(float),
        "payment_period": real_df["payment_period"].astype(float),
    }

    synth_data = {
        "payment_ratio": synth_ratio,
        "payment_size": synth_df["payment_size"].astype(float),
        "payment_delay": synth_df["payment_delay"].astype(float),
        "payment_time": synth_df["payment_time"].astype(float),
        "payment_period": synth_df["payment_period"].astype(float),
    }

    rows = []

    for name in real_data:

        r = real_data[name].replace([np.inf, -np.inf], np.nan).dropna()
        s = synth_data[name].replace([np.inf, -np.inf], np.nan).dropna()

        rq = np.percentile(r, [5, 25, 50, 75, 95, 99])
        sq = np.percentile(s, [5, 25, 50, 75, 95, 99])

        ks_stat, ks_p = ks_2samp(r, s)

        rows.append({
            "feature": name,

            "real_mean": np.mean(r),
            "synthetic_mean": np.mean(s),

            "real_std": np.std(r),
            "synthetic_std": np.std(s),

            "real_median": np.median(r),
            "synthetic_median": np.median(s),

            "real_p05": rq[0],
            "synthetic_p05": sq[0],

            "real_p25": rq[1],
            "synthetic_p25": sq[1],

            "real_p50": rq[2],
            "synthetic_p50": sq[2],

            "real_p75": rq[3],
            "synthetic_p75": sq[3],

            "real_p95": rq[4],
            "synthetic_p95": sq[4],

            "real_p99": rq[5],
            "synthetic_p99": sq[5],

            "KS": ks_stat,
            "KS_pvalue": ks_p,
        })

    result = pd.DataFrame(rows)

    print("\n========== PAYMENT GAN FIDELITY ==========")
    print(result.to_string(index=False))

    return result

def summarize_sequence_fidelity(real_df, synth_df):
    """
    Compare within-claim sequential structure.

    The comparison is performed after grouping transactions by claim.
    Sequence length itself is excluded from the GAN fidelity assessment
    because synthetic sequence lengths are inherited from real claims.
    """

    def build_sequence_features(df):

        seq_stats = []

        for claim_no, g in df.groupby("claim_no"):

            g = g.sort_values("pmt_no").copy()

            claim_size = float(g["claim_size"].iloc[0])

            ratio = (
                g["payment_size"].astype(float)
                / (claim_size + 1e-8)
            ).to_numpy()

            delay = g["payment_delay"].astype(float).to_numpy()
            time_ = g["payment_time"].astype(float).to_numpy()
            period = g["payment_period"].astype(float).to_numpy()

            # First-to-last change
            if len(g) >= 2:

                seq_stats.append({
                    "ratio_first": ratio[0],
                    "ratio_last": ratio[-1],

                    "delay_first": delay[0],
                    "delay_last": delay[-1],

                    "time_first": time_[0],
                    "time_last": time_[-1],

                    "period_first": period[0],
                    "period_last": period[-1],

                    "ratio_diff": ratio[-1] - ratio[0],
                    "delay_diff": delay[-1] - delay[0],
                    "time_diff": time_[-1] - time_[0],
                    "period_diff": period[-1] - period[0],
                })

        return pd.DataFrame(seq_stats)

    real_seq = build_sequence_features(real_df)
    synth_seq = build_sequence_features(synth_df)

    features = [
        "ratio_first",
        "ratio_last",
        "delay_first",
        "delay_last",
        "time_first",
        "time_last",
        "period_first",
        "period_last",
        "ratio_diff",
        "delay_diff",
        "time_diff",
        "period_diff",
    ]

    rows = []

    for feature in features:

        r = real_seq[feature].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        s = synth_seq[feature].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        rows.append({
            "feature": feature,
            "real_mean": r.mean(),
            "synthetic_mean": s.mean(),
            "real_std": r.std(),
            "synthetic_std": s.std(),
            "real_median": r.median(),
            "synthetic_median": s.median(),
        })

    result = pd.DataFrame(rows)

    print("\n========== SEQUENCE STRUCTURE FIDELITY ==========")
    print(result.to_string(index=False))

    return result
# ============================================================
# ================== 下游：MLP ================================
# ============================================================

class PaymentMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden_layers:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_payment_dataset(df, claim_nos=None):
    df = df.copy()
    df["claim_no"] = df["claim_no"].astype(str)
    feat_cols = [
        c for c in df.columns
        if c not in ["claim_no", "payment_size", "payment_delay"]
    ]

    X, y, groups = [], [], []

    grouped = df.groupby("claim_no")

    for claim_no, g in grouped:

        if claim_nos is not None and claim_no not in claim_nos:
            continue

        g = g.sort_values("pmt_no")

        final_pay = g.iloc[-1]["payment_size"]

        for _, r in g.iterrows():

            X.append(r[feat_cols].values)
            y.append(final_pay)
            groups.append(claim_no)

    return (
        np.asarray(X, float),
        np.asarray(y, float),
        feat_cols,
        np.asarray(groups)
    )


def train_mlp(model, tr_loader, va_loader):
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = nn.MSELoss()
    best, bad, best_state = float("inf"), 0, None

    for ep in range(MLP_EPOCHS):
        model.train()
        tr = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            tr.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl.append(loss_fn(model(xb), yb).item())

        v = np.mean(vl)
        print(f"Epoch {ep+1:03d} | train={np.mean(tr):.6f} | val={v:.6f}")

        if v < best:
            best, bad = v, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return model


# ============================================================
# ================== 主流程 ==================================
# ============================================================

def run():
    # 1) 加载真实 train
    train_tx = pd.read_csv(TRAIN_TX)
    test_tx  = pd.read_csv(TEST_TX)


    real_claims = train_tx["claim_no"].astype(str).unique()

    rng = np.random.default_rng(SEED)
    rng.shuffle(real_claims)

    split_claim = int(len(real_claims) * 0.9)

    train_claims = set(real_claims[:split_claim])
    val_claims = set(real_claims[split_claim:])

    # 2) 训练上游 GAN
    gan_train_tx = train_tx[
        train_tx["claim_no"].astype(str).isin(train_claims)
    ].copy()
    seqs, lengths, metas = build_sequences(gan_train_tx)
    print(f"Sequences: {seqs.shape}")
    G = train_gan(seqs, lengths)






    # 3) 生成合成样本
    n_synth = int(len(lengths) * SYNTH_RATIO)
    synth_df = generate_synthetic(G, lengths, metas, n_synth)
    print(f"Synthetic rows: {len(synth_df)}")

    # 3.1) Fidelity analysis
    fidelity_result = summarize_fidelity(
        gan_train_tx,
        synth_df
    )

    sequence_fidelity_result = summarize_sequence_fidelity(
        gan_train_tx,
        synth_df
    )

    fidelity_result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "payment_gan_fidelity.csv"
        ),
        index=False
    )

    sequence_fidelity_result.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "payment_gan_sequence_fidelity.csv"
        ),
        index=False
    )

    synth_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "synthetic_payment_seq.csv"
        ),
        index=False
    )
    synth_df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_payment_seq.csv"), index=False)

    # 4) 合并真实 + 合成



    # 5) Build datasets
    # Real training claims
    X_real_tr, y_real_tr, feat_cols, _ = build_payment_dataset(
        train_tx,
        train_claims
    )

    # Real validation claims
    X_real_va, y_real_va, _, _ = build_payment_dataset(
        train_tx,
        val_claims
    )

    # All synthetic claims are used only for training
    X_syn, y_syn, _, _ = build_payment_dataset(
        synth_df
    )

# Test set
    Xte, yte, _, _ = build_payment_dataset(
        test_tx
    )

    # Combine real + synthetic training data
    Xtr = np.concatenate([X_real_tr, X_syn], axis=0)
    ytr = np.concatenate([y_real_tr, y_syn], axis=0)

    Xva = X_real_va
    yva = y_real_va

    # 6) Standardization
    scaler = StandardScaler().fit(Xtr)

    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    # Log-transform target
    ytr_log = np.log1p(ytr)
    yva_log = np.log1p(yva)

    # 7) DataLoaders
    tr_loader = DataLoader(
        TensorDataset(
            torch.tensor(Xtr_s, dtype=torch.float32),
            torch.tensor(ytr_log, dtype=torch.float32)
        ),
        batch_size=MLP_BATCH,
        shuffle=True
    )

    va_loader = DataLoader(
        TensorDataset(
            torch.tensor(Xva_s, dtype=torch.float32),
            torch.tensor(yva_log, dtype=torch.float32)
        ),
        batch_size=MLP_BATCH // 4,
        shuffle=False
    )

    model = PaymentMLP(
        Xtr_s.shape[1],
        MLP_HIDDEN
    ).to(DEVICE)

    model = train_mlp(model, tr_loader, va_loader)

    # 评估
    model.eval()
    with torch.no_grad():
        preds_log = model(torch.tensor(Xte_s, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    preds = safe_expm1_clip(preds_log)

    mse = mean_squared_error(yte, preds)
    mae = mean_absolute_error(yte, preds)
    rr = preds.sum() / (yte.sum() + 1e-8)

    print("\n[FINAL]")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RR : {rr:.6f}")

    out_df = test_tx[["claim_no", "pmt_no", "payment_period"]].copy()
    out_df["y_true"] = yte
    out_df["y_pred"] = preds
    out_df.to_csv(os.path.join(OUTPUT_DIR, "payment_gan_seq_predictions.csv"), index=False)
    print("Saved predictions.")


if __name__ == "__main__":
    run()