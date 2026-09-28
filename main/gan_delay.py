# gan_delay.py
# Conditional lightweight WGAN-GP that generates single transaction rows
# then trains Delay MLP using real + synthetic rows.
# Designed to replace the GAN parts in your pipeline and be runnable as a standalone script.

import os
from copy import deepcopy
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance, ks_2samp

# ----------------- CONFIG (保持和原始脚本一致的路径与文件名) -----------------
BASE_TRAIN = r"C:\Users\24495\Desktop\experiment\data\train data"
BASE_TEST  = r"C:\Users\24495\Desktop\experiment\data\test data"
OUTPUT_DIR = r"C:\Users\24495\Desktop\experiment\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "train_claims": "train_claims.csv",
    "train_tx": "train_transactions.csv",
    "test_claims": "test_claims.csv",
    "test_tx": "test_transactions.csv"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# hyperparams
SEQ_GAN_EPOCHS = 20
SEQ_BATCH = 64
SEQ_LR = 2e-4
SEQ_NOISE_DIM = 64
BATCH_SIZE = 256
SEEDS = [42, 43, 44, 45, 46]
FIDELITY_SEED = 42

# ----------------- Reproducibility -----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- Fidelity evaluation -----------------
def evaluate_delay_gan_fidelity(X_real, X_synth, feat_cols, seed, save_dir):
    """
    Compare real training rows with GAN-generated rows.
    The delay target is excluded because it is not directly generated as a feature.
    """
    rows = []

    for j, col in enumerate(feat_cols):
        real = np.asarray(X_real[:, j], dtype=float)
        synth = np.asarray(X_synth[:, j], dtype=float)

        real = real[np.isfinite(real)]
        synth = synth[np.isfinite(synth)]

        if len(real) == 0 or len(synth) == 0:
            continue

        real_mean = float(np.mean(real))
        synth_mean = float(np.mean(synth))
        real_std = float(np.std(real))
        synth_std = float(np.std(synth))

        mean_rel_error = abs(synth_mean - real_mean) / (abs(real_mean) + 1e-8)
        std_rel_error = abs(synth_std - real_std) / (abs(real_std) + 1e-8)

        wdist = float(wasserstein_distance(real, synth))
        ks_stat, ks_p = ks_2samp(real, synth)

        quantiles = {}
        for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
            quantiles[f"real_q{int(q*100):02d}"] = float(np.quantile(real, q))
            quantiles[f"synth_q{int(q*100):02d}"] = float(np.quantile(synth, q))

        rows.append({
            "feature": col,
            "real_mean": real_mean,
            "synthetic_mean": synth_mean,
            "mean_relative_error": mean_rel_error,
            "real_std": real_std,
            "synthetic_std": synth_std,
            "std_relative_error": std_rel_error,
            "wasserstein_distance": wdist,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            **quantiles
        })

    fidelity_df = pd.DataFrame(rows)
    out_path = os.path.join(save_dir, f"delay_gan_fidelity_seed_{seed}.csv")
    fidelity_df.to_csv(out_path, index=False)

    if len(fidelity_df) > 0:
        print("\n=== Delay GAN Fidelity ===")
        print(f"Seed: {seed}")
        print(f"Mean relative mean error: {fidelity_df['mean_relative_error'].mean():.6f}")
        print(f"Mean relative std error:  {fidelity_df['std_relative_error'].mean():.6f}")
        print(f"Mean Wasserstein distance: {fidelity_df['wasserstein_distance'].mean():.6f}")
        print(f"Mean KS statistic:         {fidelity_df['ks_statistic'].mean():.6f}")
        print(f"Saved to: {out_path}")

    return fidelity_df


# ----------------- Utilities -----------------
def safe_expm1_clip(x, min_log=-5.0, max_log=15.0):
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, min_log, max_log)
    out = np.expm1(arr)
    if np.isfinite(out).any():
        med = float(np.nanmedian(out))
    else:
        med = 0.0
    mask = np.isfinite(out)
    out = np.where(mask, out, med)
    out = np.maximum(out, 0.0)
    return out

# ----------------- dataset helper (row-level) -----------------
def build_payment_delay_dataset_all_rows(df):
    """
    Build dataset where each transaction-row is an input sample.
    The label for each row is the final payment_size/payment_delay of its claim (last row within claim).
    Returns:
      X (N_rows, D_features), y_pay (N_rows,), y_delay (N_rows,), feat_cols (list)
    """
    df = df.copy()
    # ensure deterministic order within claim
    if "pmt_no" in df.columns:
        df = df.sort_values(["claim_no", "pmt_no"])
    elif "payment_time" in df.columns:
        df = df.sort_values(["claim_no", "payment_time"])
    else:
        # fallback to occurrence/payment period if available
        sort_cols = ["claim_no"]
        if "occurrence_period" in df.columns:
            sort_cols.append("occurrence_period")
        if "payment_period" in df.columns:
            sort_cols.append("payment_period")
        df = df.sort_values(sort_cols)

    feat_cols = [c for c in df.columns if c not in ["payment_size", "payment_delay", "claim_no"]]

    X_list = []
    y_pay_list = []
    y_delay_list = []

    for claim_no, g in df.groupby("claim_no", sort=False):
        grp_sorted = g.reset_index(drop=True)
        last_row = grp_sorted.iloc[-1]
        last_pay = last_row.get("payment_size", np.nan)
        last_delay = last_row.get("payment_delay", np.nan)
        for _, row in grp_sorted.iterrows():
            X_list.append(row[feat_cols].astype(float).values)
            y_pay_list.append(last_pay)
            y_delay_list.append(last_delay)

    if len(X_list) == 0:
        return np.zeros((0, len(feat_cols))), np.array([]), np.array([]), feat_cols

    X = np.vstack(X_list)
    y_pay = np.array(y_pay_list, dtype=float)
    y_delay = np.array(y_delay_list, dtype=float)
    return X, y_pay, y_delay, feat_cols

# ----------------- Models: conditional MLP generator/discriminator -----------------
class CondMLPGenerator(nn.Module):
    def __init__(self, noise_dim, cond_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, z, cond):
        # z: (B, noise_dim), cond: (B, cond_dim)
        inp = torch.cat([z, cond], dim=1)
        return self.net(inp)

class CondMLPDiscriminator(nn.Module):
    def __init__(self, in_dim, cond_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x, cond):
        inp = torch.cat([x, cond], dim=1)
        return self.net(inp).view(-1)

# ----------------- gradient penalty -----------------
def gradient_penalty(Dmodel, real, fake, cond):
    device = real.device
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = Dmodel(interpolates, cond)
    grads = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.reshape(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# ----------------- Train conditional WGAN-GP on rows -----------------
def train_cond_wgan_gp_rows(X_rows, y_delay_log, epochs=SEQ_GAN_EPOCHS, bs=SEQ_BATCH, lr=SEQ_LR,
                            noise_dim=SEQ_NOISE_DIM, cond_qs=(0.7, 0.95), gp_lambda=10.0, D_steps=5):
    """
    X_rows: numpy (N, D) or torch tensor - scaled rows (preferably scaled)
    y_delay_log: numpy (N,) log1p delays used for conditioning & tail weighting
    Returns G (generator), and also returns arrays needed for sampling cond->delay mapping.
    """
    # move to tensor on DEVICE
    if isinstance(X_rows, np.ndarray):
        X = torch.tensor(X_rows, dtype=torch.float32, device=DEVICE)
    else:
        X = X_rows.to(DEVICE).float()
    if isinstance(y_delay_log, np.ndarray):
        y = torch.tensor(y_delay_log, dtype=torch.float32, device=DEVICE)
    else:
        y = y_delay_log.to(DEVICE).float()

    N, D = X.shape

    # compute conditional buckets by quantiles
    q_low = float(torch.quantile(y, torch.tensor(cond_qs[0], device=DEVICE)).item())
    q_high = float(torch.quantile(y, torch.tensor(cond_qs[1], device=DEVICE)).item())
    # create cond (0/1/2)
    cond_arr = torch.where(y >= q_high, 2, torch.where(y >= q_low, 1, 0)).unsqueeze(1).to(DEVICE)  # shape (N,1)
    # one-hot encoding (3 classes)
    cond_onehot = torch.zeros(N, 3, device=DEVICE)
    cond_onehot.scatter_(1, cond_arr.long(), 1.0)

    # prepare generator/discriminator
    G = CondMLPGenerator(noise_dim=noise_dim, cond_dim=3, out_dim=D, hidden=256).to(DEVICE)
    D = CondMLPDiscriminator(in_dim=D, cond_dim=3, hidden=256).to(DEVICE)

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    dataset = TensorDataset(X, cond_onehot, y)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)

    print("Training conditional WGAN-GP on rows ...")
    for ep in range(1, epochs + 1):
        g_acc = 0.0
        d_acc = 0.0
        for Xb, condb, yb in loader:
            bs2 = Xb.size(0)
            # tail weight on this batch (emphasize heavy-tailed)
            q90 = float(torch.quantile(y, torch.tensor(0.9, device=DEVICE)).item())
            w = (1.0 + 3.0 * ((yb > q90).float())).to(DEVICE)

            # train discriminator D_steps times
            for _ in range(D_steps):
                z = torch.randn(bs2, noise_dim, device=DEVICE)
                fake = G(z, condb)
                D_real = D(Xb, condb)
                D_fake = D(fake.detach(), condb)
                gp = gradient_penalty(D, Xb, fake.detach(), condb)

                lossD = - ( (D_real * w).mean() - (D_fake * w).mean() ) + gp_lambda * gp

                optD.zero_grad()
                lossD.backward()
                optD.step()

            # train generator
            z2 = torch.randn(bs2, noise_dim, device=DEVICE)
            fake2 = G(z2, condb)
            D_fake2 = D(fake2, condb)
            lossG = - (D_fake2 * w).mean()

            optG.zero_grad()
            lossG.backward()
            optG.step()

            g_acc += lossG.item()
            d_acc += lossD.item()

        if ep % 10 == 0 or ep == 1:
            print(f"[cWGAN-GP rows] ep {ep}/{epochs}  D_loss={d_acc/len(loader):.4f}  G_loss={g_acc/len(loader):.4f}")

    # return generator + required metadata for sampling mapping
    return {
        "G": G,
        "q_low": q_low,
        "q_high": q_high,
        "cond_onehot_all": cond_onehot.cpu().numpy(),
        "y_all": y.cpu().numpy()
    }

# ----------------- synthesizer for rows -----------------
def generate_synthetic_rows(trained_dict, scaler, n_samples, noise_dim=SEQ_NOISE_DIM):
    """
    trained_dict: output of train_cond_wgan_gp_rows
    scaler: fitted StandardScaler on original row-features (shape D)
    Returns: synthetic_rows_unscaled (n_samples, D), synthetic_assigned_log_delays (n_samples,)
    """
    G = trained_dict["G"]
    q_low = trained_dict["q_low"]
    q_high = trained_dict["q_high"]
    cond_onehot_all = trained_dict["cond_onehot_all"]  # numpy (N,3)
    y_all = trained_dict["y_all"]  # numpy (N,)

    device = next(G.parameters()).device
    N_orig = cond_onehot_all.shape[0]

    # sample conds by sampling indices from original cond distribution (preserves distribution)
    idxs = np.random.choice(np.arange(N_orig), size=n_samples, replace=True)
    conds = torch.tensor(cond_onehot_all[idxs], dtype=torch.float32, device=device)  # (n_samples,3)

    # generate
    G.eval()
    out = []
    bs = 1024
    with torch.no_grad():
        for i in range(0, n_samples, bs):
            cur = min(bs, n_samples - i)
            z = torch.randn(cur, noise_dim, device=device)
            cond_b = conds[i:i+cur]
            s = G(z, cond_b)  # (cur, D) in scaled space (same scale as training X input)
            out.append(s.cpu().numpy())

    X_synth_scaled = np.vstack(out)  # (n_samples, D) - in scaled space
    # inverse transform to original feature space
    if scaler is None:
        X_synth_unscaled = X_synth_scaled
    else:
        X_synth_unscaled = scaler.inverse_transform(X_synth_scaled)

    # assign synthetic delay log by sampling from y_all among those indices with same cond bucket
    # using the same idxs preserves the cond distribution approx.
    y_synth_log = y_all[idxs].astype(float)

    return X_synth_unscaled, y_synth_log

# ----------------- MLP Delay regressor (same as your baseline MLPRegressor) -----------------
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
        x_flat = x.view(x.size(0), -1)
        return self.model(x_flat).squeeze(-1)

def train_mlp_regressor(model, train_loader, val_loader, epochs=40, lr=1e-3, device=DEVICE):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = float("inf")
    bad = 0
    best_state = None
    for ep in range(1, epochs+1):
        model.train()
        tr_losses=[]
        for Xb, lb, yb in train_loader:
            Xb, lb, yb = Xb.to(device), lb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(Xb, lb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        # val
        model.eval()
        vl=[]
        with torch.no_grad():
            for Xv, lv, yv in val_loader:
                Xv, lv, yv = Xv.to(device), lv.to(device), yv.to(device)
                vl.append(loss_fn(model(Xv, lv), yv).item())
        mean_val = np.mean(vl) if vl else float("inf")
        print(f"Train epoch {ep}/{epochs} train_loss={np.mean(tr_losses):.6f} val_loss={mean_val:.6f}")
        if mean_val < best:
            best = mean_val
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                print("Early stopping")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

# ----------------- Main pipeline -----------------
def run_single_seed(seed):
    set_seed(seed)
    print("\n" + "=" * 70)
    print(f"RUNNING SEED {seed}")
    print("=" * 70)

    # load inputs
    train_tx = pd.read_csv(os.path.join(BASE_TRAIN, FILES["train_tx"]))
    test_tx  = pd.read_csv(os.path.join(BASE_TEST, FILES["test_tx"]))

    for df in [train_tx, test_tx]:
        if "claim_no" in df.columns:
            df["claim_no"] = df["claim_no"].astype(str)

    # build row-level dataset
    X_train_rows, ypay_train, ydelay_train, feat_cols = build_payment_delay_dataset_all_rows(train_tx)
    X_test_rows, ypay_test, ydelay_test, _ = build_payment_delay_dataset_all_rows(test_tx)

    # drop rows with NaN delay labels
    mask_train = ~np.isnan(ydelay_train)
    X_train_rows = X_train_rows[mask_train]
    ydelay_train = ydelay_train[mask_train]
    ypay_train = ypay_train[mask_train]

    mask_test = ~np.isnan(ydelay_test)
    X_test_rows = X_test_rows[mask_test]
    ydelay_test = ydelay_test[mask_test]
    ypay_test = ypay_test[mask_test]

    if X_train_rows.shape[0] == 0:
        print("No training rows with valid delay label. Exiting seed.")
        return {
            "seed": seed,
            "Delay_MSE": np.nan,
            "Delay_MAE": np.nan,
            "Delay_RR": np.nan
        }

    # log-transform delays for labels and conditioning
    y_train_log = np.log1p(np.maximum(ydelay_train, 0.0))
    y_test_log = np.log1p(np.maximum(ydelay_test, 0.0))

    # fit scaler on row features
    scaler_row = StandardScaler().fit(X_train_rows)
    X_train_scaled = scaler_row.transform(X_train_rows)
    X_test_scaled = (
        scaler_row.transform(X_test_rows)
        if X_test_rows.shape[0] > 0
        else np.zeros((0, X_train_rows.shape[1]))
    )

    # ============== Train conditional WGAN-GP on rows ==============
    trained = train_cond_wgan_gp_rows(
        X_train_scaled,
        y_train_log,
        epochs=SEQ_GAN_EPOCHS,
        bs=SEQ_BATCH,
        lr=SEQ_LR,
        noise_dim=SEQ_NOISE_DIM
    )

    # ============== Generate synthetic rows ==============
    n_synth = X_train_scaled.shape[0]
    print(f"Generating {n_synth} synthetic rows ...")
    X_synth_unscaled, y_synth_log = generate_synthetic_rows(
        trained,
        scaler_row,
        n_synth,
        noise_dim=SEQ_NOISE_DIM
    )

    D = X_train_rows.shape[1]
    X_synth_unscaled = X_synth_unscaled.reshape(n_synth, D)

    # ============== Fidelity: representative seed 42 only ==============
    if seed == FIDELITY_SEED:
        evaluate_delay_gan_fidelity(
            X_real=X_train_rows,
            X_synth=X_synth_unscaled,
            feat_cols=feat_cols,
            seed=seed,
            save_dir=OUTPUT_DIR
        )

    # scale synthetic back for final MLP training
    X_synth_scaled = scaler_row.transform(X_synth_unscaled)

    # Combine real + synthetic
    X_real_scaled_seq = X_train_scaled.reshape(-1, 1, D)
    X_synth_scaled_seq = X_synth_scaled.reshape(-1, 1, D)
    X_aug = np.vstack([X_real_scaled_seq, X_synth_scaled_seq])
    y_aug = np.concatenate([y_train_log, y_synth_log])

    # train/val split — keep the original code's protocol
    N_aug = X_aug.shape[0]
    split = max(int(N_aug * 0.9), 1)
    Xtr = X_aug[:split]
    Xval = X_aug[split:]
    ytr = y_aug[:split]
    yval = y_aug[split:]

    tr_loader = DataLoader(
        TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.ones(Xtr.shape[0], dtype=torch.long),
            torch.tensor(ytr, dtype=torch.float32)
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(Xval, dtype=torch.float32),
            torch.ones(Xval.shape[0], dtype=torch.long),
            torch.tensor(yval, dtype=torch.float32)
        ),
        batch_size=max(8, BATCH_SIZE // 4),
        shuffle=False
    )

    # instantiate Delay MLP regressor
    input_dim = D
    model_delay = MLPRegressor(
        input_dim=input_dim,
        hidden_layers=[64],
        dropout=0.0
    ).to(DEVICE)

    # train
    print("Training Delay MLP on augmented (real + synthetic) rows ...")
    model_delay = train_mlp_regressor(
        model_delay,
        tr_loader,
        val_loader,
        epochs=40,
        lr=1e-3,
        device=DEVICE
    )

    # evaluate on test rows
    preds_log = []
    model_delay.eval()

    with torch.no_grad():
        if X_test_scaled.shape[0] > 0:
            X_test_seq = X_test_scaled.reshape(-1, 1, D)
            ds = DataLoader(
                TensorDataset(
                    torch.tensor(X_test_seq, dtype=torch.float32),
                    torch.ones(X_test_seq.shape[0], dtype=torch.long),
                    torch.zeros(X_test_seq.shape[0], dtype=torch.float32)
                ),
                batch_size=BATCH_SIZE,
                shuffle=False
            )

            for Xb, lb, _ in ds:
                Xb = Xb.to(DEVICE)
                out = model_delay(Xb).cpu().numpy()
                preds_log.append(out)

    if len(preds_log) == 0:
        preds = np.array([])
    else:
        preds = safe_expm1_clip(np.concatenate(preds_log).reshape(-1))

    # compute metrics on test rows
    if preds.size > 0 and ydelay_test.size > 0:
        mse = float(mean_squared_error(ydelay_test, preds))
        mae = float(mean_absolute_error(ydelay_test, preds))
        rr = float(preds.sum() / (ydelay_test.sum() + 1e-8))
    else:
        mse = mae = rr = np.nan

    # save predictions for this seed
    result_delay_df = pd.DataFrame({
        "claim_no": test_tx.loc[mask_test, "claim_no"].values,
        "pmt_no": test_tx.loc[mask_test, "pmt_no"].values,
        "payment_period": test_tx.loc[mask_test, "payment_period"].values,
        "y_true": ydelay_test,
        "y_pred": preds
    })

    pred_path = os.path.join(
        OUTPUT_DIR,
        f"delay_mlp_predictions_seed_{seed}.csv"
    )
    result_delay_df.to_csv(pred_path, index=False)

    print(f"\n=== Delay RESULTS seed {seed} ===")
    print(f"Delay MSE: {mse}")
    print(f"Delay MAE: {mae}")
    print(f"Delay RR:  {rr}")
    print(f"Predictions saved to: {pred_path}")

    return {
        "seed": seed,
        "Delay_MSE": mse,
        "Delay_MAE": mae,
        "Delay_RR": rr
    }


def run_gan_delay_rows():
    print("Device:", DEVICE)

    all_results = []

    for seed in SEEDS:
        result = run_single_seed(seed)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(
        OUTPUT_DIR,
        "delay_gan_seed_results.csv"
    )
    results_df.to_csv(results_path, index=False)

    # sample standard deviation (ddof=1), matching the 5-seed summary
    summary = pd.DataFrame([{
        "metric": "Delay_MSE",
        "mean": results_df["Delay_MSE"].mean(),
        "std": results_df["Delay_MSE"].std(ddof=1)
    }, {
        "metric": "Delay_MAE",
        "mean": results_df["Delay_MAE"].mean(),
        "std": results_df["Delay_MAE"].std(ddof=1)
    }, {
        "metric": "Delay_RR",
        "mean": results_df["Delay_RR"].mean(),
        "std": results_df["Delay_RR"].std(ddof=1)
    }])

    summary_path = os.path.join(
        OUTPUT_DIR,
        "delay_gan_seed_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL 5-SEED RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("\nMean ± sample SD:")
    for _, row in summary.iterrows():
        print(f"{row['metric']}: {row['mean']:.6f} ± {row['std']:.6f}")

    print(f"\nSeed results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")


# ----------------- run -----------------
if __name__ == "__main__":
    run_gan_delay_rows()

