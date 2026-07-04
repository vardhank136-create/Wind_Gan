"""
evaluate.py  -  CGAN Evaluation  (20 sets x 1000 scenarios)
============================================================
HOW TO RUN STANDALONE (after training once):
    python evaluate.py
  - Loads best_G_ema.pt + scalers from outputs_cgan/
  - Loads logger from logs_cgan/
  - Reproduces ALL plots and metrics WITHOUT retraining.

HOW TO RUN FROM main.py (called at end of train):
    from evaluate import evaluate
    evaluate(G, D, logger, fe_te_3d, fp_te_3d,
             fe_scaler, fp_scaler, n_train_days, mode)

METRICS COMPUTED (matching PDF Table 1 + extras):
    Best Wasserstein-1 Distance
    Mean QQ R2
    Mean CRPS (all farms, all hours)  -- avg over 20 sets x 1000 scenarios
    Mean 90% PI Coverage              -- avg over 20 sets
    Mean Inter-Hour Corr. |Delta|
    Spatial Variogram mean |diff|     -- NOT temporal

OUTPUTS:
  loss_curves_{mode}.png
  cdf_hourly_{mode}_farm{F}.png          24-panel hourly CDF
  qq_hourly_{mode}_farm{F}.png           24-panel QQ
  cdf_allfarms_{mode}_hour12.png         36-panel all-farms CDF
  per_farm_coverage_{mode}.png
  generated_set{S:02d}_{mode}_farm{F}_day{D}.csv   (20 files)
  crps_per_farm_hour_{mode}.png          36-panel grid
  crps_random_hour_{mode}_h{H}.png       bar chart at random hour
  variogram_grid_{mode}.png              24-panel spatial variogram grid
  variogram_allfarms_{mode}_hour{H}.png  standalone for H=0,6,12,18
  fan_chart_{mode}_farm{F}.png
  corr_heatmap_{mode}_farm{F}.png
  metrics_{mode}.json                    all metrics as JSON

MEMORY-EFFICIENT CRPS / VARIOGRAM:
  Both computed in batches of BATCH_DAYS=50 test days at a time.
  Per-batch peak RAM: ~173 MB for n_gen=1000, n_batch=50.
  20 sets x 1000 scenarios uses the same batch loop.
============================================================
"""
import os, json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config
from model import Generator
from utils  import LossLogger

# batch size for memory-efficient generation loops
BATCH_DAYS = 50


# ============================================================
#  Generation helpers
# ============================================================

def _set_gen_mode(G):
    """
    Set G to eval mode. If G_EVAL_BN_TRAIN_MODE is True, keep
    BatchNorm layers in train mode (uses batch stats -> wider dist).
    """
    G.eval()
    if config.G_EVAL_BN_TRAIN_MODE:
        for m in G.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()


@torch.no_grad()
def generate_scenarios_for_day(G, fp_3d_day, n=config.N_GEN_SAMPLES,
                                device=config.DEVICE):
    """
    Generate n FE scenarios for ONE test day.
    fp_3d_day : (24, 36) normalised FP.
    Returns   : (n, 24, 36) FE scenarios.
    """
    _set_gen_mode(G)
    fp_flat = (torch.tensor(fp_3d_day.reshape(-1), dtype=torch.float32,
                            device=device)
               .unsqueeze(0).expand(n, -1))
    z   = torch.randn(n, config.NOISE_DIM, device=device)
    out = G(z, fp_flat).cpu().numpy()
    return out.reshape(n, config.N_HOURS, config.N_FARMS)


@torch.no_grad()
def _gen_batch(G, fp_batch_flat, n_gen, device):
    """
    Generate n_gen scenarios for a BATCH of test days.
    fp_batch_flat : (n_batch, FP_DIM)
    Returns       : (n_gen, n_batch, N_HOURS, N_FARMS)
    """
    n_batch = fp_batch_flat.shape[0]
    fp_t    = torch.tensor(fp_batch_flat, dtype=torch.float32, device=device)
    ens     = np.zeros((n_gen, n_batch, config.N_HOURS, config.N_FARMS),
                       dtype=np.float32)
    _set_gen_mode(G)
    for k in range(n_gen):
        z = torch.randn(n_batch, config.NOISE_DIM, device=device)
        f = G(z, fp_t).cpu().numpy()
        ens[k] = f.reshape(n_batch, config.N_HOURS, config.N_FARMS)
    return ens


def _savefig(name):
    p = os.path.join(config.OUTPUT_DIR, name)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [fig] {p}")


# ============================================================
#  Metric helpers
# ============================================================

def crps_ensemble(ensemble: np.ndarray, obs: float) -> float:
    """Energy-score CRPS. O(N log N) via sort trick."""
    N      = len(ensemble)
    t1     = np.mean(np.abs(ensemble - obs))
    ens_s  = np.sort(ensemble)
    idx    = np.arange(N)
    t2     = 2.0 * np.sum((2 * idx - N + 1) * ens_s) / (N * N)
    return float(t1 - 0.5 * t2)


def spatial_variogram(matrix: np.ndarray) -> np.ndarray:
    """
    SPATIAL variogram across 36 farms for ONE hour.
    matrix : (n_samples, 36)  -- FE values across farms.
    Returns gamma(lag) for lag=1..35, shape (35,).
    NOT temporal -- lag is farm-index distance, not time.
    """
    N, F = matrix.shape
    gam  = np.zeros(F - 1, dtype=np.float64)
    for lag in range(1, F):
        diff      = matrix[:, lag:] - matrix[:, :-lag]   # (N, F-lag)
        gam[lag-1] = 0.5 * np.mean(diff ** 2)
    return gam


# ============================================================
#  Loss-curve smoothing
# ============================================================

def _smooth(x, window=25):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return x
    w      = max(2, min(window, max(1, len(x) // 4)))
    pad_lo = np.full(w // 2, x[0])
    pad_hi = np.full(w - 1 - w // 2, x[-1])
    padded = np.concatenate([pad_lo, x, pad_hi])
    return np.convolve(padded, np.ones(w) / w, mode="valid")


# ============================================================
#  1.  Loss curves  (BCE-appropriate)
# ============================================================

def plot_loss_curves(logger, mode=config.TRAIN_MODE):
    """2-panel loss plot with BCE equilibrium references + smoothing."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = np.arange(1, len(logger.g) + 1)

    ax = axes[0]
    if len(ep) >= 4:
        ax.plot(ep, logger.g,   color="tomato",    lw=0.6, alpha=0.35)
        ax.plot(ep, logger.d,   color="steelblue", lw=0.6, alpha=0.35)
        ax.plot(ep, _smooth(logger.g), color="darkred", lw=1.8, label="G loss (smoothed)")
        ax.plot(ep, _smooth(logger.d), color="navy",    lw=1.8, label="D loss (smoothed)")
    else:
        ax.plot(ep, logger.g, color="darkred", lw=1.8, label="G loss")
        ax.plot(ep, logger.d, color="navy",    lw=1.8, label="D loss")
    ax.plot(ep, logger.d_r, color="green",  lw=0.7, ls="--", alpha=0.7, label="D real")
    ax.plot(ep, logger.d_f, color="orange", lw=0.7, ls="--", alpha=0.7, label="D fake")
    ax.axhline(1.33, color="purple", lw=0.8, ls=":", label="D equilibrium ~1.33")
    ax.axhline(0.69, color="gray",   lw=0.6, ls=":", label="G equilibrium ~0.69")
    ax.set_title("G & D Losses (BCE, smoothed)\n"
                 "Equilibrium: G->0.69, D->1.33  (D can't tell real from fake)",
                 fontsize=9)
    ax.set_xlabel("Epoch"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    if logger.w_dist:
        ev = np.arange(1, len(logger.w_dist) + 1)
        ax.plot(ev, logger.w_dist, "go-", lw=1.5, ms=5, label="W-distance (test)")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        if len(ev) >= 4:
            ax.plot(ev, _smooth(logger.w_dist, window=5), color="darkgreen",
                    lw=2.0, alpha=0.7, label="W-distance (smoothed)")
        ax.set_title("Test-set Wasserstein Distance\n"
                     "TRUE convergence signal - target -> 0\n"
                     "Typical good value: 0.02 - 0.05", fontsize=9)
        ax.set_xlabel(f"Eval every {config.EVAL_EVERY} ep")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    plt.suptitle(f"[{mode}] Training Convergence", fontsize=12, y=1.02)
    plt.tight_layout()
    _savefig(f"loss_curves_{mode}.png")


# ============================================================
#  2.  Hourly CDF for farm 0
# ============================================================

def plot_cdf_hourly(G, fe_te_3d, fp_te_3d, n_train_days,
                    mode=config.TRAIN_MODE):
    device  = config.DEVICE
    farm    = config.QQ_FARM_IDX
    day_idx = config.QQ_COND_DAY
    n_gen   = config.N_GEN_SAMPLES
    abs_day = n_train_days + day_idx

    fp_3d_day = fp_te_3d[day_idx]        # (24, 36)
    fp_24     = fp_3d_day[:, farm]       # (24,)
    scens     = generate_scenarios_for_day(G, fp_3d_day, n=n_gen, device=device)

    print(f"\n  [CDF] Farm={farm}  TestDay={day_idx} (abs={abs_day})")
    fig, axes = plt.subplots(4, 6, figsize=(22, 16))

    for h in range(config.N_HOURS):
        row, col = divmod(h, 6)
        ax       = axes[row, col]

        fp_cond  = float(fp_24[h])
        real_now = float(fe_te_3d[day_idx, h, farm])
        gen_h    = scens[:, h, farm]

        fp_all_h = fp_te_3d[:, h, farm]
        # Use a slightly wider window (0.20) so the blue reference
        # curve includes more similar days -> fairer comparison
        mask = np.abs(fp_all_h - fp_cond) <= 0.20
        if mask.sum() < 10:
            mask = np.abs(fp_all_h - fp_cond) <= np.percentile(
                np.abs(fp_all_h - fp_cond), 30)
        real_cond = fe_te_3d[mask, h, farm]
        n_cond    = len(real_cond)
        real_all  = fe_te_3d[:, h, farm]

        ax.plot(np.sort(real_all),
                np.arange(1, len(real_all)+1)/len(real_all),
                color="steelblue", lw=1.0, alpha=0.4, ls="--")
        ax.plot(np.sort(real_cond),
                np.arange(1, n_cond+1)/n_cond,
                color="steelblue", lw=1.5)
        ax.plot(np.sort(gen_h),
                np.arange(1, n_gen+1)/n_gen,
                color="darkorange", lw=1.5)
        ax.axvline(real_now, color="black", lw=1.0, ls="--", alpha=0.9)

        ax.set_title(f"H{h:02d}  FP={fp_cond:.3f}\nFE_real={real_now:.3f}  n_sim={n_cond}",
                     fontsize=5.8)
        ax.tick_params(labelsize=5); ax.grid(alpha=0.25)

    handles = [
        Line2D([0],[0], color="steelblue",  lw=1.5,
               label="Real CDF (FP-similar days)"),
        Line2D([0],[0], color="steelblue",  lw=1.0, ls="--", alpha=0.4,
               label="Real CDF (all test days)"),
        Line2D([0],[0], color="darkorange", lw=1.5,
               label=f"Generated ({n_gen} scenarios)"),
        Line2D([0],[0], color="black",      lw=1.0, ls="--",
               label=f"Real FE - test day {day_idx}"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8,
               bbox_to_anchor=(1.0, 1.01))
    fig.suptitle(
        f"[{mode}]  Hourly CDF - Farm {farm}  |  TestDay {day_idx} "
        f"(abs={abs_day})  |  n_gen={n_gen}",
        fontsize=11, y=1.02)
    plt.tight_layout()
    _savefig(f"cdf_hourly_{mode}_farm{farm}.png")
    return scens


# ============================================================
#  3.  Hourly QQ for farm 0
# ============================================================

def plot_qq_hourly(scens, fe_te_3d, fp_te_3d, n_train_days,
                   mode=config.TRAIN_MODE):
    farm     = config.QQ_FARM_IDX
    day_idx  = config.QQ_COND_DAY
    abs_day  = n_train_days + day_idx
    q_levels = np.linspace(0.05, 0.95, 19)

    fig, axes = plt.subplots(4, 6, figsize=(22, 16))
    all_r2 = []

    for h in range(config.N_HOURS):
        row, col = divmod(h, 6)
        ax       = axes[row, col]

        fp_cond  = float(fp_te_3d[day_idx, h, farm])
        real_now = float(fe_te_3d[day_idx, h, farm])
        gen_h    = scens[:, h, farm]

        fp_all_h = fp_te_3d[:, h, farm]
        mask = np.abs(fp_all_h - fp_cond) <= 0.20
        if mask.sum() < 10:
            mask = np.abs(fp_all_h - fp_cond) <= np.percentile(
                np.abs(fp_all_h - fp_cond), 30)
        real_cond = fe_te_3d[mask, h, farm]
        n_cond    = len(real_cond)

        gq = np.quantile(gen_h,    q_levels)
        rq = np.quantile(real_cond, q_levels)

        lo = min(gq.min(), rq.min()) - 0.05
        hi = max(gq.max(), rq.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.scatter(gq, rq, s=16, color="tomato", alpha=0.9, zorder=5)
        ax.scatter([real_now], [real_now], s=40,
                   color="darkorange", marker="D", zorder=7)

        ss_r = np.sum((rq - gq)**2)
        ss_t = np.sum((rq - rq.mean())**2) + 1e-12
        r2   = max(0.0, 1.0 - ss_r/ss_t)
        all_r2.append(r2)

        ax.set_title(f"H{h:02d}  R2={r2:.2f}  FP={fp_cond:.3f}  n={n_cond}",
                     fontsize=5.8)
        ax.set_xlabel("Gen Q", fontsize=5); ax.set_ylabel("Real Q", fontsize=5)
        ax.tick_params(labelsize=5); ax.grid(alpha=0.25)

    mean_r2 = float(np.mean(all_r2))
    fig.suptitle(
        f"[{mode}]  Hourly QQ - Farm {farm}  |  TestDay {day_idx} (abs={abs_day})\n"
        f"X=gen quantiles  Y=real quantiles (FP-similar days)  |  Mean R2={mean_r2:.3f}",
        fontsize=10, y=1.02)
    plt.tight_layout()
    _savefig(f"qq_hourly_{mode}_farm{farm}.png")
    print(f"  [QQ]  Mean R2={mean_r2:.3f}")
    return mean_r2


# ============================================================
#  4.  All-farms CDF at hour 12
# ============================================================

def plot_cdf_allfarms(G, fe_te_3d, fp_te_3d, n_train_days,
                      hour=12, mode=config.TRAIN_MODE):
    device  = config.DEVICE
    day_idx = config.QQ_COND_DAY
    n_gen   = config.N_GEN_SAMPLES
    abs_day = n_train_days + day_idx

    scens = generate_scenarios_for_day(G, fp_te_3d[day_idx],
                                        n=n_gen, device=device)

    fig, axes = plt.subplots(6, 6, figsize=(22, 22))
    for f in range(config.N_FARMS):
        row, col = divmod(f, 6)
        ax       = axes[row, col]

        fp_val   = float(fp_te_3d[day_idx, hour, f])
        gen_h    = scens[:, hour, f]
        real_all = fe_te_3d[:, hour, f]
        real_now = float(fe_te_3d[day_idx, hour, f])

        mask = np.abs(fp_te_3d[:, hour, f] - fp_val) <= 0.20
        if mask.sum() < 10:
            mask = np.abs(fp_te_3d[:, hour, f] - fp_val) <= np.percentile(
                np.abs(fp_te_3d[:, hour, f] - fp_val), 30)
        real_cond = fe_te_3d[mask, hour, f]

        ax.plot(np.sort(real_all),
                np.arange(1, len(real_all)+1)/len(real_all),
                color="steelblue", lw=0.7, ls="--", alpha=0.4)
        ax.plot(np.sort(real_cond),
                np.arange(1, len(real_cond)+1)/len(real_cond),
                color="steelblue", lw=1.0)
        ax.plot(np.sort(gen_h),
                np.arange(1, n_gen+1)/n_gen,
                color="darkorange", lw=1.0)
        ax.axvline(real_now, color="black", lw=0.7, ls="--")
        ax.set_title(f"F{f}  FP={fp_val:.2f}", fontsize=5.5)
        ax.tick_params(labelsize=4); ax.grid(alpha=0.2)

    fig.suptitle(
        f"[{mode}]  All-Farm CDF at Hour {hour:02d}  |  "
        f"TestDay {day_idx} (abs={abs_day})",
        fontsize=11, y=1.005)
    plt.tight_layout()
    _savefig(f"cdf_allfarms_{mode}_hour{hour:02d}.png")


# ============================================================
#  5.  Per-farm 90% PI coverage  (averaged over N_EVAL_SETS)
# ============================================================

def plot_farm_coverage(G, fe_te_3d, fp_te_3d, mode=config.TRAIN_MODE,
                       n_sets=config.N_EVAL_SETS):
    device = config.DEVICE
    n_test = fe_te_3d.shape[0]
    n_gen  = config.N_GEN_SAMPLES

    cov_per_set = np.zeros((n_sets, config.N_FARMS), dtype=np.float32)
    print(f"  [Coverage] {n_sets} sets x {n_gen} scenarios per day ...")

    for s in range(n_sets):
        # batch over test days to stay memory-efficient
        q5_all  = np.zeros((n_test, config.N_HOURS, config.N_FARMS), dtype=np.float32)
        q95_all = np.zeros_like(q5_all)

        for d_start in range(0, n_test, BATCH_DAYS):
            d_end   = min(d_start + BATCH_DAYS, n_test)
            fp_flat = fp_te_3d[d_start:d_end].reshape(d_end - d_start, -1)
            ens     = _gen_batch(G, fp_flat, n_gen, device)  # (n_gen,nb,24,36)
            q5_all [d_start:d_end] = np.quantile(ens, 0.05, axis=0)
            q95_all[d_start:d_end] = np.quantile(ens, 0.95, axis=0)

        for farm in range(config.N_FARMS):
            real = fe_te_3d[:, :, farm]
            q5   = q5_all [:, :, farm]
            q95  = q95_all[:, :, farm]
            cov  = np.mean((real >= q5) & (real <= q95)) * 100
            cov_per_set[s, farm] = cov
        print(f"    coverage set {s+1}/{n_sets} done")

    cov_per_farm = cov_per_set.mean(axis=0)
    mean_cov     = float(cov_per_farm.mean())

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["steelblue" if c >= 90 else "tomato" for c in cov_per_farm]
    ax.bar(range(config.N_FARMS), cov_per_farm, color=colors, alpha=0.85)
    ax.axhline(90, color="green",  lw=2.0, ls="--", label="90% target")
    ax.axhline(95, color="orange", lw=1.5, ls="--", label="95% stretch")
    ax.set_xticks(range(config.N_FARMS))
    ax.set_xticklabels([f"F{i}" for i in range(36)], fontsize=7, rotation=45)
    ax.set_ylabel("Coverage %  (real FE inside Gen Q5-Q95)")
    ax.set_ylim(40, 105)
    ax.set_title(
        f"[{mode}]  Per-Farm Coverage  |  Mean={mean_cov:.1f}%  (target >= 90%)\n"
        f"Averaged over {n_sets} sets x {n_gen} scenarios",
        fontsize=11)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, c in enumerate(cov_per_farm):
        ax.text(i, c+0.5, f"{c:.0f}", ha="center", fontsize=5.5,
                weight="bold", color="navy" if c >= 90 else "white")
    plt.tight_layout()
    _savefig(f"per_farm_coverage_{mode}.png")
    print(f"  [Coverage] Mean={mean_cov:.1f}%")
    return mean_cov


# ============================================================
#  6.  20 CSV files  (one per evaluation set)
# ============================================================

def save_generated_csvs_all_sets(G, fe_te_3d, fp_te_3d,
                                  fe_scaler, fp_scaler, n_train_days,
                                  mode=config.TRAIN_MODE,
                                  n_sets=config.N_EVAL_SETS):
    device  = config.DEVICE
    farm    = config.QQ_FARM_IDX
    day_idx = config.QQ_COND_DAY
    n_gen   = config.N_GEN_SAMPLES
    abs_day = n_train_days + day_idx

    fp_3d_day = fp_te_3d[day_idx]              # (24, 36)
    fp_24     = fp_3d_day[:, farm]
    fe_24     = fe_te_3d[day_idx, :, farm]
    fp_orig   = fp_scaler.inverse_transform(
                    fp_3d_day.reshape(1, -1)).reshape(24, 36)[:, farm]
    fe_orig   = fe_scaler.inverse_transform(
                    fe_te_3d[day_idx].reshape(1, -1)).reshape(24, 36)[:, farm]

    print(f"\n  [CSV] Saving {n_sets} sets of {n_gen} scenarios "
          f"(farm={farm}, day={day_idx})")

    for s in range(n_sets):
        scens_3d = generate_scenarios_for_day(G, fp_3d_day,
                                               n=n_gen, device=device)
        scens = scens_3d[:, :, farm]   # (n_gen, 24)

        rows = []
        for h in range(config.N_HOURS):
            row = {
                "set":               s + 1,
                "hour":              h,
                "csv_row_0idx":      abs_day * config.N_HOURS + h,
                "fp_col_0idx":       36 + farm,
                "fe_col_0idx":       farm,
                "fp_condition_norm": round(float(fp_24[h]), 6),
                "fe_real_norm":      round(float(fe_24[h]), 6),
                "fp_condition_orig": round(float(fp_orig[h]), 6),
                "fe_real_orig":      round(float(fe_orig[h]), 6),
            }
            for k in range(n_gen):
                row[f"fe_gen_{k+1}"] = round(float(scens[k, h]), 6)
            rows.append(row)

        p = os.path.join(config.OUTPUT_DIR,
                         f"generated_set{s+1:02d}_{mode}"
                         f"_farm{farm}_day{day_idx}.csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"  [CSV]  set {s+1:02d}/{n_sets} -> {p}")


# ============================================================
#  7.  CRPS per farm per hour  (36-panel grid + bar chart)
#      Averaged over N_EVAL_SETS x N_GEN_SAMPLES
#      Batch-efficient: BATCH_DAYS test days at a time
# ============================================================

def plot_crps_all_farms(G, fe_te_3d, fp_te_3d,
                        mode=config.TRAIN_MODE,
                        n_sets=config.N_EVAL_SETS,
                        random_hour=None):
    device = config.DEVICE
    n_test = fe_te_3d.shape[0]
    n_gen  = config.N_GEN_SAMPLES

    rng = np.random.default_rng(config.RANDOM_SEED + 7)
    if random_hour is None:
        random_hour = int(rng.integers(0, config.N_HOURS))
    print(f"  [CRPS] random hour selected = {random_hour}")
    print(f"  [CRPS] {n_sets} sets x {n_gen} scenarios  "
          f"(batched, BATCH_DAYS={BATCH_DAYS})")

    # crps_per_set[s, farm, h] = mean CRPS over test days
    crps_per_set = np.zeros((n_sets, config.N_FARMS, config.N_HOURS),
                            dtype=np.float32)

    for s in range(n_sets):
        crps_accum = np.zeros((config.N_FARMS, config.N_HOURS),
                              dtype=np.float64)

        for d_start in range(0, n_test, BATCH_DAYS):
            d_end   = min(d_start + BATCH_DAYS, n_test)
            n_batch = d_end - d_start
            fp_flat = fp_te_3d[d_start:d_end].reshape(n_batch, -1)

            # ens: (n_gen, n_batch, 24, 36)
            ens = _gen_batch(G, fp_flat, n_gen, device)

            for farm in range(config.N_FARMS):
                for h in range(config.N_HOURS):
                    obs_bh = fe_te_3d[d_start:d_end, h, farm]  # (n_batch,)
                    ens_bh = ens[:, :, h, farm]                  # (n_gen, n_batch)
                    for dd in range(n_batch):
                        crps_accum[farm, h] += crps_ensemble(
                            ens_bh[:, dd], obs_bh[dd])

        crps_per_set[s] = (crps_accum / n_test).astype(np.float32)
        print(f"    CRPS set {s+1}/{n_sets} done")

    crps_matrix = crps_per_set.mean(axis=0)   # (36, 24)

    # ---- 36-panel grid: hour vs CRPS ----
    fig, axes = plt.subplots(6, 6, figsize=(24, 20))
    hours      = np.arange(config.N_HOURS)
    farm_mean  = crps_matrix.mean(axis=1)
    vmax       = float(crps_matrix.max())

    for f in range(config.N_FARMS):
        row, col = divmod(f, 6)
        ax = axes[row, col]
        ax.plot(hours, crps_matrix[f], color="steelblue",
                lw=1.4, marker="o", ms=3)
        ax.axvline(random_hour, color="tomato", lw=1.0, ls="--", alpha=0.8,
                   label=f"H{random_hour}")
        ax.fill_between(hours, crps_matrix[f], alpha=0.15, color="steelblue")
        ax.set_title(f"Farm {f}  |  Mean CRPS={farm_mean[f]:.4f}", fontsize=6)
        ax.set_xlabel("Hour", fontsize=5); ax.set_ylabel("CRPS", fontsize=5)
        ax.set_xlim(-0.5, 23.5); ax.set_ylim(0, vmax * 1.1)
        ax.tick_params(labelsize=4.5); ax.grid(alpha=0.25)
        if f == 0:
            ax.legend(fontsize=4, loc="upper right")

    fig.suptitle(
        f"[{mode}]  CRPS per Farm per Hour\n"
        f"X=hour (0-23)  Y=mean CRPS over {n_test} test days  "
        f"|  Red dashed = H{random_hour}\n"
        f"Averaged over {n_sets} sets x {n_gen} scenarios  "
        f"|  Global mean CRPS = {crps_matrix.mean():.4f}",
        fontsize=12, y=1.01)
    plt.tight_layout()
    _savefig(f"crps_per_farm_hour_{mode}.png")

    # ---- bar chart at random hour ----
    fig2, ax2 = plt.subplots(figsize=(16, 5))
    farm_ids  = np.arange(config.N_FARMS)
    crps_at_h = crps_matrix[:, random_hour]
    colors_bar = plt.cm.RdYlGn_r(
        (crps_at_h - crps_at_h.min()) /
        (crps_at_h.max() - crps_at_h.min() + 1e-8))
    ax2.bar(farm_ids, crps_at_h, color=colors_bar, alpha=0.9, width=0.7)
    ax2.set_xticks(farm_ids)
    ax2.set_xticklabels([f"F{i}" for i in range(config.N_FARMS)],
                        fontsize=7, rotation=45)
    ax2.set_ylabel("CRPS  (lower = better)", fontsize=10)
    ax2.set_xlabel("Wind Farm", fontsize=10)
    ax2.axhline(crps_at_h.mean(), color="navy", lw=1.5, ls="--",
                label=f"Mean CRPS = {crps_at_h.mean():.4f}")
    for i, v in enumerate(crps_at_h):
        ax2.text(i, v + 0.0005, f"{v:.4f}", ha="center",
                 va="bottom", fontsize=4.5, rotation=90)
    ax2.set_title(
        f"[{mode}]  CRPS at randomly selected Hour H{random_hour}  "
        f"for All 36 Wind Farms\n"
        f"Averaged over {n_sets} sets x {n_gen} scenarios  |  "
        f"{n_test} test days  |  Green=low (good) -> Red=high (bad)",
        fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig(f"crps_random_hour_{mode}_h{random_hour:02d}.png")

    mean_crps = float(crps_matrix.mean())
    print(f"  [CRPS] Global mean = {mean_crps:.5f}")
    return crps_matrix, random_hour, crps_per_set


# ============================================================
#  8.  SPATIAL variogram per hour  (NOT temporal)
#      Spatial: gamma(lag) across 36 farms at each hour
# ============================================================

def plot_variograms_spatial_per_hour(G, fe_te_3d, fp_te_3d,
                                     mode=config.TRAIN_MODE,
                                     n_sets=config.N_EVAL_SETS):
    device  = config.DEVICE
    n_test  = fe_te_3d.shape[0]
    n_farms = config.N_FARMS
    n_gen   = config.N_GEN_SAMPLES
    max_lag = n_farms - 1          # 35
    lags    = np.arange(1, max_lag + 1)

    # Real spatial variogram: for each hour h, (n_test, 36) -> gamma (35,)
    print("  [Variogram-spatial] computing real ...")
    real_var = np.zeros((config.N_HOURS, max_lag), dtype=np.float64)
    for h in range(config.N_HOURS):
        real_var[h] = spatial_variogram(fe_te_3d[:, h, :])

    # Generated spatial variogram: batch-efficient accumulation
    print(f"  [Variogram-spatial] generating {n_sets} sets x {n_gen} "
          f"scenarios (batched, BATCH_DAYS={BATCH_DAYS}) ...")
    gen_var_per_set = np.zeros((n_sets, config.N_HOURS, max_lag),
                               dtype=np.float64)

    for s in range(n_sets):
        # gam_sum[h, lag] accumulates sum of (0.5 * mean_sq_diff)
        # weighted by number of samples in each batch
        gam_sum  = np.zeros((config.N_HOURS, max_lag), dtype=np.float64)
        n_total  = 0

        for d_start in range(0, n_test, BATCH_DAYS):
            d_end   = min(d_start + BATCH_DAYS, n_test)
            n_batch = d_end - d_start
            fp_flat = fp_te_3d[d_start:d_end].reshape(n_batch, -1)

            ens = _gen_batch(G, fp_flat, n_gen, device)  # (n_gen, nb, 24, 36)
            # Merge n_gen and n_batch into one "sample" dimension
            # shape: (n_gen * n_batch, 24, 36)
            all_sc = ens.reshape(-1, config.N_HOURS, n_farms)

            for h in range(config.N_HOURS):
                mat = all_sc[:, h, :]         # (n_gen*n_batch, 36)
                for lag in range(1, n_farms):
                    diff = mat[:, lag:] - mat[:, :-lag]  # (n*nb, 36-lag)
                    # weight by number of (farm-pair, sample) instances
                    gam_sum[h, lag-1] += (
                        0.5 * np.sum(diff**2) / (n_farms - lag))

            n_total += n_gen * n_batch

        gen_var_per_set[s] = gam_sum / n_total
        print(f"    Variogram set {s+1}/{n_sets} done")

    gen_var_mean = gen_var_per_set.mean(axis=0)    # (24, 35)
    var_abs_diff = np.abs(real_var - gen_var_mean)
    mean_var_diff = float(var_abs_diff.mean())
    print(f"  [Variogram-spatial] mean |real-gen| = {mean_var_diff:.5f}")

    # ---- 24-panel grid (4 rows x 6 cols) ----
    fig, axes = plt.subplots(4, 6, figsize=(24, 14))
    vmax = float(max(real_var.max(), gen_var_mean.max()))

    for h in range(config.N_HOURS):
        row, col = divmod(h, 6)
        ax = axes[row, col]
        ax.plot(lags, real_var[h], color="steelblue", lw=1.5, label="Real")
        ax.plot(lags, gen_var_mean[h], color="darkorange", lw=1.5,
                ls="--", label="Generated")
        ax.fill_between(lags, real_var[h], gen_var_mean[h],
                        color="gray", alpha=0.12)
        ax.set_title(f"Hour {h:02d}", fontsize=7)
        ax.set_xlabel("Farm-pair lag", fontsize=5)
        ax.set_ylabel("gamma(lag)",    fontsize=5)
        ax.set_ylim(0, vmax * 1.05)
        ax.tick_params(labelsize=4.5); ax.grid(alpha=0.25)
        if h == 0:
            ax.legend(fontsize=5)

    fig.suptitle(
        f"[{mode}]  Spatial Variogram across 36 Farms - per Hour\n"
        f"X=farm-pair lag (1..35)  Y=gamma  "
        f"(Gen avg of {n_sets} sets x {n_gen} scenarios)\n"
        f"Mean |real-gen| = {mean_var_diff:.5f}  "
        f"(NOT temporal - lag is farm index distance)",
        fontsize=12, y=1.01)
    plt.tight_layout()
    _savefig(f"variogram_grid_{mode}.png")

    # ---- standalone plots for representative hours ----
    for h in [0, 6, 12, 18]:
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(lags, real_var[h], color="steelblue", lw=2.0,
                 marker="o", ms=4, label="Real (all test days)")
        ax2.plot(lags, gen_var_mean[h], color="darkorange", lw=2.0,
                 marker="s", ms=4, ls="--",
                 label=f"Generated (avg of {n_sets} sets)")
        ax2.fill_between(lags, real_var[h], gen_var_mean[h],
                         color="gray", alpha=0.15)
        ax2.set_xlabel("Farm-pair lag (|i - j|)", fontsize=10)
        ax2.set_ylabel("Spatial variogram gamma(lag)", fontsize=10)
        ax2.set_title(
            f"[{mode}]  Spatial Variogram across 36 Farms - Hour {h:02d}\n"
            f"Mean |real-gen| at this hour = {var_abs_diff[h].mean():.5f}",
            fontsize=11)
        ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        _savefig(f"variogram_allfarms_{mode}_hour{h:02d}.png")

    return real_var, gen_var_mean, mean_var_diff


# ============================================================
#  9.  Fan chart  (30-day window, 50/70/95% PI)
# ============================================================

def plot_fan_chart(G, fe_te_3d, fp_te_3d, mode=config.TRAIN_MODE):
    device    = config.DEVICE
    farm      = config.FAN_CHART_FARM
    start_day = config.FAN_CHART_START_DAY
    n_gen     = config.N_GEN_SAMPLES
    n_days    = min(config.FAN_CHART_N_DAYS, fe_te_3d.shape[0] - start_day)

    q_lo = {"95": 0.025, "70": 0.15, "50": 0.25}
    q_hi = {"95": 0.975, "70": 0.85, "50": 0.75}
    colors = {"95": "#d6eaf8", "70": "#7fb3d3", "50": "#2980b9"}
    labels = {"95": "95% PI", "70": "70% PI", "50": "50% PI"}

    all_lo = {k: [] for k in q_lo}; all_hi = {k: [] for k in q_hi}
    all_med = []; all_real = []

    print(f"  [Fan] Farm={farm}  days {start_day}-{start_day+n_days-1}")
    for d in range(start_day, start_day + n_days):
        scens = generate_scenarios_for_day(G, fp_te_3d[d],
                                            n=n_gen, device=device)
        for h in range(config.N_HOURS):
            col_h = scens[:, h, farm]
            for k in q_lo:
                all_lo[k].append(float(np.quantile(col_h, q_lo[k])))
                all_hi[k].append(float(np.quantile(col_h, q_hi[k])))
            all_med.append(float(np.median(col_h)))
            all_real.append(float(fe_te_3d[d, h, farm]))

    T = len(all_med); x = np.arange(T)
    fig, ax = plt.subplots(figsize=(20, 6))
    for k in ["95", "70", "50"]:
        ax.fill_between(x, all_lo[k], all_hi[k],
                        color=colors[k], alpha=0.85, label=labels[k])
    ax.plot(x, all_med,  color="navy",  lw=1.2, label="Median (generated)")
    ax.plot(x, all_real, color="black", lw=1.0, ls="--",
            label="Real FE (test)", alpha=0.9)
    day_ticks = np.arange(0, T, config.N_HOURS)
    for dt in day_ticks:
        ax.axvline(dt, color="gray", lw=0.4, ls=":", alpha=0.6)
    ax.set_xticks(day_ticks + 12)
    ax.set_xticklabels([f"D{start_day+i}" for i in range(n_days)],
                       fontsize=7, rotation=45)
    ax.set_xlabel("Test Day  (each block = 24 hours)", fontsize=10)
    ax.set_ylabel("Normalised FE", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"[{mode}]  Fan Chart - Wind Farm {farm}  |  "
        f"{n_days}-day window  |  {n_gen} scenarios/hour",
        fontsize=11)
    plt.tight_layout()
    _savefig(f"fan_chart_{mode}_farm{farm}.png")


# ============================================================
#  10.  Inter-hour correlation heatmap  (24x24 Pearson)
# ============================================================

def plot_correlation_heatmaps(G, fe_te_3d, fp_te_3d,
                              mode=config.TRAIN_MODE, farm=None):
    device = config.DEVICE
    n_test = fe_te_3d.shape[0]
    n_gen  = config.N_GEN_SAMPLES
    if farm is None:
        farm = config.QQ_FARM_IDX

    # Real correlation
    real_mat = fe_te_3d[:, :, farm]       # (n_test, 24)
    C_real   = np.corrcoef(real_mat.T)

    # Generated: average n_gen scenarios per day, then correlate
    fp_flat = torch.tensor(fp_te_3d.reshape(n_test, -1),
                           dtype=torch.float32, device=device)
    gen_acc = np.zeros((n_test, config.N_HOURS), dtype=np.float64)
    with torch.no_grad():
        _set_gen_mode(G)
        for _ in range(n_gen):
            z = torch.randn(n_test, config.NOISE_DIM, device=device)
            f = G(z, fp_flat).cpu().numpy()
            gen_acc += f.reshape(n_test, config.N_HOURS,
                                  config.N_FARMS)[:, :, farm]
    gen_acc /= n_gen
    C_gen   = np.corrcoef(gen_acc.T)
    C_diff  = np.abs(C_real - C_gen)

    hour_labels = [f"H{h:02d}" for h in range(config.N_HOURS)]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    data_list  = [C_real, C_gen, C_diff]
    titles     = ["Real Correlation", "Generated Correlation",
                  "|Real - Generated|"]
    cmaps      = ["RdBu_r", "RdBu_r", "Reds"]
    vmins      = [-1, -1, 0]; vmaxs = [1, 1, 1]

    for ax, mat, title, cmap, vmin, vmax in zip(
            axes, data_list, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(config.N_HOURS))
        ax.set_yticks(range(config.N_HOURS))
        ax.set_xticklabels(hour_labels, fontsize=5, rotation=90)
        ax.set_yticklabels(hour_labels, fontsize=5)
        ax.set_title(title, fontsize=11, pad=8)
        if title.startswith("|"):
            for i in range(config.N_HOURS):
                for j in range(config.N_HOURS):
                    if mat[i, j] > 0.1:
                        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center",
                                va="center", fontsize=3, color="white")

    mean_abs_diff = float(C_diff.mean())
    fig.suptitle(
        f"[{mode}]  Inter-Hour Correlation Heatmaps - Farm {farm}\n"
        f"Real vs Generated (averaged over {n_gen} samples per test day)  "
        f"|  Mean |diff| = {mean_abs_diff:.4f}\n"
        f"Good model: Real ≈ Generated  |  |diff| ≈ 0",
        fontsize=12, y=1.02)
    plt.tight_layout()
    _savefig(f"corr_heatmap_{mode}_farm{farm}.png")
    print(f"  [Corr] Mean |diff|={mean_abs_diff:.4f}  farm={farm}")
    return mean_abs_diff


# ============================================================
#  Main evaluate entry point
# ============================================================

def evaluate(G, D, logger, fe_te_3d, fp_te_3d,
             fe_scaler, fp_scaler, n_train_days,
             mode=config.TRAIN_MODE):
    farm    = config.QQ_FARM_IDX
    day     = config.QQ_COND_DAY
    abs_day = n_train_days + day

    print(f"\n{'='*62}")
    print(f"  Evaluation  ({config.N_EVAL_SETS} sets x "
          f"{config.N_GEN_SAMPLES} scenarios)")
    print(f"  Farm={farm}  TestDay={day}  AbsDay={abs_day}")
    print(f"  Sample shape : (24, 72) per day")
    print(f"{'='*62}")

    # ---- standard plots ----
    plot_loss_curves(logger, mode)

    scens_day = plot_cdf_hourly(G, fe_te_3d, fp_te_3d, n_train_days, mode)
    mean_r2   = plot_qq_hourly(scens_day, fe_te_3d, fp_te_3d,
                               n_train_days, mode)

    plot_cdf_allfarms(G, fe_te_3d, fp_te_3d, n_train_days,
                      hour=12, mode=mode)

    # ---- metrics (averaged over N_EVAL_SETS x N_GEN_SAMPLES) ----
    mean_cov = plot_farm_coverage(G, fe_te_3d, fp_te_3d, mode)

    save_generated_csvs_all_sets(G, fe_te_3d, fp_te_3d,
                                  fe_scaler, fp_scaler,
                                  n_train_days, mode)

    crps_matrix, rnd_hour, crps_per_set = plot_crps_all_farms(
        G, fe_te_3d, fp_te_3d, mode)

    real_var, gen_var, var_abs_diff = plot_variograms_spatial_per_hour(
        G, fe_te_3d, fp_te_3d, mode)

    plot_fan_chart(G, fe_te_3d, fp_te_3d, mode)

    corr_diff = plot_correlation_heatmaps(G, fe_te_3d, fp_te_3d, mode)

    # ---- aggregate scalars ----
    mean_crps      = float(crps_matrix.mean())
    crps_set_means = crps_per_set.reshape(config.N_EVAL_SETS, -1).mean(axis=1)
    crps_set_std   = float(crps_set_means.std())
    mean_var_diff  = float(var_abs_diff)

    # Best W from logger (minimum W-distance seen during training)
    best_w_log = float(min(logger.w_dist)) if logger.w_dist else float("nan")

    results = {
        # --- PDF Table 1 metrics ---
        "best_wasserstein_1_distance": best_w_log,
        "mean_qq_r2":                  mean_r2,
        "mean_crps_all_farms_hours":   mean_crps,
        "mean_90pct_pi_coverage":      mean_cov,
        "mean_interhour_corr_absdiff": corr_diff,
        # --- additional metrics ---
        "variogram_mean_abs_diff":     mean_var_diff,
        "crps_per_set_mean":           crps_set_means.tolist(),
        "crps_set_std":                crps_set_std,
        "crps_random_hour":            int(rnd_hour),
        "n_eval_sets":                 config.N_EVAL_SETS,
        "n_gen_per_set":               config.N_GEN_SAMPLES,
        "farm_plotted":                farm,
        "condition_day":               day,
        "abs_day":                     abs_day,
        "sample_shape":
            "(24, 72) per day - cols 0-35=FE, 36-71=FP",
    }

    with open(os.path.join(config.OUTPUT_DIR,
                           f"metrics_{mode}.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- print metrics table matching PDF Table 1 ----
    print(f"\n{'='*62}")
    print(f"  EVALUATION RESULTS  "
          f"(avg of {config.N_EVAL_SETS} sets x {config.N_GEN_SAMPLES} scenarios)")
    print(f"{'='*62}")
    print(f"  Metric                         | Value")
    print(f"  -------------------------------|--------")
    print(f"  Best Wasserstein-1 Distance    | {best_w_log:.4f}")
    print(f"  Mean QQ R^2                    | {mean_r2:.3f}")
    print(f"  Mean CRPS (all farms/hours)    | {mean_crps:.4f}")
    print(f"  Mean 90% PI Coverage           | {mean_cov:.1f}%")
    print(f"  Mean Inter-Hour Corr. |Delta|  | {corr_diff:.4f}")
    print(f"  Variogram Mean |diff|          | {mean_var_diff:.4f}")
    print(f"  CRPS std across {config.N_EVAL_SETS} sets       | {crps_set_std:.4f}")
    print(f"{'='*62}")
    print(f"\n  Outputs saved to : {config.OUTPUT_DIR}/")
    return results


# ============================================================
#  Standalone runner  (python evaluate.py)
# ============================================================

def _load_trained_model():
    """Load best EMA generator from saved checkpoint."""
    G_ema_path = os.path.join(config.CHECKPOINT_DIR, "best_G_ema.pt")
    full_ckpt  = os.path.join(config.CHECKPOINT_DIR, "best.pt")

    G = Generator().to(config.DEVICE)
    if os.path.exists(G_ema_path):
        G.load_state_dict(
            torch.load(G_ema_path, map_location=config.DEVICE))
        print(f"[load] G_ema -> {G_ema_path}")
    elif os.path.exists(full_ckpt):
        ck = torch.load(full_ckpt, map_location=config.DEVICE)
        G.load_state_dict(ck["G_ema"])
        print(f"[load] G_ema from {full_ckpt}")
    else:
        raise FileNotFoundError(
            "No checkpoint found. Run main.py to train first.")
    G.eval()
    return G


def main_standalone():
    """
    Run full evaluation using the saved trained model.
    Call this after training has completed once.
    No retraining needed.
    """
    from data_loading import load_test_only, fe_scaler, fp_scaler

    print("=" * 62)
    print("  evaluate.py  -  STANDALONE MODE")
    print("  Loading trained model + test data, no retraining")
    print("=" * 62)

    G = _load_trained_model()
    fe_te_3d, fp_te_3d, n_train_days, n_test_days = load_test_only()
    print(f"[load] test set: {n_test_days} days, "
          f"n_train_days={n_train_days}")

    log_path = os.path.join(config.LOG_DIR,
                            f"loss_{config.TRAIN_MODE}.json")
    if os.path.exists(log_path):
        logger = LossLogger.load(log_path)
        print(f"[load] logger -> {log_path}")
    else:
        logger = LossLogger()
        print("[load] no logger found - loss curves will be empty")

    evaluate(G, None, logger, fe_te_3d, fp_te_3d,
             fe_scaler, fp_scaler,
             n_train_days, config.TRAIN_MODE)


if __name__ == "__main__":
    main_standalone()
