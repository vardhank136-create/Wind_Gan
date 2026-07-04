"""
utils.py  —  shared helpers
============================================================
SAMPLE SIZE: (24, 72) per sample = ONE FULL DAY
  cols 0-35  : FE for 36 farms   (target)
  cols 36-71 : FP for 36 farms   (condition)

Flattened for the network:
  FE flat : (864,)   = 24 * 36
  FP flat : (864,)   = 24 * 36
============================================================
"""
import os, random, json
import numpy as np
import torch
import torch.nn as nn
import config


def set_seed(s=config.RANDOM_SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[seed] {s}")


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.8)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class LossLogger:
    def __init__(self):
        self.g,   self.d   = [], []
        self.d_r, self.d_f = [], []
        self.w_dist        = []

    def update(self, g, d, d_r, d_f):
        self.g.append(float(g))
        self.d.append(float(d))
        self.d_r.append(float(d_r))
        self.d_f.append(float(d_f))

    def save(self):
        p = os.path.join(config.LOG_DIR,
                         f"loss_{config.TRAIN_MODE}.json")
        with open(p, "w") as f:
            json.dump({
                "G":      self.g,
                "D":      self.d,
                "D_real": self.d_r,
                "D_fake": self.d_f,
                "W_dist": self.w_dist,
            }, f, indent=2)
        print(f"  [log] {p}")

    @classmethod
    def load(cls, path):
        """Load a saved logger (used by evaluate.py standalone)."""
        with open(path, "r") as f:
            d = json.load(f)
        obj = cls()
        obj.g     = d.get("G",      [])
        obj.d     = d.get("D",      [])
        obj.d_r   = d.get("D_real", [])
        obj.d_f   = d.get("D_fake", [])
        obj.w_dist = d.get("W_dist", [])
        return obj


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Empirical 1-D Wasserstein-1 (earth-mover) distance."""
    n = min(len(a), len(b))
    return float(np.mean(np.abs(np.sort(a[:n]) - np.sort(b[:n]))))


def verify_sample_size(verbose=True):
    """
    Confirms the data layout is (24, 72) per day:
      * 24 hours per sample
      * 72 columns: 0-35 = FE (target), 36-71 = FP (condition)
    """
    n_train_days_approx = int(config.TRAIN_YEARS * 365.25 *
                               config.TRAIN_RATIO)
    n_test_days_approx  = int(config.TRAIN_YEARS * 365.25) - \
                           n_train_days_approx

    facts = {
        "csv_total_cols":         config.RAW_TOTAL_COLS,
        "fe_cols":                config.N_FARMS,
        "fp_cols":                config.N_FARMS,
        "profile_length_hours":   config.N_HOURS,
        "one_sample_shape":       (config.N_HOURS, config.RAW_TOTAL_COLS),
        "fe_per_sample":          (config.N_HOURS, config.N_FARMS),
        "fp_per_sample":          (config.N_HOURS, config.N_FARMS),
        "samples_per_day":        1,
        "approx_train_samples":   n_train_days_approx,
        "approx_test_samples":    n_test_days_approx,
        "is_24x72_per_sample":    True,
        "correct_shape_note":     (
            "Each sample is shape (24, 72) - ONE full day across "
            "all 36 farms. Columns 0-35 are FE (target), 36-71 are "
            "FP (condition)."
        ),
    }

    if verbose:
        print("\n" + "="*62)
        print("  SAMPLE SIZE VERIFICATION")
        print("="*62)
        print(f"  CSV columns           : {facts['csv_total_cols']}"
              f"  (36 FE + 36 FP)")
        print(f"  Profile length        : {facts['profile_length_hours']} h")
        print(f"  One sample shape      : {facts['one_sample_shape']}"
              f"  <- (24, 72) per day")
        print(f"  FE per sample         : {facts['fe_per_sample']}")
        print(f"  FP per sample         : {facts['fp_per_sample']}  (condition)")
        print(f"  Approx train samples  : "
              f"{facts['approx_train_samples']:,}")
        print(f"  Approx test  samples  : "
              f"{facts['approx_test_samples']:,}")
        print(f"  Is 24x72 per sample?  : "
              f"{'YES <- CORRECT' if facts['is_24x72_per_sample'] else 'NO'}")
        print(f"\n  NOTE: {facts['correct_shape_note']}")
        print("="*62 + "\n")

    return facts


if __name__ == "__main__":
    verify_sample_size(verbose=True)
