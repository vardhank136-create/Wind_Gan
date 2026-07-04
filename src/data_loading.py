"""
data_loading.py  -  Wind Farm CGAN  (24 x 72 per sample)
============================================================
SAMPLE CONSTRUCTION:
  For day d:
    FP_sample[d] = FP_raw[d*24 : (d+1)*24, :]   shape (24, 36)
    FE_sample[d] = FE_raw[d*24 : (d+1)*24, :]   shape (24, 36)

  Combined sample shape : (24, 72)
    cols 0-35  : FE (target)
    cols 36-71 : FP (condition)

  For the network we flatten:
    FE_flat : (864,)   row-major (hour, farm)
    FP_flat : (864,)   row-major (hour, farm)

SPLIT:
  First TRAIN_YEARS=6 years, chronological 80/20 by day.
  Scaler fitted on TRAIN only, applied to train+test.
============================================================
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config
from utils import verify_sample_size


# -- Scaler ----------------------------------------------------

class GlobalMinMax:
    """Single global (min, max) -> [0,1]. Same as before."""
    def __init__(self):
        self.min_   = None
        self.range_ = None

    def fit(self, x: np.ndarray):
        self.min_   = float(x.min())
        rng = float(x.max()) - self.min_
        self.range_ = rng if rng > 1e-8 else 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min_) / self.range_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.range_ + self.min_

    def save(self, path):
        np.savez(path, min=np.array([self.min_]),
                 range=np.array([self.range_]))

    def load(self, path):
        if not path.endswith(".npz"):
            path = path + ".npz"
        d = np.load(path)
        self.min_   = float(d["min"][0])
        self.range_ = float(d["range"][0])
        return self


fe_scaler = GlobalMinMax()
fp_scaler = GlobalMinMax()


# -- Dataset ---------------------------------------------------

class WindFarmDayDataset(Dataset):
    """
    Each item: (fe_flat_864, fp_flat_864, day_idx)
    fe_flat, fp_flat : torch.FloatTensor (864,)

    The 864-d vectors are row-major flattenings of the (24, 36)
    matrices: index = h * 36 + f.
    """
    def __init__(self, fe_flat, fp_flat, days):
        self.fe   = torch.tensor(fe_flat, dtype=torch.float32)
        self.fp   = torch.tensor(fp_flat, dtype=torch.float32)
        self.days = torch.tensor(days,    dtype=torch.int32)

    def __len__(self):
        return len(self.fe)

    def __getitem__(self, i):
        return self.fe[i], self.fp[i], self.days[i]


# -- Main loader -----------------------------------------------

def load_data(file_path=config.DATA_FILE, batch_size=config.BATCH_SIZE):
    print(f"\n[Data] {'='*55}")
    print(f"[Data]  Wind Farm CGAN  -  (24 x 72) per sample")
    print(f"[Data]  File : {file_path}")

    verify_sample_size(verbose=True)

    df = pd.read_csv(file_path, header=0)
    total_rows, total_cols = df.shape
    print(f"[Data]  CSV  : {total_rows} rows x {total_cols} cols "
          f"(~{total_rows/8760:.1f} yr)")

    assert total_cols == config.RAW_TOTAL_COLS, \
        (f"Expected {config.RAW_TOTAL_COLS} cols, got {total_cols}.")

    # Cap to first TRAIN_YEARS years
    max_days  = int(config.TRAIN_YEARS * 365.25)
    max_hours = max_days * config.N_HOURS
    use_rows  = min(total_rows, max_hours)
    n_days    = use_rows // config.N_HOURS
    n_hours   = n_days * config.N_HOURS
    print(f"[Data]  Using {config.TRAIN_YEARS} yr -> "
          f"{n_days} days ({n_hours/8760:.2f} yr)")

    data   = df.iloc[:n_hours, :].values.astype("float32")
    FE_raw = data[:, :config.N_FARMS]    # (n_hours, 36)
    FP_raw = data[:, config.N_FARMS:]    # (n_hours, 36)

    # Reshape to (n_days, 24, 36)
    FE_3d = FE_raw.reshape(n_days, config.N_HOURS, config.N_FARMS)
    FP_3d = FP_raw.reshape(n_days, config.N_HOURS, config.N_FARMS)

    # Flatten to (n_days, 864) row-major: index = h * 36 + f
    FE_flat = FE_3d.reshape(n_days, config.FE_DIM)
    FP_flat = FP_3d.reshape(n_days, config.FP_DIM)
    day_arr = np.arange(n_days)

    print(f"[Data]  Sample shape    : (24, 72) per day")
    print(f"[Data]  Flat FE dim     : {FE_flat.shape}  (n_days, 864)")
    print(f"[Data]  Flat FP dim     : {FP_flat.shape}  (n_days, 864)")

    # Train / test chronological split (by day)
    n_train_days = int(n_days * config.TRAIN_RATIO)
    n_test_days  = n_days - n_train_days

    FE_tr = FE_flat[:n_train_days]; FE_te = FE_flat[n_train_days:]
    FP_tr = FP_flat[:n_train_days]; FP_te = FP_flat[n_train_days:]
    days_tr = day_arr[:n_train_days]
    days_te = day_arr[n_train_days:]

    print(f"[Data]  Train days : {n_train_days}")
    print(f"[Data]  Test  days : {n_test_days}")

    # Fit scalers on train only
    fe_scaler.fit(FE_tr)
    fp_scaler.fit(FP_tr)
    fe_scaler.save(os.path.join(config.OUTPUT_DIR, "fe_scaler"))
    fp_scaler.save(os.path.join(config.OUTPUT_DIR, "fp_scaler"))

    FE_tr_n = fe_scaler.transform(FE_tr)
    FP_tr_n = fp_scaler.transform(FP_tr)
    FE_te_n = fe_scaler.transform(FE_te)
    FP_te_n = fp_scaler.transform(FP_te)

    print(f"[Data]  FE train norm : mean={FE_tr_n.mean():.4f} "
          f"std={FE_tr_n.std():.4f}")
    print(f"[Data]  FP train norm : mean={FP_tr_n.mean():.4f} "
          f"std={FP_tr_n.std():.4f}")

    tr_ds = WindFarmDayDataset(FE_tr_n, FP_tr_n, days_tr)
    te_ds = WindFarmDayDataset(FE_te_n, FP_te_n, days_te)

    tr_ld = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True,
        persistent_workers=True)
    te_ld = DataLoader(
        te_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=0, pin_memory=True)

    # 3-D test arrays for evaluation: (n_test_days, 24, 36)
    FE_te_3d = FE_te_n.reshape(n_test_days, config.N_HOURS,
                                config.N_FARMS)
    FP_te_3d = FP_te_n.reshape(n_test_days, config.N_HOURS,
                                config.N_FARMS)

    abs_day = n_train_days + config.QQ_COND_DAY
    print(f"\n[Data]  Eval reference: test-day {config.QQ_COND_DAY} "
          f"(abs={abs_day})")
    print(f"[Data]  fe_te_3d shape : {FE_te_3d.shape}  (test_days, 24, 36)")
    print(f"[Data]  fp_te_3d shape : {FP_te_3d.shape}")
    print(f"[Data] {'='*55}\n")

    return (tr_ld, te_ld,
            FE_te_3d, FP_te_3d,
            n_train_days, n_test_days)


def load_test_only(file_path=config.DATA_FILE):
    """
    Load ONLY the test-set 3D arrays + scalers (already saved during training).
    Used by evaluate.py when running standalone after training.
    """
    df = pd.read_csv(file_path, header=0)
    total_rows = df.shape[0]
    max_days  = int(config.TRAIN_YEARS * 365.25)
    max_hours = max_days * config.N_HOURS
    use_rows  = min(total_rows, max_hours)
    n_days    = use_rows // config.N_HOURS
    n_hours   = n_days * config.N_HOURS

    data   = df.iloc[:n_hours, :].values.astype("float32")
    FE_raw = data[:, :config.N_FARMS]
    FP_raw = data[:, config.N_FARMS:]

    FE_3d = FE_raw.reshape(n_days, config.N_HOURS, config.N_FARMS)
    FP_3d = FP_raw.reshape(n_days, config.N_HOURS, config.N_FARMS)

    n_train_days = int(n_days * config.TRAIN_RATIO)
    n_test_days  = n_days - n_train_days

    FE_te = FE_3d[n_train_days:]   # (n_test, 24, 36)
    FP_te = FP_3d[n_train_days:]

    # load saved scalers
    fe_scaler.load(os.path.join(config.OUTPUT_DIR, "fe_scaler"))
    fp_scaler.load(os.path.join(config.OUTPUT_DIR, "fp_scaler"))

    # apply normalisation
    FE_te_n = fe_scaler.transform(FE_te.reshape(n_test_days, -1)) \
                       .reshape(n_test_days, config.N_HOURS, config.N_FARMS)
    FP_te_n = fp_scaler.transform(FP_te.reshape(n_test_days, -1)) \
                       .reshape(n_test_days, config.N_HOURS, config.N_FARMS)

    return FE_te_n, FP_te_n, n_train_days, n_test_days


if __name__ == "__main__":
    res = load_data()
    print(f"\nfe_te_3d : {res[2].shape}")
    print(f"fp_te_3d : {res[3].shape}")
    print(f"n_train_days={res[4]}  n_test_days={res[5]}")
