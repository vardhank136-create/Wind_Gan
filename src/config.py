"""
config.py  -  Wind Farm CGAN  (BCE+SN+TTUR + structured condition)
============================================================
ROOT-CAUSE FIX FOR THE NARROW CDFs:

Previous versions passed FP as a flat 864-dim vector to G.
For producing ONE FE value at (hour h, farm f), the network
had to extract the relevant signal from 864 numbers - mostly
irrelevant cross-farm/cross-hour FP values. The network solved
this the easy way: collapse to a near-deterministic mean given
the condition. Result: orange CDFs were narrow S-curves.

FIX: Structured condition pathway. For each output cell
(hour h, farm f), the model now has DIRECT access to:
  * FP_{h,f}        - that cell's own FP   (1 number,
                       primary signal)
  * FP_{h,:}        - all-farms FP this hr (36 numbers,
                       spatial context)
  * FP_{:,f}        - this farm's FP day  (24 numbers,
                       temporal context)

This is still (24, 72) per sample, still CGAN, still MLP-MLP,
still pure feed-forward. Only the conditioning pathway changes.

ANOTHER FIX (training):
Previous run reached best W=0.025 at eval ~9 then drifted up to
0.033 by eval 18. Patience 10 was too long. Reduced to 4 evals.
The best checkpoint is always restored anyway - this just saves
wasted compute and prevents over-training.
============================================================
"""
import torch, os

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_FILE      = os.path.join(BASE_DIR, "wind_gan.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_cgan")
LOG_DIR        = os.path.join(BASE_DIR, "logs_cgan")
OUTPUT_DIR     = os.path.join(BASE_DIR, "outputs_cgan")
for _d in [CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)

# -- Data ------------------------------------------------------
N_FARMS        = 36
N_HOURS        = 24
RAW_TOTAL_COLS = 72
TRAIN_YEARS    = 6
TRAIN_RATIO    = 0.80

FE_DIM = N_HOURS * N_FARMS   # 864
FP_DIM = N_HOURS * N_FARMS   # 864

# -- Model -----------------------------------------------------
NOISE_DIM   = 128
G_HIDDEN    = 512
D_HIDDEN    = 512

# Stronger D dropout for the small dataset (~1,750 days)
D_DROPOUT_1 = 0.4
D_DROPOUT_2 = 0.4
D_DROPOUT_3 = 0.3

# Structured condition embedding sizes
COND_GLOBAL_DIM = 256        # global FP summary fed into G trunk
COND_SLICE_DIM  = 64         # per-cell slice embedding

# -- Training (BCE + SN + TTUR + EMA) --------------------------
BATCH_SIZE        = 64
EPOCHS            = 800       # upper bound; early stopping decides
LR_G              = 5e-5
LR_D              = 2e-4      # TTUR: 4x LR_G
BETAS             = (0.5, 0.999)
GRAD_CLIP         = 5.0
LABEL_SMOOTH_REAL = 0.9
RANDOM_SEED       = 42
EVAL_EVERY        = 25

# -- Early stopping (TIGHT - prevent over-training) ------------
EARLY_STOP_PATIENCE   = 4    # was 10 -> 4 (was over-training)
EARLY_STOP_MIN_EPOCHS = 200

# -- Evaluation ------------------------------------------------
N_GEN_SAMPLES  = 1000
N_EVAL_SETS    = 20
QQ_COND_DAY    = 0
QQ_FARM_IDX    = 0
QUANTILES      = [0.05, 0.25, 0.50, 0.75, 0.95]

# -- Fan chart -------------------------------------------------
FAN_CHART_START_DAY = 0
FAN_CHART_N_DAYS    = 30
FAN_CHART_FARM      = 0

# Use BN in train-mode at eval-time (BatchNorm batch-stats trick).
# Setting this True makes G.eval() retain BN train-mode behaviour
# during scenario generation. This often EXPANDS generated
# distributions by ~30-50% (fixes narrow-CDF symptom).
G_EVAL_BN_TRAIN_MODE = True

TRAIN_MODE = "CGAN-Wind-BCE"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
