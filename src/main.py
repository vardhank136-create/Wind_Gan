"""
main.py  -  Wind Farm Forecast Error CGAN  (v5: structured cond)
============================================================
Run:    python main.py            -> trains AND evaluates
Run:    python evaluate.py        -> evaluates only (after train)

------------------------------------------------------------
WHAT WENT WRONG IN v4 AND HOW v5 FIXES IT:

v4 plot showed:
  * G/D losses sat right at equilibrium (G=0.80, D=1.38)
  * D(real) and D(fake) cleanly separated (0.78, 0.60)
  * W-distance hit minimum 0.025 at eval ~9 then drifted UP
    to 0.033 by eval 18 (drift past optimum)
  * BUT CDFs: orange curves were narrow steep S-shapes vs
    broad blue real curves (variance underestimation)

ROOT CAUSE: The condition vector was a flat 864-d FP. To
generate one FE value, the network had to extract relevant
signal from 864 mostly-irrelevant numbers. The path of least
resistance: collapse to a deterministic mean given the
condition. -> narrow CDFs.

v5 FIX 1 - STRUCTURED CONDITION (the main change):
For each output cell (hour h, farm f), the network now has
DIRECT access to the FP slices that actually matter:
  * fp_local        - that exact cell's FP value (1 number)
  * hour slice (h)  - all 36 farms at that hour (-> 64-d emb)
  * farm slice (f)  - that farm across 24 hours (-> 64-d emb)
  * global summary  - whole-day FP (-> 256-d emb)
  * per-cell noise  - 32-d noise unique to (h, f)
  * global noise    - 64-d shared noise

A small per-cell decoder (481 -> 256 -> 128 -> 1) produces each
FE value. Because the right FP signal is now ON the table for
each cell, the network has no incentive to collapse.

v5 FIX 2 - TIGHT EARLY STOPPING:
Patience reduced from 10 evals to 4 evals. The previous run
already FOUND the optimum at eval 9 - we just kept training
past it for 200+ epochs, drifting up. Best checkpoint is still
auto-restored, this just stops the wasted compute.

v5 FIX 3 - BN BATCH-STATS AT GENERATION TIME (safety net):
config.G_EVAL_BN_TRAIN_MODE = True keeps BatchNorm in train
mode during sampling. This often expands generated distributions
by 30-50%. v5 G uses LayerNorm, so this is mostly a no-op here
but kept for robustness.

WHAT IS PRESERVED:
  * BCE + spectral norm + label smoothing + TTUR + EMA  (proven)
  * (24, 72) per-day sample structure                   (your req)
  * 20 sets x 1000 scenarios evaluation
  * 20 CSV outputs
  * Spatial variogram across 36 farms
  * All metrics, all plots, standalone evaluate.py

WHAT TO EXPECT:
  * Loss panel 1: G smoothed -> ~0.69, D smoothed -> ~1.33,
    D(real) and D(fake) separated.
  * Loss panel 2: W-distance drops, plateaus around 0.022-0.030,
    early-stop triggers within ~100 epochs of the minimum.
  * CDF plots: orange curves with similar SPREAD to blue real
    curves - not narrow vertical S-shapes anymore. There may
    still be some location bias hour-to-hour, but the WIDTH
    should match closely.
============================================================
"""
import config
from data_loading import load_data, fe_scaler, fp_scaler
from train        import train
from evaluate     import evaluate
from utils        import set_seed, verify_sample_size


def main():
    print(f"\n{'='*62}")
    print(f"  Wind Farm FE-CGAN  (v5: structured condition)")
    print(f"  Device      : {config.DEVICE}")
    print(f"  Loss        : BCE + spectral-norm + label smoothing"
          f"({config.LABEL_SMOOTH_REAL})")
    print(f"  Sample      : (24, 72) per day")
    print(f"  Condition pathway:")
    print(f"    Tier 1 (global FP)   : 864 -> "
          f"{config.COND_GLOBAL_DIM}")
    print(f"    Tier 2 (hour slice)  : 36 -> "
          f"{config.COND_SLICE_DIM}  per hour")
    print(f"    Tier 2 (farm slice)  : 24 -> "
          f"{config.COND_SLICE_DIM}  per farm")
    print(f"    Tier 3 (per-cell FP) : 1 scalar per cell")
    print(f"  G : per-cell decoder, shared MLP across "
          f"{config.N_HOURS}*{config.N_FARMS} cells")
    print(f"  D : flat-input critic, hidden={config.D_HIDDEN}")
    print(f"  D dropouts  : {config.D_DROPOUT_1}/{config.D_DROPOUT_2}"
          f"/{config.D_DROPOUT_3}")
    print(f"  TTUR        : LR_D={config.LR_D:.1e}  "
          f"LR_G={config.LR_G:.1e}")
    print(f"  Adam        : betas={config.BETAS}")
    print(f"  Max epochs  : {config.EPOCHS}  (early stop: "
          f"patience={config.EARLY_STOP_PATIENCE} evals)")
    print(f"  Min epochs  : {config.EARLY_STOP_MIN_EPOCHS}")
    print(f"  Eval        : {config.N_EVAL_SETS} sets x "
          f"{config.N_GEN_SAMPLES} scenarios")
    print(f"  BN at eval  : {'train-mode (expand dist)' if config.G_EVAL_BN_TRAIN_MODE else 'eval-mode (running stats)'}")
    print(f"{'='*62}\n")

    facts = verify_sample_size(verbose=True)
    assert facts["is_24x72_per_sample"], (
        "Sample shape error - should be (24, 72) per day."
    )

    set_seed()

    (tr_ld, te_ld,
     fe_te_3d, fp_te_3d,
     n_train_days, n_test_days) = load_data()

    G, D, logger, best_w = train(tr_ld, fe_te_3d, fp_te_3d)

    metrics = evaluate(G, D, logger,
                       fe_te_3d, fp_te_3d,
                       fe_scaler, fp_scaler,
                       n_train_days, config.TRAIN_MODE)

    print(f"\n{'='*62}")
    print(f"  FINAL RESULTS  (v5)")
    print(f"  Best W-dist (EMA)        : {best_w:.5f}")
    print(f"  Mean QQ R^2              : {metrics['mean_qq_r2']:.3f}")
    print(f"  Mean coverage            : {metrics['mean_farm_cov']:.1f}%")
    print(f"  Mean CRPS (avg 20 sets)  : {metrics['mean_crps']:.5f}")
    print(f"  CRPS std across sets     : {metrics['crps_set_std']:.5f}")
    print(f"  Variogram mean|diff|     : "
          f"{metrics['variogram_mean_abs_diff']:.5f}")
    print(f"  Corr mean|diff|          : "
          f"{metrics['corr_mean_abs_diff']:.5f}")
    print(f"\n  Outputs                  : {config.OUTPUT_DIR}/")
    print(f"  Re-run evaluation only   : python evaluate.py")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
