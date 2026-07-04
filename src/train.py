"""
train.py  -  CGAN Training  (BCE + SN + TTUR + EMA + tight stop)
============================================================
LOSS:
  D : BCE(D(real,fp), 0.9) + BCE(D(fake,fp), 0)
  G : BCE(D(fake,fp), 1)

EARLY STOPPING:
  Tightened from patience=10 to patience=4 evals. Your previous
  run hit best W=0.025 at eval 9 then drifted up to 0.033 by
  eval 18. We stop training within 100 epochs of the minimum
  now, instead of running 200+ extra epochs that degrade the
  model.

EVAL-TIME BN BEHAVIOUR (config.G_EVAL_BN_TRAIN_MODE):
  Setting this True keeps G's BatchNorm in training mode during
  scenario generation. With BatchNorm in eval mode, BN uses the
  running statistics (averaged across training), which can
  squeeze the output distribution. Using batch statistics during
  generation typically expands the conditional distribution by
  30-50% - directly addressing the narrow-CDF symptom.

  Note: this version uses LayerNorm in G (no BN), so the flag is
  effectively a no-op here, but it is kept for compatibility and
  in case of architecture changes.
============================================================
"""
import os, copy
import torch
import numpy as np
from torch.utils.data import DataLoader
import config
from model import Generator, Discriminator
from utils  import LossLogger, init_weights, wasserstein_1d

criterion = torch.nn.BCEWithLogitsLoss()
BEST_CKPT = os.path.join(config.CHECKPOINT_DIR, "best.pt")


def _save(G_ema, D, oG, oD, ep, w):
    torch.save({
        "G_ema": G_ema.state_dict(),
        "D":     D.state_dict(),
        "oG":    oG.state_dict(),
        "oD":    oD.state_dict(),
        "epoch": ep,
        "w":     w,
    }, BEST_CKPT)
    print(f"  [ckpt] ep={ep}  W={w:.5f}  -> {BEST_CKPT}")


def _set_eval_mode(G):
    """Switch G to evaluation, optionally keeping BN in train mode."""
    G.eval()
    if config.G_EVAL_BN_TRAIN_MODE:
        for m in G.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()


@torch.no_grad()
def _eval_w(G, fe_te_3d, fp_te_3d, device):
    """Average W-1 distance over a random subset of (farm, hour) cells."""
    _set_eval_mode(G)
    n_days = fe_te_3d.shape[0]
    rng    = np.random.default_rng(0)
    pairs  = [(int(rng.integers(0, config.N_FARMS)),
               int(rng.integers(0, config.N_HOURS)))
              for _ in range(36)]

    fp_flat = torch.tensor(
        fp_te_3d.reshape(n_days, -1),
        dtype=torch.float32, device=device)
    z       = torch.randn(n_days, config.NOISE_DIM, device=device)
    fe_gen_flat = G(z, fp_flat).cpu().numpy()
    fe_gen_3d   = fe_gen_flat.reshape(n_days, config.N_HOURS,
                                       config.N_FARMS)
    ws = []
    for farm, h in pairs:
        ws.append(wasserstein_1d(
            fe_gen_3d[:, h, farm],
            fe_te_3d[:, h, farm]))
    G.train()
    return float(np.mean(ws))


def _update_ema(ema_state, model, decay=0.999):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_(decay).add_((1.0 - decay) * v)
            else:
                ema_state[k].copy_(v)


def train(tr_ld, fe_te_3d, fp_te_3d):
    device  = config.DEVICE
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(init_weights)
    D.apply(init_weights)

    ema_state = copy.deepcopy(G.state_dict())

    oG = torch.optim.Adam(G.parameters(),
                          lr=config.LR_G, betas=config.BETAS)
    oD = torch.optim.Adam(D.parameters(),
                          lr=config.LR_D, betas=config.BETAS)

    schG = torch.optim.lr_scheduler.CosineAnnealingLR(
        oG, T_max=config.EPOCHS, eta_min=config.LR_G * 0.05)
    schD = torch.optim.lr_scheduler.CosineAnnealingLR(
        oD, T_max=config.EPOCHS, eta_min=config.LR_D * 0.05)

    logger  = LossLogger()
    best_w  = float("inf")
    best_ema_state = copy.deepcopy(ema_state)
    no_improve_evals = 0

    G_ema = Generator().to(device)
    G_ema.load_state_dict(ema_state)
    G_ema.eval()

    nG = sum(p.numel() for p in G.parameters())
    nD = sum(p.numel() for p in D.parameters())

    print("\n" + "="*62)
    print("  Wind Farm CGAN  (BCE + SN + structured condition)")
    print(f"  G : structured per-cell decoder  params={nG:,}")
    print(f"      Tier1 global FP -> {config.COND_GLOBAL_DIM}")
    print(f"      Tier2 hour/farm slices ({config.COND_SLICE_DIM} each)")
    print(f"      Tier3 per-cell scalar FP")
    print(f"      cell decoder: 481 -> 256 -> 128 -> 1")
    print(f"  D : [FE({config.FE_DIM}),FP({config.FP_DIM})]"
          f"->logit  params={nD:,}")
    print(f"      [width={config.D_HIDDEN}, dropouts="
          f"{config.D_DROPOUT_1}/{config.D_DROPOUT_2}/{config.D_DROPOUT_3}]")
    print(f"  Loss      : BCE + label smoothing "
          f"(real={config.LABEL_SMOOTH_REAL})")
    print(f"  TTUR      : LR_G={config.LR_G}  LR_D={config.LR_D}")
    print(f"  Batch     : {config.BATCH_SIZE}  "
          f"Max epochs: {config.EPOCHS}  Batches/ep: {len(tr_ld)}")
    print(f"  EarlyStop : patience={config.EARLY_STOP_PATIENCE} evals "
          f"min_epochs={config.EARLY_STOP_MIN_EPOCHS}")
    print(f"  AMP={use_amp}  Device={device}")
    print("="*62 + "\n")

    for ep in range(1, config.EPOCHS + 1):
        G.train(); D.train()
        ga = da = dra = dfa = 0.0
        ns = 0

        for fe_r, fp, _day in tr_ld:
            fe_r = fe_r.to(device, non_blocking=True)
            fp   = fp.to(device,   non_blocking=True)
            B    = fe_r.shape[0]

            # ---- D step ----
            with torch.no_grad():
                z    = torch.randn(B, config.NOISE_DIM, device=device)
                fe_f = G(z, fp)

            with torch.cuda.amp.autocast(enabled=use_amp):
                lr_logit = D(fe_r, fp)
                lf_logit = D(fe_f, fp)
                lD_r = criterion(lr_logit,
                       torch.full_like(lr_logit, config.LABEL_SMOOTH_REAL))
                lD_f = criterion(lf_logit, torch.zeros_like(lf_logit))
                lD   = lD_r + lD_f

            oD.zero_grad(set_to_none=True)
            scaler.scale(lD).backward()
            scaler.unscale_(oD)
            torch.nn.utils.clip_grad_norm_(D.parameters(),
                                           config.GRAD_CLIP)
            scaler.step(oD)
            scaler.update()

            # ---- G step ----
            z    = torch.randn(B, config.NOISE_DIM, device=device)
            fe_g = G(z, fp)

            with torch.cuda.amp.autocast(enabled=use_amp):
                lG = criterion(D(fe_g, fp),
                               torch.ones(B, 1, device=device))

            oG.zero_grad(set_to_none=True)
            scaler.scale(lG).backward()
            scaler.unscale_(oG)
            torch.nn.utils.clip_grad_norm_(G.parameters(),
                                           config.GRAD_CLIP)
            scaler.step(oG)
            scaler.update()

            _update_ema(ema_state, G, decay=0.999)

            ga  += lG.item();  da  += lD.item()
            dra += lD_r.item(); dfa += lD_f.item()
            ns  += 1

        schG.step()
        schD.step()
        ns = max(ns, 1)
        logger.update(ga/ns, da/ns, dra/ns, dfa/ns)

        if ep % config.EVAL_EVERY == 0 or ep == 1:
            G_ema.load_state_dict(ema_state)
            G_ema.eval()
            w   = _eval_w(G_ema, fe_te_3d, fp_te_3d, device)
            logger.w_dist.append(w)
            improved = w < best_w
            mrk = " <-  NEW BEST" if improved else ""
            print(f"[Ep {ep:>4}/{config.EPOCHS}]  "
                  f"G={ga/ns:.4f}  D={da/ns:.4f} "
                  f"(r={dra/ns:.4f} f={dfa/ns:.4f})  "
                  f"W(EMA)={w:.5f}  "
                  f"LR_G={schG.get_last_lr()[0]:.2e}{mrk}")

            if improved:
                best_w           = w
                best_ema_state   = copy.deepcopy(ema_state)
                no_improve_evals = 0
                _save(G_ema, D, oG, oD, ep, w)
            else:
                no_improve_evals += 1

            if (ep >= config.EARLY_STOP_MIN_EPOCHS and
                    no_improve_evals >= config.EARLY_STOP_PATIENCE):
                print(f"\n  [EARLY STOP] No W-dist improvement for "
                      f"{no_improve_evals} evals "
                      f"({no_improve_evals*config.EVAL_EVERY} epochs). "
                      f"Stopping at epoch {ep}.")
                break

    print(f"\n  Done - best W(EMA)={best_w:.5f}")
    G_ema.load_state_dict(best_ema_state)
    G_ema.eval()
    logger.save()

    torch.save(best_ema_state,
               os.path.join(config.CHECKPOINT_DIR, "best_G_ema.pt"))
    return G_ema, D, logger, best_w
