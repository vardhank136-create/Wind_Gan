"""
model.py  -  CGAN  (BCE + SN + structured condition)  (24x72)
============================================================
KEY CHANGE: STRUCTURED CONDITION PATHWAY

Previous: G(z(128), fp_864) -> fe_864
  Problem: G has to extract relevant FP for cell (h, f) from
  864 mostly-irrelevant numbers. It collapses to a deterministic
  mean -> narrow CDFs.

Now: structured 3-tier condition pathway. For each output cell
(h, f), the network has direct access to:

  Tier 1 - global summary:
    fp_summary = MLP(fp_864 -> 256)
    Captures the overall FP signature of the day.

  Tier 2 - per-cell slices (computed once, used for all cells):
    h_slice[h] : embedding of FP[h, :] (36 numbers)
                  -> 64-d representation of "this hour across farms"
    f_slice[f] : embedding of FP[:, f] (24 numbers)
                  -> 64-d representation of "this farm across hours"

  Tier 3 - per-cell direct value:
    fp_local[h, f] : the actual scalar FP value at that cell.

  Decoder per cell:
    [z_cell, fp_summary, h_slice[h], f_slice[f], fp_local[h, f]]
        -> per-cell MLP -> scalar FE value

ARCHITECTURE BLOCK BY BLOCK:

GENERATOR:
  Inputs: z (B, 128), fp (B, 864) reshaped to (B, 24, 36)
  fp_summary = MLP(fp_flat) -> (B, 256)   # global

  hour_proj : Linear(36 -> 64) applied to fp_per_hour (B, 24, 36)
              -> hour_slices (B, 24, 64)
  farm_proj : Linear(24 -> 64) applied to fp_per_farm (B, 36, 24)
              -> farm_slices (B, 36, 64)

  For every (h, f) cell, build cell condition:
    cell_cond = concat(z_cell, fp_summary, hour_slice[h],
                        farm_slice[f], fp_local[h,f])
    dimension = z_cell_dim + 256 + 64 + 64 + 1

  z_cell : a (B, 24, 36, k) tensor of cell-specific noise
            obtained by reshaping the global noise z and
            broadcasting (cheaper than separate noise per cell
            but each cell still has its own noise dimension).

  Cell decoder is a small MLP (3 layers, 128 hidden) shared
  across all 864 cells (parameter sharing across (h, f),
  per-farm bias added at the end).

DISCRIMINATOR (BCE, spectral norm):
  Same as before - flat input is fine for D. Strong dropout to
  prevent memorisation of the ~1,750-day training set.

NOISE INJECTION:
  Small Gaussian noise added at the bottleneck inside G prevents
  variance collapse even further (helps with narrow CDFs).
============================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class Generator(nn.Module):
    """
    Per-cell decoder G with structured FP condition.

    Forward:
      z  : (B, NOISE_DIM)
      fp : (B, FP_DIM=864)  - flattened (24, 36) FP
    Output:
      fe : (B, FE_DIM=864)  - flattened (24, 36) generated FE
    """
    def __init__(self):
        super().__init__()

        # Per-cell noise dim. Instead of producing a giant
        # 24*36*32-dim cell-noise vector from z (which would be
        # 14M parameters), we produce hour-noise and farm-noise
        # independently and combine them. This gives every cell
        # its own noise channel with O(N_HOURS + N_FARMS) cost
        # rather than O(N_HOURS * N_FARMS).
        self.cell_noise_dim = 32

        # Hour-noise: z -> (B, 24, 32)
        self.hour_noise_mlp = nn.Sequential(
            nn.Linear(config.NOISE_DIM, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, config.N_HOURS * self.cell_noise_dim),
        )
        # Farm-noise: z -> (B, 36, 32)
        self.farm_noise_mlp = nn.Sequential(
            nn.Linear(config.NOISE_DIM, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, config.N_FARMS * self.cell_noise_dim),
        )

        # Global FP summary
        self.fp_global = nn.Sequential(
            nn.Linear(config.FP_DIM, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, config.COND_GLOBAL_DIM),
        )

        # Hour slice projector: 36 -> 64
        self.hour_proj = nn.Sequential(
            nn.Linear(config.N_FARMS, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, config.COND_SLICE_DIM),
        )

        # Farm slice projector: 24 -> 64
        self.farm_proj = nn.Sequential(
            nn.Linear(config.N_HOURS, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, config.COND_SLICE_DIM),
        )

        # Global noise compressor (shared across all cells, gives
        # each cell access to a 64-d summary of z).
        self.global_noise_proj = nn.Linear(config.NOISE_DIM, 64)

        # Per-cell decoder. Input dim:
        #   cell_noise (32) + global_noise (64) + global_fp (256)
        #   + hour_slice (64) + farm_slice (64) + fp_local (1)
        # = 32 + 64 + 256 + 64 + 64 + 1 = 481
        cell_in = (self.cell_noise_dim + 64 +
                   config.COND_GLOBAL_DIM +
                   config.COND_SLICE_DIM * 2 + 1)
        self.cell_decoder = nn.Sequential(
            nn.Linear(cell_in, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

        # Per-farm output bias (length 36) - lets each farm pick
        # up its own offset.
        self.farm_bias = nn.Parameter(torch.zeros(config.N_FARMS))

    def forward(self, z, fp):
        B = z.shape[0]
        # Reshape FP -> (B, 24, 36)
        fp_3d = fp.view(B, config.N_HOURS, config.N_FARMS)

        # ---- Tier 1: global FP summary (B, 256) ----
        g = self.fp_global(fp)
        g = g.unsqueeze(1).unsqueeze(1).expand(
            B, config.N_HOURS, config.N_FARMS, -1)   # (B,24,36,256)

        # ---- Tier 2: hour slices and farm slices ----
        # hour slice: per hour, look at all 36 farms
        h_in = fp_3d                                      # (B, 24, 36)
        h_slices = self.hour_proj(h_in)                   # (B, 24, 64)
        h_slices = h_slices.unsqueeze(2).expand(
            B, config.N_HOURS, config.N_FARMS, -1)        # (B,24,36,64)

        # farm slice: per farm, look at all 24 hours
        f_in = fp_3d.transpose(1, 2)                       # (B, 36, 24)
        f_slices = self.farm_proj(f_in)                    # (B, 36, 64)
        f_slices = f_slices.unsqueeze(1).expand(
            B, config.N_HOURS, config.N_FARMS, -1)         # (B,24,36,64)

        # ---- Tier 3: per-cell scalar FP ----
        fp_local = fp_3d.unsqueeze(-1)                     # (B,24,36,1)

        # ---- Noise pathways ----
        # hour-noise (B, 24, 32) and farm-noise (B, 36, 32),
        # combined multiplicatively to give per-cell noise.
        hn = self.hour_noise_mlp(z).view(B, config.N_HOURS,
                                          self.cell_noise_dim)
        fn = self.farm_noise_mlp(z).view(B, config.N_FARMS,
                                          self.cell_noise_dim)
        # broadcast: (B, 24, 1, 32) * (B, 1, 36, 32) -> (B, 24, 36, 32)
        cn = hn.unsqueeze(2) * fn.unsqueeze(1)

        # global noise summary (B, 64) -> broadcast
        gn = self.global_noise_proj(z)                     # (B, 64)
        gn = gn.unsqueeze(1).unsqueeze(1).expand(
            B, config.N_HOURS, config.N_FARMS, -1)         # (B,24,36,64)

        # ---- Concatenate per-cell condition ----
        cond = torch.cat([cn, gn, g, h_slices,
                          f_slices, fp_local], dim=-1)
        # (B, 24, 36, 32+64+256+64+64+1 = 481)

        # ---- Per-cell decoder (shared MLP across cells) ----
        # Reshape to (B*24*36, 481) for batched MLP
        cond_flat = cond.view(-1, cond.shape[-1])
        out = self.cell_decoder(cond_flat)                 # (B*24*36, 1)
        out = out.view(B, config.N_HOURS, config.N_FARMS)  # (B, 24, 36)

        # Add per-farm bias
        out = out + self.farm_bias.view(1, 1, config.N_FARMS)

        out = out.view(B, config.FE_DIM)                   # (B, 864)
        return torch.sigmoid(out)


class Discriminator(nn.Module):
    """Spectral-norm BCE discriminator (same as v4)."""
    def __init__(self):
        super().__init__()
        SN     = nn.utils.spectral_norm
        in_dim = config.FE_DIM + config.FP_DIM       # 1728
        H      = config.D_HIDDEN                       # 512

        self.net = nn.Sequential(
            SN(nn.Linear(in_dim, H)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.D_DROPOUT_1),

            SN(nn.Linear(H, H)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.D_DROPOUT_2),

            SN(nn.Linear(H, H // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.D_DROPOUT_3),

            SN(nn.Linear(H // 2, 128)),
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Linear(128, 1)),     # raw logit
        )

    def forward(self, fe, fp):
        x = torch.cat([fe, fp], dim=1)
        return self.net(x)


if __name__ == "__main__":
    B  = 8
    z  = torch.randn(B, config.NOISE_DIM)
    fp = torch.rand(B, config.FP_DIM)

    G = Generator(); D = Discriminator()
    fe_g  = G(z, fp)
    logit = D(fe_g, fp)

    nG = sum(p.numel() for p in G.parameters())
    nD = sum(p.numel() for p in D.parameters())
    print(f"G output (FE)   : {fe_g.shape}  range "
          f"[{fe_g.min():.3f},{fe_g.max():.3f}]")
    print(f"  std across batch dimension (per-cell variability):")
    print(f"    {fe_g.std(dim=0).mean():.4f}  (higher = more diverse)")
    print(f"D logit         : {logit.shape}")
    print(f"D logit range   : [{logit.min():.3f}, {logit.max():.3f}]")
    print(f"G params : {nG:,}   D params : {nD:,}")
    print(f"\nCondition pathway:")
    print(f"  Tier 1 global FP  : {config.FP_DIM} -> "
          f"{config.COND_GLOBAL_DIM}")
    print(f"  Tier 2 hour slice : {config.N_FARMS} -> "
          f"{config.COND_SLICE_DIM}  (per hour, x{config.N_HOURS})")
    print(f"  Tier 2 farm slice : {config.N_HOURS} -> "
          f"{config.COND_SLICE_DIM}  (per farm, x{config.N_FARMS})")
    print(f"  Tier 3 local FP   : 1 (direct scalar)")
    print(f"  Cell decoder input: 481-d  -> 1 scalar per cell")
    print(f"  Total cells       : {config.N_HOURS}*"
          f"{config.N_FARMS} = {config.N_HOURS*config.N_FARMS}")

    # Test: same FP, two different z -> outputs should differ
    z1 = torch.randn(1, config.NOISE_DIM)
    z2 = torch.randn(1, config.NOISE_DIM)
    fp_fix = torch.rand(1, config.FP_DIM)
    o1 = G(z1, fp_fix); o2 = G(z2, fp_fix)
    diff = (o1 - o2).abs().mean().item()
    print(f"\n  |G(z1,fp) - G(z2,fp)| mean : {diff:.4f}  "
          f"(should be > 0; tests stochasticity)")
