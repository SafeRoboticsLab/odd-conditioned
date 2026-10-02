"""E068 — TIGHT t_identify for FR-death vs FL-death: is E067's slow identification (linear AUC 0.73 @0.48s) a
weak-discriminator artifact, or a real information limit? Replace the linear discriminator with the BEST encoder
we'd actually use — a 1D-conv over the proprio history window (the RMA-style latent architecture) — trained
properly per time-since-onset. This is Buzi's "latent-space encoding of θ" used to measure the *tightest*
identifiability from actor-observable proprio (base IMU + joints; Go2 has NO foot-force sensor).

Rollout policy = a NEUTRAL stabilizer (go2_stabilize_adv, never trained on degradation) so the obs divergence
reflects the PHYSICS of which-leg-died, not a policy's leg bias. θ=0.2, no pull. Report AUC(τ); t_identify =
first τ with AUC>0.9. If it now hits 0.9 in ~2 steps -> info WAS there (linear was too weak) -> collapse like
shove/slip. If it stays low -> which-leg info genuinely isn't in proprio -> t_identify large -> a real window
ingredient.
"""
import os, sys, io, contextlib, mujoco
import torch as th, torch.nn as nn
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, DT = "cuda:0", 512, 0.02
T0, STEPS, THETA = 30, 100, 0.2
WMAX = 24                                  # max history window fed to the encoder
CK = "results/go2_weak_leg_runs/go2_stabilize_adv/checkpoints/model_49999872_steps.zip"  # NEUTRAL stabilizer


def leg_ids(m, leg):
    return th.tensor([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{leg}_{j}_joint")
                      for j in ("hip", "thigh", "calf")], device=DEV)


def collect(leg, rounds=2):
    outs = []
    for r in range(rounds):
        with contextlib.redirect_stdout(io.StringIO()):
            env = make_tensor("go2_stabilize", N, DEV, adversary=True); model, norm = load_twin(CK, DEV, quiet=True)
        inner = env.mj; m = inner.sim.mj_model; tm = inner.sim.model
        ids = leg_ids(m, leg); nom = tm.actuator_forcerange[:, ids, :].clone()
        env.force_scale = 0.0 * th.ones(N, device=DEV); dstb = th.zeros(N, 3, device=DEV)
        obs = env.reset(); seq = []
        for t in range(STEPS):
            if t >= T0:
                tm.actuator_forcerange[:, ids, :] = nom * THETA
            with th.no_grad():
                a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
            obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
            seq.append(obs[:, :47].clone())
        env.close(); outs.append(th.stack(seq, 1).cpu())          # [N, STEPS, 47]
    return th.cat(outs, 0)                                          # [N*rounds, STEPS, 47]


class ConvEnc(nn.Module):                                          # RMA-style 1D-conv latent -> FR/FL logit
    def __init__(self, feat=47):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):                                          # x:[B,47,W]
        return self.net(x).squeeze(-1)


def auc(sc, y):
    order = np.argsort(sc); r = np.empty(len(sc)); r[order] = np.arange(1, len(sc) + 1)
    P, Q = y.sum(), (1 - y).sum()
    return 0.5 if P == 0 or Q == 0 else float((r[y == 1].sum() - P * (P + 1) / 2) / (P * Q))


def window_at(X, t, w):                                            # last w frames ending at t -> [B,47,w] (pad early)
    lo = max(0, t - w + 1); win = X[:, lo:t + 1].transpose(1, 2)   # [B,47,<=w]
    if win.shape[2] < w:
        win = th.cat([th.zeros(win.shape[0], 47, w - win.shape[2]), win], dim=2)
    return win


if __name__ == "__main__":
    print("collecting FR-death and FL-death rollouts under the NEUTRAL stabilizer...")
    XF, XL = collect("FR"), collect("FL"); M = XF.shape[0]
    print(f"  {M} rollouts/class, {STEPS} steps")
    g = th.Generator().manual_seed(0); idx = th.randperm(M, generator=g)
    tr, te = idx[:int(0.8 * M)], idx[int(0.8 * M):]
    print("\nTIGHT t_identify — 1D-conv encoder AUC vs time-since-onset τ (window<=24 frames):")
    print(f"{'τ(steps)':>8} {'τ(s)':>6} {'AUC':>6}")
    dstar = None
    for tau in [0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 24, 30, 40, 50, 60]:
        t = T0 + tau
        if t >= STEPS: break
        w = min(WMAX, tau + 1)
        Xtr = th.cat([window_at(XF[tr], t, w), window_at(XL[tr], t, w)], 0).to(DEV)
        ytr = th.cat([th.ones(len(tr)), th.zeros(len(tr))]).to(DEV)
        Xte = th.cat([window_at(XF[te], t, w), window_at(XL[te], t, w)], 0).to(DEV)
        yte = np.concatenate([np.ones(len(te)), np.zeros(len(te))])
        enc = ConvEnc().to(DEV); opt = th.optim.Adam(enc.parameters(), 2e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        for ep in range(120):
            enc.train(); perm = th.randperm(Xtr.shape[0], device=DEV)
            for i in range(0, Xtr.shape[0], 512):
                b = perm[i:i + 512]; opt.zero_grad(); lossf(enc(Xtr[b]), ytr[b]).backward(); opt.step()
        enc.eval()
        with th.no_grad(): sc = enc(Xte).cpu().numpy()
        a = auc(sc, yte)
        if a > 0.9 and dstar is None: dstar = tau
        print(f"{tau:>8} {tau*DT:>6.2f} {a:>6.2f}")
    print(f"\n=> TIGHT t_identify (encoder AUC>0.9) = {dstar} steps = {dstar*DT if dstar else float('nan')}s")
    print("Compare E067 LINEAR: AUC 0.73 @0.48s (never hit 0.9). If encoder hits 0.9 fast -> info WAS there ->")
    print("collapse (linear was too weak). If encoder ALSO stays low -> which-leg genuinely not in proprio -> window ingredient.")
