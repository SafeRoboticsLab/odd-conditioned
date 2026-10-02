"""E064 RMA-style baseline — the honest "is opaque adaptation as safe as knowing θ?" test (professor threat b).

Teacher = the dynamic-trained CONDITIONED policy π(x, θ) (48-dim obs). Student φ = a proper RMA-style estimator:
a 1D-conv over ~50 steps of proprio+action history -> θ̂ (supervised regression to the TRUE θ, decoupled from the
policy — unlike our end-to-end history arm which overfit, E054). At eval we run π but feed θ̂=φ(history) in place of
true θ. Reports:
  * survival/slam on the leg-death ramp vs CONDITIONED-oracle and vs BLIND
  * identification latency δ*: steps after the leg dies until θ̂ tracks the drop  ==  the commitment window, quantified
If φ matches the oracle, the empirical safety gap is 'just adaptation' and the contribution rests on certification.
If φ LAGS (δ*>0 costs safety during the window), a fast/sound belief is what closes it.

Usage: python experiments/E064_rma.py --seed 0
"""
import os, sys, io, contextlib, argparse
import torch as th, torch.nn as nn
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, DT = "cuda:0", 0.02
K = 50                                   # history length (RMA ~50 steps)
PROPRIO = 47                             # conditioned actor obs = 47 proprio + 1 θ; proprio = obs[:, :47]
FEAT = PROPRIO + 12                      # proprio + action
CKROOT = "results/go2_weak_leg_runs"
TASK = "go2_weak_leg_conditioned_soft"   # 48-dim obs (proprio + θ); we drive θ manually
PULL = th.tensor([0., 1., 0.])


class Student(nn.Module):
    """1D-conv over the K-step [proprio+action] window -> θ̂ ∈ (0,1]. RMA-style adaptation module."""
    def __init__(self, feat=FEAT, k=K):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(feat, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                # x: [B, feat, K]
        return 0.2 + 0.8 * th.sigmoid(self.head(self.enc(x))).squeeze(-1)   # -> [0.2, 1.0]


def rand_theta_schedule(nenv, steps):
    """Per-env random dynamic θ(t): θ_start->θ_end step at t_change (matches the training LegDeathCallback dist)."""
    ts = th.rand(nenv, device=DEV) * 0.8 + 0.2
    te = th.rand(nenv, device=DEV) * 0.8 + 0.2
    tc = th.randint(20, steps - 20, (nenv,), device=DEV)
    sched = th.stack([th.where(th.arange(steps, device=DEV)[None] < tc[:, None], ts[:, None], te[:, None])])
    return sched[0]                      # [nenv, steps]


def collect(model, norm, nenv=128, steps=180, rounds=5, sub=2):
    """Teacher rollouts under random dynamic θ. Returns windows [M, FEAT, K] and targets θ [M] ON CPU (the GPU is
    busy with training runs; keep only the live sim state on GPU, offload the growing dataset)."""
    env = make_tensor(TASK, nenv, DEV, adversary=True)
    inner = env.mj; _ensure_fr_cache(inner); ids, nom = inner._fr_act_ids, inner._fr_nominal_forcerange
    env.force_scale = 0.3 * th.ones(nenv, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, spec(TASK).dstb_dim).contiguous()
    Xs, Ys = [], []
    for r in range(rounds):
        sched = rand_theta_schedule(nenv, steps)
        obs = env.reset(); hist = th.zeros(nenv, FEAT, K, device=DEV); last_a = th.zeros(nenv, 12, device=DEV)
        for t in range(steps):
            theta = sched[:, t]
            inner.sim.model.actuator_forcerange[:, ids, :] = nom * theta[:, None, None]
            inner._fr_torque_frac[:] = theta                         # teacher sees TRUE θ
            frame = th.cat([obs[:, :PROPRIO], last_a], dim=1)        # [nenv, FEAT]
            hist = th.roll(hist, -1, dims=2); hist[:, :, -1] = frame
            with th.no_grad():
                a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
            if t >= K and t % sub == 0:                              # full windows, subsampled, OFFLOAD to CPU
                Xs.append(hist.detach().cpu()); Ys.append(theta.detach().cpu())
            last_a = a[:, :12]
            obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
    env.close()
    return th.cat(Xs, 0), th.cat(Ys, 0)     # on CPU


def train_student(X, Y, epochs=8, bs=2048):
    phi = Student().to(DEV); opt = th.optim.Adam(phi.parameters(), 1e-3)
    n = X.shape[0]
    for ep in range(epochs):
        perm = th.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = X[idx].to(DEV, non_blocking=True); yb = Y[idx].to(DEV, non_blocking=True)
            pred = phi(xb); loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        print(f"  epoch {ep} mse={tot/n:.4f}")
    return phi


def eval_ramp(model, norm, phi, fr, mode):
    """mode: 'oracle' (true θ), 'rma' (θ̂ from φ), 'blind' (θ frozen at 1.0). Ramp θ 1.0->0.2 @ t=100."""
    nenv, steps, tsw = 128, 250, 100
    env = make_tensor(TASK, nenv, DEV, adversary=True)
    inner = env.mj; _ensure_fr_cache(inner); ids, nom = inner._fr_act_ids, inner._fr_nominal_forcerange
    env.force_scale = fr * th.ones(nenv, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, spec(TASK).dstb_dim).contiguous()
    obs = env.reset(); alive = th.ones(nenv, dtype=th.bool, device=DEV); slam = th.zeros(nenv, dtype=th.bool, device=DEV)
    hist = th.zeros(nenv, FEAT, K, device=DEV); last_a = th.zeros(nenv, 12, device=DEV)
    thetahat_trace = []
    for t in range(steps):
        theta = 1.0 if t < tsw else 0.2
        inner.sim.model.actuator_forcerange[:, ids, :] = nom * float(theta)     # physics = TRUE θ
        frame = th.cat([obs[:, :PROPRIO], last_a], dim=1)
        hist = th.roll(hist, -1, dims=2); hist[:, :, -1] = frame
        if mode == "oracle":
            inner._fr_torque_frac[:] = float(theta)
        elif mode == "blind":
            inner._fr_torque_frac[:] = 1.0
        else:                                                                   # rma
            with th.no_grad(): th_hat = phi(hist)
            inner._fr_torque_frac[:] = th_hat
            thetahat_trace.append(float(th_hat.mean()))
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        last_a = a[:, :12]
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        ic = inner.termination_manager._term_dones.get("illegal_contact")
        nf = dones & ~touts & alive
        if ic is not None: slam |= nf & ic
        alive &= ~(dones & ~touts)
    env.close()
    # δ*: steps after the switch until mean θ̂ crosses below 0.6 (detects the drop to 0.2)
    dstar = float('nan')
    if mode == "rma":
        post = thetahat_trace[tsw:]
        for j, v in enumerate(post):
            if v < 0.6: dstar = j * DT; break
    return float(alive.float().mean()), float(slam.float().mean()), dstar


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    ck = f"{CKROOT}/E064_s{a.seed}/go2_weak_leg_conditioned_soft_dyn_adv/checkpoints/model_49999872_steps.zip"
    assert os.path.exists(ck), ck
    with contextlib.redirect_stdout(io.StringIO()):
        _env = make_tensor(TASK, 4, DEV, adversary=True); model, norm = load_twin(ck, DEV, quiet=True); _env.close()
    print(f"[seed {a.seed}] collecting teacher rollouts for the RMA student…")
    X, Y = collect(model, norm)
    print(f"  data: {X.shape[0]} windows")
    phi = train_student(X, Y)
    print("\nramp eval (survival / slam / δ*):")
    out = {}
    for fr in (0.3, 0.5):
        for mode in ("oracle", "rma", "blind"):
            s, sl, ds = eval_ramp(model, norm, phi, fr, mode)
            out[f"{int(fr*50)}N|{mode}"] = [s, sl, ds]
            print(f"  {int(fr*50)}N {mode:8s} surv={s:.2f} slam={sl:.2f}" + (f" δ*={ds:.2f}s" if mode == "rma" else ""))
    od = os.path.expanduser("~/artifacts/odd-conditioned/E064-fair-fight")
    os.makedirs(od, exist_ok=True)
    import json; json.dump(out, open(f"{od}/rma_seed{a.seed}.json", "w"), indent=2)
    print(f"saved -> {od}/rma_seed{a.seed}.json")
