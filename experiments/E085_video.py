"""E085 — videos of the BATCH-2 evaluation (E084). 3 panels: V1 (batch-1 switching) | V2 (certified
automaton) | ONE-WAY, per wave (square/sine) x condition (benign/gusty). Guard logic mirrors
experiments/E084_automaton.py exactly (V2: 4-state automaton w/ certified get-up + abort; V1: prone-V guard).
Trunk tint = host-model majority (per-env colors don't render): STAND gray->red with W, DESCENDING orange,
REST blue, GETTINGUP green. Tag bar shows live survivor count (first tip/slam = death, E081 censoring; the
sim still auto-resets, so a fallen robot may visibly respawn — it no longer counts).
Graph: W(t) + gusts, no-respawn survival per arm, V2 automaton-state ribbon.
"""
import os, sys, io, contextlib, math, mujoco
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
import numpy as np, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from E084_automaton import (CK, W_of, gust_scale, value_of, in_rest_target, in_stance_target,
                            EPS_DN, K_DN, EPS_UP, K_UP, EPS_ABORT, K_ABORT, REFRACT, ALPHA, DT, STEPS,
                            WARMUP, UP_W_GATE)

DEV, NENV = "cuda:0", 16
ARMS = ["V1", "V2", "ONE-WAY"]
COL = {"V1": "#8e44ad", "V2": "#1a5276", "ONE-WAY": "#e67e22", "REST-ONLY": "#1e8449"}
GRAY, RED = np.array([0.5, 0.5, 0.5, 1.0]), np.array([0.85, 0.1, 0.1, 1.0])
ST_COL = {1: np.array([0.90, 0.49, 0.13, 1.0]),   # DESCENDING orange
          2: np.array([0.15, 0.35, 0.90, 1.0]),   # REST blue
          3: np.array([0.12, 0.52, 0.29, 1.0])}   # GETTINGUP green
ST_HEX = {0: "#b03a2e", 1: "#e67e22", 2: "#1a5276", 3: "#1e8449"}


def rollout(sched, cond, arm):
    N = NENV
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", N, DEV, adversary=True, render_mode="rgb_array")
        try: env.mj.cfg.viewer.max_extra_envs = N - 1
        except Exception: pass
        tw = {k: load_twin(v, DEV, quiet=True) for k, v in CK.items() if os.path.exists(v)}
    inner = env.mj
    inner.cfg.episode_length_s = 10_000.0   # kill the 20s timeout: no free mid-eval resets (Buzi's catch)
    mm = inner.sim.mj_model
    base_id = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
    trunk = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == base_id]
    dstb = th.zeros(N, spec("go2_weight_rest_hi_at_0").dstb_dim, device=DEV); dstb[:, 1] = 1.0
    obs = env.reset()
    st = th.zeros(N, dtype=th.long, device=DEV)
    vs_bar = th.zeros(N, device=DEV); vu_bar = th.zeros(N, device=DEV)
    below = th.zeros(N, device=DEV); above = th.zeros(N, device=DEV); abort_c = th.zeros(N, device=DEV)
    settle = th.zeros(N, device=DEV); standok = th.zeros(N, device=DEV)
    refr = th.zeros(N, device=DEV)
    alive = th.ones(N, dtype=th.bool, device=DEV)
    v1_in_rest = th.zeros(N, dtype=th.bool, device=DEV)
    frames, S, stmaj = [], [], []
    for t in range(STEPS):
        W = W_of(sched, t)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = 0.25
        env.force_scale = gust_scale(cond, t) * th.ones(N, device=DEV)
        m_s, n_s = tw["stand"]; m_r, n_r = tw["rest"]
        Vs = value_of(env, m_s, n_s)
        vs_bar = Vs.clone() if t == 0 else (1 - ALPHA) * vs_bar + ALPHA * Vs
        if arm == "V2":
            m_u, n_u = tw["getup"]
            Vu = value_of(env, m_u, n_u)
            vu_bar = Vu.clone() if t == 0 else (1 - ALPHA) * vu_bar + ALPHA * Vu
            below = th.where(vs_bar < EPS_DN, below + 1, th.zeros_like(below))
            above = th.where(vu_bar > EPS_UP, above + 1, th.zeros_like(above))
            abort_c = th.where(vu_bar < EPS_ABORT, abort_c + 1, th.zeros_like(abort_c))
            settle = th.where(in_rest_target(inner), settle + 1, th.zeros_like(settle))
            standok = th.where(in_stance_target(inner), standok + 1, th.zeros_like(standok))
            can = (refr <= 0) & (t >= WARMUP)
            go_desc = (st == 0) & (below >= K_DN) & can
            go_rest = (st == 1) & (settle >= 5)
            go_up = (st == 2) & (above >= K_UP) & can
            if UP_W_GATE is not None and W >= UP_W_GATE:
                go_up = th.zeros_like(go_up)
            go_stand = (st == 3) & (standok >= 5)
            go_abort = (st == 3) & (abort_c >= K_ABORT)
            st = th.where(go_desc, th.ones_like(st), st)
            st = th.where(go_rest, 2 * th.ones_like(st), st)
            st = th.where(go_up, 3 * th.ones_like(st), st)
            st = th.where(go_stand, th.zeros_like(st), st)
            st = th.where(go_abort, th.ones_like(st), st)
            sw = go_desc | go_up | go_abort
            refr = th.where(sw, th.full_like(refr, REFRACT), refr - 1)
        else:  # V1 / ONE-WAY (batch-1 guard)
            below = th.where(vs_bar < EPS_DN, below + 1, th.zeros_like(below))
            above = th.where(vs_bar > 0.15, above + 1, th.zeros_like(above))
            can = (refr <= 0) & (t >= WARMUP)
            go_dn = (~v1_in_rest) & (below >= K_DN) & can
            go_up = v1_in_rest & (above >= 25) & can & th.tensor(arm == "V1", device=DEV)
            v1_in_rest = th.where(go_dn, th.ones_like(v1_in_rest), v1_in_rest)
            v1_in_rest = th.where(go_up, th.zeros_like(v1_in_rest), v1_in_rest)
            refr = th.where(go_dn | go_up, th.full_like(refr, REFRACT), refr - 1)
            st = th.where(v1_in_rest, 2 * th.ones_like(st), th.zeros_like(st))
        with th.no_grad():
            a_s = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
            a = th.where((st >= 1).unsqueeze(-1) & (st <= 2).unsqueeze(-1), a_r, a_s)
            if arm == "V2":
                m_d, n_d = tw["descend"]
                a_d = th.clamp(m_d.policy._predict(n_d(obs), deterministic=True), -1, 1)
                a = th.where((st == 1).unsqueeze(-1), a_d, a)
                m_u, n_u = tw["getup"]
                a_u = th.clamp(m_u.policy._predict(n_u(obs), deterministic=True), -1, 1)
                a = th.where((st == 3).unsqueeze(-1), a_u, a)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        z = th.zeros(N, device=DEV)
        tip = inner.termination_manager._term_dones.get("fell_over", z).bool()
        slam = inner.termination_manager._term_dones.get("illegal_contact", z).bool()
        alive &= ~(tip | slam)
        # a respawned (censored) env restarts its automaton so the visuals stay sane
        rb = dones.bool() if th.is_tensor(dones) else th.zeros(N, dtype=th.bool, device=DEV)
        if rb.any():
            st = th.where(rb, th.zeros_like(st), st)
            v1_in_rest = th.where(rb, th.zeros_like(v1_in_rest), v1_in_rest)
            refr = th.where(rb, th.zeros_like(refr), refr)
        S.append(float(alive.float().mean()))
        # majority automaton state among LIVE envs -> host trunk tint
        stv = st[alive] if alive.any() else st
        maj = int(th.mode(stv).values) if stv.numel() else 0
        stmaj.append(maj)
        tint = GRAY + (RED - GRAY) * min(1.0, W / 250.0) if maj == 0 else ST_COL[maj]
        for gg in trunk: mm.geom_rgba[gg] = tint
        frames.append(np.asarray(env.render()))
    env.close()
    return frames, np.array(S), np.array(stmaj)


def tag(img, txt):
    img = np.ascontiguousarray(img).copy(); img[:26] = (img[:26] * 0.3).astype(img.dtype)
    if HAVE_PIL:
        im = Image.fromarray(img); ImageDraw.Draw(im).text((8, 6), txt, fill=(255, 255, 255)); img = np.asarray(im)
    return img


def ribbon_spans(stmaj):
    spans, s0 = [], 0
    for k in range(1, len(stmaj) + 1):
        if k == len(stmaj) or stmaj[k] != stmaj[s0]:
            spans.append((s0 * DT, k * DT, ST_HEX[int(stmaj[s0])])); s0 = k
    return spans


def graph_frame(i, sched, cond, data, spans, W, H):
    t = np.arange(STEPS) * DT
    Wv = np.array([W_of(sched, k) for k in range(STEPS)])
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    ax2.plot(t, Wv, color="#222", lw=1.0, alpha=0.25)
    ax2.plot(t[:i + 1], Wv[:i + 1], color="#222", lw=1.8, label="load W(t)")
    if cond == "gusty":
        for g0 in np.arange(0, t[-1], 2.0):
            ax2.axvspan(g0, g0 + 0.5, color="#f39c12", alpha=0.10)
    for arm in ARMS:
        ax.plot(t, data[arm], color=COL[arm], lw=1.0, alpha=0.25)
        ax.plot(t[:i + 1], data[arm][:i + 1], color=COL[arm], lw=2.4, label=f"{arm} survival")
    for (a0, a1, c) in spans:
        if a0 > t[i]: break
        ax.axvspan(a0, min(a1, t[i]), ymin=0.0, ymax=0.05, color=c, alpha=0.9)
    ax.text(0.05, 0.075, "V2 automaton state", fontsize=7, color="#333", transform=ax.get_xaxis_transform())
    ax.axvline(t[i], color="#555", lw=1.5)
    ax.set_xlim(0, t[-1]); ax.set_ylim(0, 1.05); ax2.set_ylim(0, 260)
    ax.set_xlabel("time (s)"); ax.set_ylabel("survival (no respawn)"); ax2.set_ylabel("W (N)")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9); ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout(pad=0.5); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
    return buf


if __name__ == "__main__":
    od = os.path.expanduser("~/artifacts/odd-conditioned/E084-automaton")
    for cond in ("benign", "gusty"):
        for sched in ("square", "sine"):
            grids, surv, sm = {}, {}, None
            for arm in ARMS:
                f, S, stmaj = rollout(sched, cond, arm)
                grids[arm], surv[arm] = f, S
                if arm == "V2": sm = stmaj
                print(f"{sched}/{cond} {arm}: {len(f)} frames, S(20s)={S[-1]:.2f}", flush=True)
            spans = ribbon_spans(sm)
            Hh, Ww = grids[ARMS[0]][0].shape[:2]
            SUB = 2
            idxs = list(range(0, STEPS, SUB))
            top = [np.hstack([tag(grids[a][i][:Hh, :Ww],
                                  f"{a}   alive {int(round(surv[a][i]*NENV))}/{NENV}") for a in ARMS])
                   for i in idxs]
            TW = top[0].shape[1]; GH = 320
            comb = [np.vstack([top[k], graph_frame(idxs[k], sched, cond, surv, spans, TW, GH)[:GH, :TW]])
                    for k in range(len(idxs))]
            vp = f"{od}/batch2_{sched}_{cond}.mp4"
            imageio.mimsave(vp, comb, fps=30, macro_block_size=1)
            print("video ->", vp, flush=True)
