"""E086 — single-event schedules over 24 s (Buzi's test): does certified bidirectionality pay when the ODD
excursion happens ONCE?
  pulse : W = 40 N, jumps to 220 N for t in [8,16) s, back to 40 N   (light | heavy | light, 8 s each)
  period: W = 130 - 110*cos(2*pi*t/24)  — one full sine period (starts/ends 20 N, peak 240 N at t=12)
With one excursion, V2 pays the descent floor once (like ONE-WAY) and its near-free certified get-up should
recover the final light phase — the transit-count argument's best case. Survival protocol as E084 (N=256,
no respawn); same 6 arms; videos as E085 (V1 | V2 | ONE-WAY).
"""
import os, sys, math, json
from _paths import _ART
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import E084_automaton as E
import E085_video as EV
STNAME = {0: 'STAND', 1: 'DESCENDING', 2: 'REST', 3: 'GETTING-UP'}

STEPS, DT = 1200, 0.02


def W_of(sched, t):
    s = t * DT
    if sched == "pulse":
        return 220.0 if 8.0 <= s < 16.0 else 40.0
    return 130.0 - 110.0 * math.cos(2 * math.pi * s / 24.0)


E.STEPS = STEPS; E.W_of = W_of
EV.STEPS = STEPS; EV.W_of = W_of
E.UP_W_GATE = 130.0; EV.UP_W_GATE = 130.0   # belief-gated return: only when W back in stand ODD

if __name__ == "__main__":
    od = os.path.expanduser(_ART + "/E084-automaton")
    out = {}
    for cond in ("benign", "gusty"):
        for sched in ("pulse", "period"):
            print(f"\n===== {sched} / {cond} =====", flush=True)
            print(f"{'arm':>10} {'S(24s)':>7} {'afford|alive':>12}  deaths by state", flush=True)
            for arm in ["V2", "V2-REUSE", "V1", "ONE-WAY", "STAND-ONLY", "REST-ONLY"]:
                r = E.rollout(sched, cond, arm)
                out[f"{cond}|{sched}|{arm}"] = r
                dd = " ".join(f"{k}:{v}" for k, v in r["deaths"].items() if v)
                print(f"{arm:>10} {r['final']:>7.2f} {r['afford_alive']:>12.2f}  {dd}", flush=True)
    json.dump(out, open(f"{od}/results_single.json", "w"))
    print(f"\nsaved -> {od}/results_single.json", flush=True)

    # videos (benign — the behavioral showcase)
    for sched in ("pulse", "period"):
        cond = "benign"
        grids, surv, sm = {}, {}, None
        for arm in EV.ARMS:
            f, S, stmaj = EV.rollout(sched, cond, arm)
            grids[arm], surv[arm] = f, S
            if arm == "V2": sm = stmaj
            print(f"video {sched}/{cond} {arm}: S={S[-1]:.2f}", flush=True)
        spans = EV.ribbon_spans(sm)
        Hh, Ww = grids[EV.ARMS[0]][0].shape[:2]
        idxs = list(range(0, STEPS, 2))
        top = [np.hstack([EV.tag(grids[a][i][:Hh, :Ww],
                                 f"{a}"
                                 + (f" [{STNAME[int(sm[i])]}]" if a == "V2" else "")
                                 + f"   alive {int(round(surv[a][i]*EV.NENV))}/{EV.NENV}") for a in EV.ARMS])
               for i in idxs]
        TW = top[0].shape[1]; GH = 320
        comb = [np.vstack([top[k], EV.graph_frame(idxs[k], sched, cond, surv, spans, TW, GH)[:GH, :TW]])
                for k in range(len(idxs))]
        import imageio.v2 as imageio
        vp = f"{od}/single_{sched}.mp4"
        imageio.mimsave(vp, comb, fps=30, macro_block_size=1)
        print("video ->", vp, flush=True)
