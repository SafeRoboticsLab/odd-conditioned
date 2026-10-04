"""E079 — dense grid sweeps for F2, the certifiable-region map.
Panel A (weight demo): (W x pull) grid, high-CoM h=0.25, policies = E075-recal stand + rest_hi.
Panel B (compound demo): (theta x pull) grid at fixed W=80/h=0.25, policies = compound_stand + compound_rest.
Per cell: fail fraction = tip (fell_over) OR slam (> 80+1.3W). One env build per (config,policy); cells via reset.
"""
import os, sys, io, contextlib, json
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, N, STEPS = "cuda:0", 96, 250
WS = [0, 30, 60, 90, 120, 150, 180, 210, 240]
THS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
PULLS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]          # x50N = 5..35N
CK = {
    "w_stand": "results/go2_weight_runs/E075_recal/go2_weight_stand_hi_adv/checkpoints/model_49999872_steps.zip",
    "w_rest": "results/go2_weight_runs/go2_weight_rest_hi_adv/checkpoints/model_49999872_steps.zip",
    "c_stand": "results/go2_compound_runs/go2_compound_stand_adv/checkpoints/model_49999872_steps.zip",
    "c_rest": "results/go2_compound_runs/go2_compound_rest_adv/checkpoints/model_49999872_steps.zip",
}
PULLV = th.tensor([0., 1., 0.])


def run_cell(env, inner, model, norm, dstb, W, theta, fr):
    env.force_scale = fr * th.ones(N, device=DEV)
    obs = env.reset()
    fr_ids = getattr(inner, "_fr_act_ids", None)
    nom = getattr(inner, "_fr_nominal_forcerange", None)
    fail = th.zeros(N, dtype=th.bool, device=DEV)
    cap = 80.0 + 1.3 * W
    for t in range(STEPS):
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous() if W > 0 else None
        if hasattr(inner, "_weight_W"): inner._weight_W[:] = W
        if hasattr(inner, "_weight_h"): inner._weight_h[:] = 0.25
        if theta is not None and fr_ids is not None:
            inner._fr_torque_frac[:] = theta
            inner.sim.model.actuator_forcerange[:, fr_ids, :] = nom * theta
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        z = th.zeros(N, device=DEV)
        tip = inner.termination_manager._term_dones.get("fell_over", z).bool()
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        slam = th.norm(fh, dim=-1).flatten(1).amax(1) > cap
        fail |= tip | slam
    return float(fail.float().mean())


def sweep(task, ck, cells, theta_mode):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(task, N, DEV, adversary=True)
        model, norm = load_twin(ck, DEV, quiet=True)
    inner = env.mj
    _ensure_fr_cache(inner)
    dstb = (PULLV).to(DEV)[None].expand(N, spec(task).dstb_dim).contiguous()
    out = {}
    for (a, fr) in cells:
        W, theta = (a, None) if not theta_mode else (80.0, a)
        out[f"{a}|{fr}"] = run_cell(env, inner, model, norm, dstb, W, theta, fr)
        print(f"  {task} {'θ' if theta_mode else 'W'}={a} pull={fr*50:.0f}N fail={out[f'{a}|{fr}']:.2f}", flush=True)
    env.close()
    return out


if __name__ == "__main__":
    res = {}
    cellsW = [(W, fr) for W in WS for fr in PULLS]
    cellsT = [(t, fr) for t in THS for fr in PULLS]
    res["w_stand"] = sweep("go2_weight_rest_hi_at_0", CK["w_stand"], cellsW, False)
    res["w_rest"] = sweep("go2_weight_rest_hi_at_0", CK["w_rest"], cellsW, False)
    res["c_stand"] = sweep("go2_compound_rest_at_100", CK["c_stand"], cellsT, True)
    res["c_rest"] = sweep("go2_compound_rest_at_100", CK["c_rest"], cellsT, True)
    out = os.path.expanduser(_ART + "/E077-figures")
    os.makedirs(out, exist_ok=True)
    json.dump({"WS": WS, "THS": THS, "PULLS": PULLS, "res": res}, open(f"{out}/F2_grid.json", "w"), indent=2)
    print("saved ->", f"{out}/F2_grid.json")
