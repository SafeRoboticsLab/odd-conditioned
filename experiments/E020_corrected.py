"""E020 — CORRECTED re-evaluation of the payload arms. Supersedes the safe_rate(up<0.4) numbers in
E015/E016/E017/E019, which were WRONG: the tensor env auto-resets a terminated env inside step_tensor, so
reading projected_gravity afterwards masked every fall (and also corrupts position-based drift). The
reset-INDEPENDENT safety metric is the FAILURE RATE from `dones & ~timeouts` (= g<0 termination = the
task's own safety def): falls per env-second (lower = safer). Force capped at <=50 N (higher is unphysical).

Three tables: (1) in-dist light-rigid / heavy-sloshy; (2) OOD 12kg sloshy / rigid; (3) mass sweep.
Compares CONDITIONED (raw theta) vs BLIND vs HISTORY.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

R = "results/go2_payload_runs"; DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.])
ARMS = ["CONDITIONED", "BLIND", "HISTORY"]
RUNS = {a: f"{R}/go2_payload_{a.lower()}_gameplaysac" for a in ARMS}


def fall_rate(run, task, fr, steps=300, nenv=128):
    env = make_tensor(task, nenv, DEV, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run}/tensornormalize.pt", env); nm.training = False
    n = nenv; dd = spec(task).dstb_dim; env.force_scale = fr * th.ones(n, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(n, dd).contiguous(); obs = env.reset()
    nfail = 0
    for _ in range(steps):
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        nfail += int((dones & (~touts)).sum())
    env.close()
    return nfail / (nenv * steps * 0.02)   # falls per env-second


def table(title, rowkey, rows, taskfn, forces):
    print(f"\n=== {title} — falls/env-sec (lower = safer) ===")
    for fr in forces:
        print(f"[{fr*50:.0f} N]  {rowkey:>10} | " + " | ".join(f"{a:>12}" for a in ARMS))
        for r in rows:
            cells = [f"{fall_rate(RUNS[a], taskfn(a, r), fr):.2f}" for a in ARMS]
            print(f"        {str(r):>10} | " + " | ".join(f"{c:>12}" for c in cells))


if __name__ == "__main__":
    # (1) in-dist
    INDIST = {"light_rigid": {"CONDITIONED": "go2_payload_conditioned_light_rigid", "BLIND": "go2_payload_light_rigid", "HISTORY": "go2_payload_history_light_rigid"},
              "heavy_sloshy": {"CONDITIONED": "go2_payload_conditioned_heavy_sloshy", "BLIND": "go2_payload_heavy_sloshy", "HISTORY": "go2_payload_history_heavy_sloshy"}}
    table("(1) IN-DIST", "payload", list(INDIST), lambda a, r: INDIST[r][a], [0.35, 1.0])
    # (2) OOD 12kg
    OOD = {"12kg_sloshy": {"CONDITIONED": "go2_payload_conditioned_ood_sloshy", "BLIND": "go2_payload_ood_sloshy", "HISTORY": "go2_payload_history_ood_sloshy"},
           "12kg_rigid": {"CONDITIONED": "go2_payload_conditioned_ood_rigid", "BLIND": "go2_payload_ood_rigid", "HISTORY": "go2_payload_history_ood_rigid"}}
    table("(2) OOD (extrapolated theta)", "payload", list(OOD), lambda a, r: OOD[r][a], [0.35, 1.0])
    # (3) mass sweep (sloshy)
    OBS = {"CONDITIONED": "conditioned", "BLIND": "blind", "HISTORY": "history"}
    table("(3) MASS SWEEP sloshy", "mass(kg)", [8, 12, 16, 20, 25], lambda a, m: f"go2_payload_sweep_{OBS[a]}_{m}", [1.0])
    print("\ntrain box: mass<=7.5kg. HISTORY/CONDITIONED < BLIND => theta/introspection helps (premise LIVES).")
