"""E037 (stage 1 of the KILL-TEST) — WHERE do the trained arms genuinely break, and is there LEAD TIME?

The project's central premise — "the arms fail SILENTLY out-of-envelope, so you need B̂ to know" — is still
an ASSERTION: blind/history survived every OOD we tried (E019 said so, but with the PRE-FIX masked metric).
Before investing further in B̂ machinery we need a regime where a policy ACTUALLY FAILS, otherwise there is
no t_fail to compare a detector against.

This sweeps payload mass 8..25 kg (sloshy top-heavy; tasks are BUILT at each mass so model+inertia stay
consistent — no runtime-override blowup) for BLIND / HISTORY / CONDITIONED, and reports, with the CORRECTED
reset-independent metric `(dones & ~timeouts)`:
  * falls/env-sec           — the E020 metric
  * frac of envs that fell  — did it break at all
  * median t_first_fall     — IS THERE LEAD TIME? a detector is only useful if failure is not instant

Static θ throughout (mass fixed per task), which is deliberate: it keeps B̂ valid as-is in stage 2, with no
drift-inflation prerequisite. Stage 2 (three-timestamp comparison) runs at whichever mass breaks first with
a non-trivial t_fail.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

R = "results/go2_payload_runs"; DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.])
DT = 0.02; STEPS = 300; NENV = 128
MASSES = [8, 12, 16, 20, 25]
ARMS = {"BLIND": (f"{R}/go2_payload_blind_gameplaysac", "blind"),
        "HISTORY": (f"{R}/go2_payload_history_gameplaysac", "history"),
        "CONDITIONED": (f"{R}/go2_payload_conditioned_gameplaysac", "conditioned")}


def run(run_dir, task, fr):
    env = make_tensor(task, NENV, DEV, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    dd = spec(task).dstb_dim; env.force_scale = fr * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    obs = env.reset()
    nfall = 0
    t_first = th.full((NENV,), -1.0, device=DEV)
    for t in range(STEPS):
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        fail = dones & (~touts)                       # reset-independent: g<0 termination
        nfall += int(fail.sum())
        fresh = fail & (t_first < 0)                  # FIRST fall only
        t_first = th.where(fresh, th.full_like(t_first, t * DT), t_first)
    env.close()
    fell = t_first >= 0
    med = float(t_first[fell].median()) if bool(fell.any()) else float("nan")
    return nfall / (NENV * STEPS * DT), float(fell.float().mean()), med


if __name__ == "__main__":
    arms = list(ARMS)
    print(f"\nE037 BREAK SWEEP — sloshy top-heavy, train max 7.5 kg, {STEPS*DT:.1f}s episodes, n={NENV}.")
    print("cells: falls/env-sec | frac-fell | median t_first_fall (s)")
    for fr in (0.0, 1.0):
        print(f"\n=== force {fr:.1f} ({fr*50:.0f} N pull) ===")
        print(f"{'mass':>5} | " + " | ".join(f"{a:^26}" for a in arms))
        for mass in MASSES:
            cells = []
            for a in arms:
                rd, obs = ARMS[a]
                try:
                    fps, frac, med = run(rd, f"go2_payload_sweep_{obs}_{mass}", fr)
                    cells.append(f"{fps:5.2f} | {frac:4.2f} | {med:5.2f}")
                except Exception as e:
                    cells.append(f"ERR {type(e).__name__}")
            print(f"{mass:>4}k | " + " | ".join(f"{c:^26}" for c in cells))
    print("\nPick for stage 2: the lowest mass where an arm's frac-fell is substantial AND median t_first_fall")
    print("is late enough (>~1s) that a detector could plausibly have warned in time.")
