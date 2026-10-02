"""E040 (probe A — ATTRIBUTION CONTROL). E039 showed retraining under the sound 0.4.0 critic collapses the
policy. The professor flagged that "only §7 (entropy removed from the critic target) changed" is NOT airtight:
the runs also ride our hand-ported RSS 0.4.0 env, so a port slip (margins/obs/force curriculum/assets) would
give the same signature. This isolates the two: take the KNOWN-COMPETENT v0.3.0 blind actor (reached ~0.95),
graft it into a fresh 0.4.0 ReachAvoidSAC2P, and eval it in the 0.4.0-BUILT env under the SAME protocol as
E037 (fixed +y pull, corrected falls/env-sec = (dones & ~timeouts)).

  * If the grafted v0.3.0 policy REPRODUCES E037's numbers in the 0.4.0 env -> the port is CLEAN and the E039
    collapse is attributable to §7 (the critic-target change), as the professor argues.
  * If it FALLS APART -> the 0.4.0 RSS port changed the env; the E039 story is a port bug, not §7. Audit
    margins/force curriculum first.

Mechanics: load a valid 0.4.0 ReachAvoidSAC2P instance (an E039 blind checkpoint) as the chassis, overwrite its
actor (and dstb) with the v0.3.0 weights (from the v0.3.0 blind final_model.zip policy.pth), and use the v0.3.0
tensornormalize (the obs stats that actor expects). Run via `source activate.sh` (0.4.0 overlay).

E037 blind reference (falls/env-sec | frac-fell):
  0N : 8k 0.00|0.00 · 12k 0.14|0.61 · 16k 0.42|0.99 · 20k 0.53|1.00
  50N: 8k 0.10|0.47 · 12k 0.11|0.40 · 16k 0.26|0.65 · 20k 0.30|0.71
"""
import os, sys, io, zipfile
os.environ.setdefault("MUJOCO_GL", "egl")
import torch as th
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); DT = 0.02; STEPS = 300; NENV = 128
MASSES = [8, 12, 16, 20]
CHASSIS = "/tmp/e039blind/model_24999936_steps.zip"                 # valid 0.4.0 ReachAvoidSAC2P
V030_ZIP = "results/go2_payload_runs/go2_payload_blind_gameplaysac/final_model.zip"
V030_NORM = "results/go2_payload_runs/go2_payload_blind_gameplaysac/tensornormalize.pt"


def v030_weights():
    z = zipfile.ZipFile(V030_ZIP)
    sd = th.load(io.BytesIO(z.read("policy.pth")), map_location=DEV, weights_only=False)
    actor = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
    dstb = {k[len("dstb_actor."):]: v for k, v in sd.items() if k.startswith("dstb_actor.")}
    return actor, dstb


def build(task):
    """Fresh 0.4.0 ReachAvoidSAC2P (E039 chassis) with the v0.3.0 blind actor+dstb grafted in."""
    env = make_tensor(task, NENV, DEV, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True, family="off_policy"))   # ReachAvoidSAC2P
    m = Algo.load(CHASSIS, env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    actor_sd, dstb_sd = v030_weights()
    miss, unexp = m.policy.actor.load_state_dict(actor_sd, strict=False)
    assert not [k for k in miss if k.startswith(("latent_pi", "mu.", "log_std."))], f"actor graft miss: {miss}"
    if hasattr(m.policy, "dstb_actor") and dstb_sd:
        m.policy.dstb_actor.load_state_dict(dstb_sd, strict=False)
    nm = TensorVecNormalize.load(V030_NORM, env); nm.training = False
    return env, m, nm


def run(env, m, nm, task, fr):
    dd = spec(task).dstb_dim; env.force_scale = fr * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    obs = env.reset(); nfall = 0; fell = th.zeros(NENV, dtype=th.bool, device=DEV)
    for _ in range(STEPS):
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        fail = dones & (~touts); nfall += int(fail.sum()); fell |= fail
    return nfall / (NENV * STEPS * DT), float(fell.float().mean())


if __name__ == "__main__":
    print("\nE040 ATTRIBUTION CONTROL — v0.3.0 blind actor grafted into 0.4.0 ReachAvoidSAC2P + 0.4.0 env.")
    print("cells: falls/env-sec | frac-fell   (compare to E037 in the header)")
    print(f"{'mass':>5} | {'force 0N':>18} | {'force 50N':>18}")
    for mass in MASSES:
        task = f"go2_payload_sweep_blind_{mass}"
        env, m, nm = build(task)
        c0 = run(env, m, nm, task, 0.0)
        c1 = run(env, m, nm, task, 1.0)
        env.close()
        print(f"{mass:>4}k | {c0[0]:7.2f} | {c0[1]:8.2f} | {c1[0]:7.2f} | {c1[1]:8.2f}")
    print("\nMATCH E037 => port CLEAN, E039 collapse is §7 (entropy-in-target removal). DIVERGE => port bug.")
