"""E035 — B̂ -> FALLBACK filter demo. On an OOD payload (12kg sloshy, beyond the <=7.5kg training box), the
set-membership B̂ (E018) fires its EMPTY-SET OOD flag (~delta*=0.1-0.2s). The filter then HANDS OFF from the
standing policy (blind, whose guarantee is only certified inside the ODD box) to the trained soft-descent
fallback, which lowers the robot to a low feet-supported pose that is safe under the CONTACT-FORCE safe set.
We compare, under the OOD payload + a pull:
  NO_FILTER : blind standing the whole episode (gambling on the uncertified policy)
  FILTER    : blind standing until the B̂ OOD flag (t=T_HO), then the descent fallback
Metric (end_criterion=timeout -> no reset, so up/height are the TRUE state): topple frac (up<0.3 ever),
reached-low frac (descended & still level), and peak non-foot contact force. Both policies read the SAME
47-dim blind obs, so the handoff is a clean policy swap in one env.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, imageio.v2 as imageio
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize
try:
    from PIL import Image, ImageDraw; HAVE_PIL = True
except Exception: HAVE_PIL = False

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); DT = 0.02
OOD_TASK = "go2_payload_ood_sloshy"                 # 12kg sloshy, BUILT (correct inertia), blind 47-dim obs
STAND = "results/go2_payload_runs/go2_payload_blind_gameplaysac"
DRUN = "results/go2_payload_runs/go2_payload_descent_reachavoidsac"
DESC_MODEL = f"{DRUN}/checkpoints/model_74999808_steps.zip"; DESC_NORM = f"{DRUN}/checkpoints/tensornorm_75001344.pt"
FR = 0.4; T_HO = 20; STEPS = 160                    # 20N pull; B̂ OOD flag -> handoff at 0.4s; 3.2s episode


def load_descent():
    e = make_tensor("go2_payload_descent", 2, DEV, adversary=False)      # matching action space for load
    Algo = getattr(safety_sb3, algo_name("go2_payload_descent", adversary=False).replace("PPO", "SAC"))
    m = Algo.load(DESC_MODEL, env=e, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(DESC_NORM, e); nm.training = False; e.close()
    return m, nm


def run(filter_on, nenv, render, desc_m, desc_nm):
    env = make_tensor(OOD_TASK, nenv, DEV, adversary=True, end_criterion="timeout",
                      **({"render_mode": "rgb_array"} if render else {}))
    if render:
        try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
        except Exception: pass
    SAlgo = getattr(safety_sb3, algo_name(OOD_TASK, adversary=True).replace("PPO", "SAC"))
    sm = SAlgo.load(f"{STAND}/final_model.zip", env=env, device=DEV,
                    custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    snm = TensorVecNormalize.load(f"{STAND}/tensornormalize.pt", env); snm.training = False
    dd = spec(OOD_TASK).dstb_dim; env.force_scale = FR * th.ones(nenv, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, dd).contiguous()
    robot = env.mj.scene["robot"]; sens = env.mj.scene["nonfoot_ground_touch"]; obs = env.reset()
    toppled = th.zeros(nenv, dtype=th.bool, device=DEV); fmax = th.zeros(nenv, device=DEV); frames = []
    for t in range(STEPS):
        use_desc = filter_on and t >= T_HO
        with th.no_grad():
            if use_desc:
                a = th.clamp(desc_m.policy._predict(desc_nm.normalize_obs(obs), deterministic=True), -1, 1)
            else:
                a = th.clamp(sm.policy._predict(snm.normalize_obs(obs), deterministic=True), -1, 1)
        d_use = th.zeros_like(dstb) if use_desc else dstb          # fallback isn't fighting the pull; it yields
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, d_use], dim=1))
        up = -robot.data.projected_gravity_b[:, 2]
        toppled |= (up < 0.3)
        fh = sens.data.force_history if sens.data.force_history is not None else sens.data.force
        fmax = th.maximum(fmax, th.norm(fh, dim=-1).flatten(1).amax(1))
        if render:
            fr = np.asarray(env.render()).copy()
            if HAVE_PIL:
                lab = ("FILTER: DESCEND" if use_desc else "FILTER: stand") if filter_on else "NO FILTER: stand"
                im = Image.fromarray(fr); ImageDraw.Draw(im).text((6, 6), f"{lab}  t={t*DT:.1f}s", fill=(255, 255, 0)); fr = np.asarray(im)
            frames.append(fr)
    base_z = robot.data.root_link_pos_w[:, 2]; up_end = -robot.data.projected_gravity_b[:, 2]
    reached_low = ((base_z < 0.22) & (up_end > 0.4))
    env.close()
    return dict(topple=float(toppled.float().mean()), low=float(reached_low.float().mean()),
                fmax_p90=float(th.quantile(fmax, 0.9)), fmax_max=float(fmax.max())), frames


if __name__ == "__main__":
    desc_m, desc_nm = load_descent()
    print(f"E035 B̂->fallback handoff on OOD 12kg sloshy + {FR*50:.0f}N pull (handoff at t={T_HO*DT:.1f}s). timeout, no reset.")
    for name, filt in [("NO_FILTER", False), ("FILTER", True)]:
        r, _ = run(filt, 128, False, desc_m, desc_nm)
        print(f"  {name:10}: topple {r['topple']:.2f} | reached-low {r['low']:.2f} | non-foot force p90 {r['fmax_p90']:.0f} max {r['fmax_max']:.0f} N")
    # side-by-side video
    _, fno = run(False, 6, True, desc_m, desc_nm)
    _, fyes = run(True, 6, True, desc_m, desc_nm)
    h = min(fno[0].shape[0], fyes[0].shape[0]); w = min(fno[0].shape[1], fyes[0].shape[1])
    comb = [np.hstack([fno[i][:h, :w], fyes[i][:h, :w]]) for i in range(min(len(fno), len(fyes)))]
    os.makedirs("results/E035", exist_ok=True); imageio.mimsave("results/E035/filter_handoff.mp4", comb, fps=30, macro_block_size=1)
    print(f"wrote results/E035/filter_handoff.mp4  (left NO_FILTER | right FILTER)")
