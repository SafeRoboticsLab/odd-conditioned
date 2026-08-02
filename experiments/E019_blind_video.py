"""Render the BLIND policy on light-rigid and heavy-sloshy at increasing force (50/150/250 N) — to SEE why
blind is so robust (it just reacts). One mp4 per payload, the 3 force levels side by side (labeled)."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, imageio.v2 as imageio
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize
try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

BLIND = "results/go2_payload_runs/go2_payload_blind_gameplaysac"
DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); FORCES = [1.0, 3.0, 5.0]


def render(task, fr, steps=200, nenv=2):
    env = make_tensor(task, nenv, DEV, adversary=True, render_mode="rgb_array")
    try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
    except Exception: pass
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    model = Algo.load(f"{BLIND}/final_model.zip", env=env, device=DEV,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    norm = TensorVecNormalize.load(f"{BLIND}/tensornormalize.pt", env); norm.training = False
    dd = spec(task).dstb_dim
    env.force_scale = fr * th.ones(nenv, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, dd).contiguous()
    obs = env.reset(); frames = []
    for _ in range(steps):
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, *_ = env.step_tensor(th.cat([a, dstb], dim=1))
        frames.append(np.asarray(env.render()))
    env.close(); return frames


def label(frames, txt):
    out = []
    for f in frames:
        f = np.ascontiguousarray(f).copy(); f[:32, :, :] = (f[:32, :, :] * 0.3).astype(f.dtype)
        if HAVE_PIL:
            im = Image.fromarray(f); ImageDraw.Draw(im).text((6, 8), txt, fill=(255, 255, 255)); f = np.asarray(im)
        out.append(f)
    return out


if __name__ == "__main__":
    for task, name in [("go2_payload_light_rigid", "light_rigid"), ("go2_payload_heavy_sloshy", "heavy_sloshy")]:
        panels = [label(render(task, fr), f"BLIND {name} {fr*50:.0f}N") for fr in FORCES]
        n = min(len(p) for p in panels)
        h = min(p[0].shape[0] for p in panels); w = min(p[0].shape[1] for p in panels)
        comb = [np.hstack([panels[j][i][:h, :w] for j in range(len(FORCES))]) for i in range(n)]
        out = f"results/E019/blind_{name}_forces.mp4"; os.makedirs("results/E019", exist_ok=True)
        imageio.mimsave(out, comb, fps=30, macro_block_size=1)
        print(f"wrote {out} ({n} frames, {comb[0].shape[1]}x{comb[0].shape[0]}) forces={[int(f*50) for f in FORCES]}N")
