"""E029b — verify the STANCE-HOLD (action=0 = default standing pose, use_default_offset=True) really topples,
and SEE how. Render a=0 on light_rigid (1.2kg) at 0N and 50N, track trunk uprightness + first-fall step.
If it collapses immediately -> passive stance can't reject the reset perturbation (real, not artifact).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, imageio.v2 as imageio
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
try:
    from PIL import Image, ImageDraw; HAVE_PIL = True
except Exception: HAVE_PIL = False

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); DT = 0.02; STEPS = 150; NENV = 2


def run(fr):
    env = make_tensor("go2_payload_light_rigid", NENV, DEV, adversary=True, render_mode="rgb_array")
    try: env.mj.cfg.viewer.max_extra_envs = 1
    except Exception: pass
    dd = spec("go2_payload_light_rigid").dstb_dim; env.force_scale = fr * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    robot = env.mj.scene["robot"]; obs = env.reset(); frames = []; up = []; first = None
    for t in range(STEPS):
        a = th.zeros(NENV, 12, device=DEV)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        up.append(float((-robot.data.projected_gravity_b[:, 2]).mean()))
        if first is None and int((dones & (~touts)).sum()) > 0: first = t
        f = np.asarray(env.render()).copy()
        if HAVE_PIL:
            im = Image.fromarray(f); ImageDraw.Draw(im).text((6, 6), f"STANCE a=0  {fr*50:.0f}N  up={up[-1]:.2f}", fill=(255, 255, 0)); f = np.asarray(im)
        frames.append(f)
    env.close()
    return frames, np.array(up), first


if __name__ == "__main__":
    allframes = []
    for fr in (0.0, 1.0):
        frames, up, first = run(fr)
        print(f"STANCE a=0 @ {fr*50:.0f}N: first-fall step = {first} ({'never' if first is None else f'{first*DT:.2f}s'}); "
              f"uprightness t=0 {up[0]:.2f} -> t=0.5s {up[min(25,len(up)-1)]:.2f} -> end {up[-1]:.2f}")
        allframes.append(frames)
    n = min(len(a) for a in allframes); h = min(a[0].shape[0] for a in allframes); w = min(a[0].shape[1] for a in allframes)
    comb = [np.hstack([allframes[j][i][:h, :w] for j in range(2)]) for i in range(n)]
    out = "results/E029b/stance_check.mp4"; os.makedirs("results/E029b", exist_ok=True)
    imageio.mimsave(out, comb, fps=30, macro_block_size=1)
    print(f"wrote {out}")
