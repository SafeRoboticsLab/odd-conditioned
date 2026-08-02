"""E029c — is the STANCE-HOLD failure a TOPPLE or a SAG? The post-step uprightness is masked by auto-reset.
Sample uprightness + base height at the START of each step (pre-reset), and at each termination record the
state that ENTERED the terminal step (prev_up, prev_z) ~ the fall condition. up<<1 => topple; up~1 & z low => sag.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); DT = 0.02; STEPS = 250; NENV = 128


def run(fr):
    env = make_tensor("go2_payload_light_rigid", NENV, DEV, adversary=True)
    dd = spec("go2_payload_light_rigid").dstb_dim; env.force_scale = fr * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    robot = env.mj.scene["robot"]; env.reset()
    ups, zs = [], []; nfail = 0
    for t in range(STEPS):
        prev_up = (-robot.data.projected_gravity_b[:, 2]).clone()      # pre-step (state entering this step)
        prev_z = robot.data.root_link_pos_w[:, 2].clone()
        a = th.zeros(NENV, 12, device=DEV)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        m = dones & (~touts)
        if int(m.sum()) > 0:
            ups.append(prev_up[m]); zs.append(prev_z[m]); nfail += int(m.sum())
    env.close()
    up = th.cat(ups) if ups else th.tensor([1.0]); z = th.cat(zs) if zs else th.tensor([0.32])
    return nfail / (NENV * STEPS * DT), up, z


if __name__ == "__main__":
    print("\nE029c stance a=0 failure MODE (state one step before terminal reset):")
    for fr in (0.0, 1.0):
        rate, up, z = run(fr)
        print(f"  {fr*50:4.0f}N: {rate:.2f} falls/env-s | pre-fall uprightness mean {up.mean():.2f} "
              f"min {up.min():.2f} (1=upright) | base-z mean {z.mean():.2f} min {z.min():.2f} m (spawn 0.32)")
    print("  up<<1 => TOPPLE; up~1 & z<0.15 => SAG. (CORNER_LOW 0.10, trunk half-h 0.05)")
