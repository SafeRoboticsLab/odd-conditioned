"""Measure the state distribution at which the stand expert takes over in the payload walk (the BRAKE entry):
the walker walks toward the goal, the load ramps up (E092 'period' schedule), and the belief trigger fires once
W >= 60 N for 3 steps. Prints percentiles of the entry states, so the stand expert's training start
distribution can be matched to them (the 'enter every transition inside its trained start set' rule).

    python scripts/probe_brake_entry.py [--n 256]
"""
import argparse
import contextlib
import io
import sys

import torch as th

sys.path.insert(0, "experiments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    import E089_goal_walk as G
    import E092_payload_walk as E
    from robot_safety_sandbox import make_tensor, spec
    n, dev = a.n, E.DEV
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", n, dev, adversary=True)
    inner = env.mj
    inner.cfg.episode_length_s = 10_000.0
    walker = G.Walker(n)
    dstb = th.zeros(n, spec("go2_weight_rest_hi_at_0").dstb_dim, device=dev); dstb[:, 1] = 1.0
    env.reset()
    d = inner.scene["robot"].data
    yaw0 = G.yaw_of(d)
    goal = d.root_link_pos_w[:, :2].clone()
    goal[:, 0] += E.GOAL_D * th.cos(yaw0); goal[:, 1] += E.GOAL_D * th.sin(yaw0)
    hold = th.randint(0, 30, (n,), device=dev)
    reached = th.zeros(n, dtype=th.bool, device=dev)
    wcnt, rec = 0, None
    djp = th.tensor(G.DEFAULT_JOINT_POS, device=dev)
    for t in range(E.STEPS):
        W, h = E.Wh_of("period", t)
        env.base_load = th.tensor([0., 0., -W], device=dev)[None].expand(n, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = h
        env.force_scale = 0.0 * th.ones(n, device=dev)
        wcnt = wcnt + 1 if W >= E.W_TRIG else 0
        if wcnt >= 3 and t >= 100:
            rec = {"t": t * E.DT, "W": W,
                   "lin_vel_b": d.root_link_lin_vel_b.clone(), "ang_vel_b": d.root_link_ang_vel_b.clone(),
                   "tilt": th.maximum(d.projected_gravity_b[:, 0].abs(), d.projected_gravity_b[:, 1].abs()),
                   "height": d.root_link_pos_w[:, 2].clone(),
                   "joint_off": d.joint_pos - djp, "joint_vel": d.joint_vel.clone()}
            break
        cmd = G.steer_cmd(d, goal, reached, th.ones(n, dtype=th.bool, device=dev))
        cmd[t < hold] = 0.0
        a_w = walker.act(inner, cmd)
        env.step_tensor(th.cat([a_w, dstb], dim=1))
    env.close()
    q = th.tensor([0.05, 0.5, 0.95], device=dev)

    def show(name, x):
        x = x.flatten().float()
        p = th.quantile(x, q)
        print(f"{name:28s} p5 {p[0]:+.3f}  p50 {p[1]:+.3f}  p95 {p[2]:+.3f}  |max| {x.abs().max():.3f}")

    print(f"brake entry at t={rec['t']:.2f}s, W={rec['W']:.0f} N (period schedule, no push), n={n}")
    for i, ax in enumerate("xyz"):
        show(f"base lin vel {ax} (body, m/s)", rec["lin_vel_b"][:, i])
    for i, ax in enumerate("xyz"):
        show(f"base ang vel {ax} (body, rad/s)", rec["ang_vel_b"][:, i])
    show("tilt (|g_xy| max)", rec["tilt"])
    show("base height (m)", rec["height"])
    show("joint offset from default (rad)", rec["joint_off"])
    show("joint velocity (rad/s)", rec["joint_vel"])


if __name__ == "__main__":
    main()
