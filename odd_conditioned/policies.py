"""The policies: reach-avoid safety twins trained with safety-stable-baselines on robot-safety-sandbox tasks, and
the nominal walker from go2_atomic_skills.

A twin is a two-player reach-avoid PPO learner (``ReachAvoidPPO2P``). Its control player is the policy; its
state-value net is the reach-avoid certificate V(x) of the mode it was trained for (V >= 0: the mode's
specification is still certifiable from x). Both read the same 48-d actor observation, normalized with the
twin's frozen statistics.
"""
import math

import torch as th

from .paths import checkpoint
from .sim import DEV, DT, quiet, robot

# name -> what it is (task trained on; see docs/TRAINING.md for the commands)
POLICIES = {
    "stand":              "STAND expert, carried load W ~ U[0, 120] N at h = 0.25 m   (go2_weight_stand_hi, hi=120)",
    "rest":               "REST expert: settle and lie down under any load            (go2_weight_rest_hi)",
    "getup":              "get-up funnel REST -> STAND; its value is V_up              (go2_getup, warm-started)",
    "descend":            "descent funnel STAND -> REST                               (go2_descend, from rest)",
    "stand_wide":         "STAND expert trained on W ~ U[0, 150] N (noisier certificate) (go2_weight_stand_hi)",
    "unified":            "single-spec baseline: one policy for l = max(l_stand, l_rest) (go2_weight_unified_hi)",
    "unified_discounted": "unified baseline that discounts resting (prefers standing)   (go2_weight_unified_disc_hi)",
    "leg_stand":          "STAND expert for a derated front-right leg                 (go2_leg_stand)",
    "compound_stand":     "STAND expert, leg derating while carrying 80 N             (go2_compound_stand)",
    "compound_rest":      "REST expert for the compound case                          (go2_compound_rest)",
}
AUTOMATON = ("stand", "rest", "getup", "descend")


class Twin:
    """A trained reach-avoid twin: ``act(obs)`` is its deterministic control action, ``value(env)`` its
    certificate at the environment's current state."""

    def __init__(self, name: str):
        from robot_safety_sandbox.eval.policies import load_twin
        self.name, self.path = name, checkpoint(name)
        with quiet():
            self.model, self.norm = load_twin(self.path, DEV, quiet=True)
        self.model.policy.set_training_mode(False)

    def act(self, obs):
        with th.no_grad():
            return th.clamp(self.model.policy._predict(self.norm(obs), deterministic=True), -1, 1)

    def value(self, env):
        """V(x) at the current state, read from the actor observation the env cached on its last step/reset."""
        with th.no_grad():
            return self.model.policy.predict_values(self.norm(env.mj._zoo_last_obs)).squeeze(-1)


def load(names, overrides=None):
    """``{name: Twin}`` in the given order; ``overrides`` maps a name to another checkpoint (name or .zip)."""
    overrides = overrides or {}
    return {k: Twin(overrides.get(k, k)) for k in names}


class Walker:
    """The nominal task policy: go2_atomic_skills' joystick walker (trained at zero load), batched.
    It knows nothing about loads or leg faults — the safety filter owns those."""

    def __init__(self, n: int):
        from go2_atomic_skills.nets import WalkerNorm, load_actor
        from go2_atomic_skills.obs import CTRL_GAIN, DEFAULT_JOINT_POS, PHASE_STAND_EPS, WALK_PHASE_PERIOD
        self.net = load_actor("walker_actor.pt", 47, DEV)
        self.norm = WalkerNorm(DEV)
        self.default_q = th.tensor(DEFAULT_JOINT_POS, device=DEV)
        self.gain, self.period, self.stand_eps = CTRL_GAIN, WALK_PHASE_PERIOD, PHASE_STAND_EPS
        self.last_ctrl = th.zeros(n, 12, device=DEV)
        self.clock = th.zeros(n, device=DEV)

    def reset(self, mask):
        """Restart the gait clock of the envs in ``mask`` (on every return to walking)."""
        self.last_ctrl[mask] = 0.0
        self.clock[mask] = 0.0

    def act(self, env, cmd):
        """Action for joystick command ``cmd`` = (vx, vy, wz). Advances every env's gait clock."""
        d = robot(env)
        p = (self.clock * DT) % self.period / self.period
        phase = th.stack([th.sin(p * 2 * math.pi), th.cos(p * 2 * math.pi)], dim=1)
        phase = th.where((th.linalg.norm(cmd, dim=1, keepdim=True) < self.stand_eps), th.zeros_like(phase), phase)
        obs = th.cat([d.root_link_ang_vel_b, d.projected_gravity_b, cmd, phase,
                      d.joint_pos - self.default_q, d.joint_vel, self.last_ctrl], dim=1)
        with th.no_grad():
            a = th.clamp(self.net(self.norm(obs)), -1.0, 1.0)
        self.last_ctrl = self.gain * a
        self.clock += 1
        return a


def yaw_of(d):
    q = d.root_link_quat_w    # (N, 4) w, x, y, z
    return th.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def steer_cmd(d, goal, off):
    """Joystick command toward ``goal``: vx = 1 m/s, wz = P(heading error); zero where ``off``."""
    pos = d.root_link_pos_w[:, :2]
    gvec = goal - pos
    hd_err = th.atan2(gvec[:, 1], gvec[:, 0]) - yaw_of(d)
    hd_err = th.atan2(th.sin(hd_err), th.cos(hd_err))
    cmd = th.zeros(pos.shape[0], 3, device=DEV)
    cmd[:, 0] = 1.0
    cmd[:, 2] = th.clamp(1.5 * hd_err, -0.5, 0.5)
    cmd[off] = 0.0
    return cmd


class LegResidual:
    """Torque-saturation residual: the worst leg's mean gap between the demanded PD torque and the torque the
    motors actually deliver, as a fraction of nominal. A derated motor shows a persistent positive gap. The robot
    knows its own PD command and senses motor current; nothing here reads θ."""

    def __init__(self, env):
        import mujoco
        import numpy as np
        self.env = env
        mm = env.mj.sim.mj_model
        jn = list(env.mj.scene["robot"].joint_names)
        amap = [jn.index(mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, mm.actuator_trnid[a, 0]).split("/")[-1])
                for a in range(mm.nu)]
        self.amap = th.tensor(amap, device=DEV, dtype=th.long)
        self.kp = th.tensor(mm.actuator_gainprm[:, 0], device=DEV, dtype=th.float32)
        self.kd = th.tensor(-mm.actuator_biasprm[:, 2], device=DEV, dtype=th.float32)
        self.f_nom = th.tensor(np.abs(mm.actuator_forcerange[:, 1]), device=DEV, dtype=th.float32).clamp_min(1.0)
        self.legs = th.tensor([[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], device=DEV)   # FL FR RL RR

    def __call__(self):
        d, data = robot(self.env), self.env.mj.sim.data
        q, qd = d.joint_pos[:, self.amap], d.joint_vel[:, self.amap]
        demand = self.kp[None] * (data.ctrl - q) - self.kd[None] * qd
        gap = (demand.abs() - data.actuator_force.abs()).clamp(min=0) / self.f_nom[None]
        return gap[:, self.legs].mean(-1).max(dim=1).values
