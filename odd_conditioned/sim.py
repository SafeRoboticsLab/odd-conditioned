"""Thin layer over the robot-safety-sandbox GPU environments: build and seed a batch, drive the ODD, read outcomes.

Every experiment evaluates on an RSS task with the learned adversary switched off: the disturbance channel
carries a scripted lateral push instead (``push_scale``), and the ODD (carried load W at height h, front-right
leg torque fraction θ) is written into the simulation every control step.

Seeding: ``make_env`` seeds torch before building the batch (the env draws its internal generators then) and
``reset`` seeds again before the reset, so a rollout depends only on its seed, not on how many models were
loaded in between.
"""
import contextlib
import io
import os
import warnings

import torch as th

os.environ.setdefault("MUJOCO_GL", "egl")
# Harmless library notices that would otherwise interleave with every result table.
warnings.filterwarnings("ignore", message="Use of index_put_ on expanded tensors", category=UserWarning)
warnings.filterwarnings("ignore", message="You are trying to run .* on the GPU", category=UserWarning)

DEV = "cuda:0"
DT = 0.02                                   # control period: 50 Hz
FORCE_MAX = 50.0                            # push magnitude at push scale 1 (N)
WEIGHT_TASK = "go2_weight_rest_hi_at_0"     # evaluation surface of the weight-ladder and walking experiments
LOAD_H = 0.25                               # default height of a carried load's centre of mass above the base (m)
GUST_PERIOD, GUST_LEN = 2.0, 0.5            # gusty pushes: one 0.5 s gust every 2 s

# push condition -> (ambient, gust) scale; a scale s is a constant lateral (+y) force of s * FORCE_MAX
PUSH = {"none": (0.0, 0.0), "benign": (0.2, 0.2), "medium": (0.2, 0.4), "gusty": (0.2, 0.7)}


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def make_env(task: str, n: int, seed: int, render: bool = False):
    """An ``n``-env GPU batch of RSS ``task`` with the disturbance channel exposed (scripted, not learned)."""
    from robot_safety_sandbox import make_tensor
    th.manual_seed(seed)
    with quiet():
        env = make_tensor(task, n, DEV, adversary=True, **({"render_mode": "rgb_array"} if render else {}))
        if render:
            try:
                env.mj.cfg.viewer.max_extra_envs = max(1, n - 1)
            except Exception:  # noqa: BLE001 - older viewers render the host env only
                pass
    env.task_id, env.n = task, n
    return env


def reset(env, seed: int):
    th.manual_seed(seed)
    return env.reset()


def endless(env):
    """Disable the episode timeout: an evaluation never gets a free mid-rollout reset."""
    env.mj.cfg.episode_length_s = 10_000.0


def loose_termination(env):
    """Death = a genuine flip-over (tilt > 80 deg); the non-foot contact termination is switched off."""
    tm = env.mj.termination_manager
    names = getattr(tm, "_term_names", None) or tm.active_terms
    for name, cfg in zip(names, tm._term_cfgs):
        if name == "fell_over":
            cfg.params["limit_angle"] = 1.3962634
        if name == "illegal_contact":
            cfg.params["force_threshold"] = 1e9


def lateral_push(env):
    """The scripted disturbance action: a unit +y direction for every env (magnitude set by ``push_scale``)."""
    from robot_safety_sandbox import spec
    d = th.zeros(env.n, spec(env.task_id).dstb_dim, device=DEV)
    d[:, 1] = 1.0
    return d


def push_scale(push: str, t: int) -> float:
    ambient, gust = PUSH[push]
    return gust if ((t * DT) % GUST_PERIOD) < GUST_LEN else ambient


def set_push(env, scale: float):
    env.force_scale = scale * th.ones(env.n, device=DEV)


def set_load(env, W: float, h: float = LOAD_H):
    """A carried load: weight W (N) whose centre of mass sits h (m) above the base — an inverted pendulum."""
    env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(env.n, 3).contiguous()
    env.mj._weight_W[:] = W
    env.mj._weight_h[:] = h


class LegFault:
    """Front-right leg actuator derating: the FR motors' torque limits scaled to a fraction θ of nominal."""

    def __init__(self, env):
        from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache
        self.inner = env.mj
        _ensure_fr_cache(self.inner)
        self.ids, self.nominal = self.inner._fr_act_ids, self.inner._fr_nominal_forcerange

    def set(self, theta: float):
        self.inner.sim.model.actuator_forcerange[:, self.ids, :] = self.nominal * float(theta)
        self.inner._fr_torque_frac[:] = float(theta)


def robot(env):
    return env.mj.scene["robot"].data


def _term(env, name):
    return env.mj.termination_manager._term_dones.get(name, th.zeros(env.n, device=DEV)).bool()


def tipped(env):
    return _term(env, "fell_over")


def failed(env):
    """The task's own failure terminations this step: fell over, or a non-foot body hit the ground hard."""
    return _term(env, "fell_over") | _term(env, "illegal_contact")


def tilt(env):
    pg = robot(env).projected_gravity_b
    return th.maximum(pg[:, 0].abs(), pg[:, 1].abs())


def standing(env):
    return (robot(env).root_link_pos_w[:, 2] > 0.18) & (tilt(env) < 0.3)


def nonfoot_impact(env):
    """Peak non-foot ground-contact force over the sensor history (N), per env."""
    s = env.mj.scene["nonfoot_ground_touch"]
    fh = s.data.force_history if s.data.force_history is not None else s.data.force
    return th.norm(fh, dim=-1).flatten(1).amax(1)


def slam_cap(W: float) -> float:
    """Non-foot impact a body can take when the robot carries W: the slam threshold of the weight-ladder margin."""
    return 80.0 + 1.3 * W


def in_rest_target(env):
    """Settled REST: lying low, level and still."""
    d = robot(env)
    v = th.linalg.norm(d.root_link_lin_vel_b, dim=1)
    w = th.linalg.norm(d.root_link_ang_vel_b, dim=1)
    return (d.root_link_pos_w[:, 2] < 0.15) & (tilt(env) < 0.25) & (v < 0.30) & (w < 0.50)


def in_stance_target(env):
    """A recovered stance: up, level and slow."""
    d = robot(env)
    v = th.linalg.norm(d.root_link_lin_vel_b, dim=1)
    return (d.root_link_pos_w[:, 2] > 0.20) & (tilt(env) < 0.25) & (v < 0.30)
