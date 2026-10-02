"""E071 (T003, G3) — WEIGHT-LADDER HANDOFF FILTER + ramp demo + trigger-delay sweep.

A growing carried load W(t) makes the STAND spec go infeasible. This builds the CERTIFIED HANDOFF: run the
STAND twin while its own reach-avoid value V_stand(x,W) still certifies standing, and hand off PERMANENTLY to
the REST twin (soft belly rest) once V_stand drops below eps. The demo pits it against the two fixed-ODD
filters it should dominate — fixed-normal (STAND-only, overconfident) and fixed-worst (REST-only, no
affordance) — plus an ORACLE handoff (switch at the envelope-known W*).

Ramp: W = 0 for t<100, then linear 0->300 N over 300 steps, then hold 300. Both the physics wrench
(``env.base_load``) and the obs conditioning (``env.mj._weight_W``, read by the ``weight_theta`` obs term)
are driven per step (base_load takes precedence for the physics; _weight_W feeds the policy/critic obs).

Value readout (Task 1): see experiments/_value_util.stand_value. V_stand = STAND twin's state-only PPO value
net on the normalized 48-dim actor obs; V>=0 == standing still certifiable. Validated to contract
monotonically with W (0.044@W0 -> -0.012@W250 under the stand policy at pull 0.2).

Usage:  python experiments/E071_handoff.py            # runs both pulls (0.2, 0.5) + trigger sweep @0.2
        python experiments/E071_handoff.py --pull 0.2 # single pull
Outputs: ~/artifacts/odd-conditioned/E071-handoff/{results.json, README.txt}
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import torch as th

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_safety_sandbox import make_tensor, spec  # noqa: E402
from robot_safety_sandbox.eval.policies import load_twin  # noqa: E402
from _value_util import stand_value  # noqa: E402

DEV = "cuda:0"
N = 128
STEPS = 500
RAMP_START, RAMP_LEN, W_MAX = 100, 300, 300.0        # W: 0 until 100, linear to 300 over 300, then hold
PULL = th.tensor([0.0, 1.0, 0.0])                    # lateral (+y) ambient pull direction
EPS = 0.015                                          # handoff trigger threshold on V_stand (see calibration)
HYST = 5                                             # consecutive V<eps steps before permanent switch
W_ORACLE = 180.0                                     # envelope-known W* for the oracle handoff
CK = "results/go2_weight_runs/go2_weight_{m}_adv/checkpoints/model_49999872_steps.zip"
OUT = os.path.expanduser("~/artifacts/odd-conditioned/E071-handoff")

# Slam cap (rest-spec safety): nonfoot ground force must stay under 80 + 1.3*W(t).
CAP_BASE, CAP_SLOPE = 80.0, 1.3


def W_of(t: int) -> float:
  if t < RAMP_START:
    return 0.0
  if t < RAMP_START + RAMP_LEN:
    return (t - RAMP_START) / RAMP_LEN * W_MAX
  return W_MAX


def _load_models():
  with contextlib.redirect_stdout(io.StringIO()):
    ms, ns = load_twin(CK.format(m="stand"), DEV, quiet=True)
    mr, nr = load_twin(CK.format(m="rest"), DEV, quiet=True)
  ms.policy.set_training_mode(False)
  mr.policy.set_training_mode(False)
  return ms, ns, mr, nr


def rollout(models, pull_scale, switch_mode, W_switch=None, eps=EPS):
  """One ramp rollout of N envs on the permissive REST cfg, driving W(t) each step.

  switch_mode: "V"  -> V_stand<eps for HYST consecutive steps -> permanent switch to REST (handoff).
               "W"  -> switch permanently once W(t) >= W_switch (forced/oracle; W_switch=inf == STAND-only,
                       W_switch<=0 == REST-only from t=0).
  Returns a metrics dict.
  """
  ms, ns, mr, nr = models
  with contextlib.redirect_stdout(io.StringIO()):
    env = make_tensor("go2_weight_rest_at_0", N, DEV, adversary=True)
  inner = env.mj
  env.force_scale = pull_scale * th.ones(N, device=DEV)
  dstb = PULL.to(DEV)[None].expand(N, spec("go2_weight_rest_at_0").dstb_dim).contiguous()
  obs = env.reset()

  switched = th.zeros(N, dtype=th.bool, device=DEV)
  below = th.zeros(N, device=DEV)
  switch_step = th.full((N,), -1, dtype=th.long, device=DEV)
  switch_W = th.full((N,), -1.0, device=DEV)
  standing_t = th.zeros(N, device=DEV)
  standing_pre = th.zeros(N, device=DEV)               # standing steps BEFORE switch
  tipped = th.zeros(N, dtype=th.bool, device=DEV)
  slam = th.zeros(N, dtype=th.bool, device=DEV)
  peak_force_post = th.zeros(N, device=DEV)            # peak nonfoot force after switch (gentleness)
  max_vz_post = th.zeros(N, device=DEV)               # max |v_z| after switch (descent speed)
  V_trace = []                                         # mean V_stand per step (diagnostic)

  for t in range(STEPS):
    W = W_of(t)
    env.base_load = th.tensor([0.0, 0.0, -W], device=DEV)[None].expand(N, 3).contiguous()
    inner._weight_W = th.full((N,), W, device=DEV)

    V = stand_value(env, ms, ns)                       # V_stand at current state
    V_trace.append(float(V.mean()))

    if switch_mode == "V":
      below = th.where(V < eps, below + 1.0, th.zeros_like(below))
      new = (below >= HYST) & ~switched
    else:  # "W"
      thr = float("inf") if W_switch is None else W_switch
      new = (th.full((N,), W, device=DEV) >= thr) & ~switched
    if new.any():
      switch_step[new] = t
      switch_W[new] = W
    switched = switched | new

    with th.no_grad():
      a_stand = th.clamp(ms.policy._predict(ns(obs), deterministic=True), -1, 1)
      a_rest = th.clamp(mr.policy._predict(nr(obs), deterministic=True), -1, 1)
    a = th.where(switched.unsqueeze(-1), a_rest, a_stand)
    obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))

    d = inner.scene["robot"].data
    pg = d.projected_gravity_b
    tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
    is_stand = (d.root_link_pos_w[:, 2] > 0.18) & (tilt < 0.3)
    standing_t += is_stand.float()
    standing_pre += (is_stand & ~switched).float()
    s = inner.scene["nonfoot_ground_touch"]
    fh = s.data.force_history if s.data.force_history is not None else s.data.force
    force = th.norm(fh, dim=-1).flatten(1).amax(1)
    slam |= force > (CAP_BASE + CAP_SLOPE * W)
    tipped |= inner.termination_manager._term_dones.get(
      "fell_over", th.zeros(N, device=DEV)).bool()
    vz = d.root_link_lin_vel_w[:, 2].abs()
    post = switched
    peak_force_post = th.where(post, th.maximum(peak_force_post, force), peak_force_post)
    max_vz_post = th.where(post, th.maximum(max_vz_post, vz), max_vz_post)

  h_end = inner.scene["robot"].data.root_link_pos_w[:, 2]
  did_switch = switched
  env.close()

  # window for gentleness when there was NO switch (STAND-only): use the ramp tail t>=RAMP_START.
  res = dict(
    stand_frac=float(standing_t.mean()) / STEPS,
    stand_frac_pre=float(standing_pre.mean()) / STEPS,
    h_end=float(h_end.mean()),
    tip=float(tipped.float().mean()),
    slam=float(slam.float().mean()),
    switched_frac=float(did_switch.float().mean()),
    peak_force_post=float(peak_force_post[did_switch].mean()) if did_switch.any() else float("nan"),
    max_vz_post=float(max_vz_post[did_switch].mean()) if did_switch.any() else float("nan"),
    V_final=V_trace[-1],
  )
  if did_switch.any():
    sw = switch_W[did_switch]
    res.update(switchW_mean=float(sw.mean()), switchW_std=float(sw.std()),
               switchW_min=float(sw.min()), switchW_max=float(sw.max()),
               switchT_mean=float(switch_step[did_switch].float().mean()))
  return res


# ── Task 1: V_stand vs W discrimination (the empirical certificate contraction) ──────────────────
def value_vs_W(models, pull=0.2, Ws=(0, 90, 150, 200, 250), warm=150):
  """Under the STAND policy at fixed W, report mean/std V_stand + stand% after `warm` steps."""
  ms, ns, mr, nr = models
  rows = {}
  for W in Ws:
    task = f"go2_weight_rest_at_{W}"
    with contextlib.redirect_stdout(io.StringIO()):
      env = make_tensor(task, N, DEV, adversary=True)
    inner = env.mj
    env.force_scale = pull * th.ones(N, device=DEV)
    dstb = PULL.to(DEV)[None].expand(N, spec(task).dstb_dim).contiguous()
    obs = env.reset()
    st = th.zeros(N, device=DEV)
    for _ in range(warm):
      with th.no_grad():
        a = th.clamp(ms.policy._predict(ns(obs), deterministic=True), -1, 1)
      obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
      d = inner.scene["robot"].data
      pg = d.projected_gravity_b
      tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
      st += ((d.root_link_pos_w[:, 2] > 0.18) & (tilt < 0.3)).float()
    V = stand_value(env, ms, ns)
    rows[W] = dict(V_mean=float(V.mean()), V_std=float(V.std()), stand_frac=float(st.mean()) / warm)
    env.close()
  return rows


def fmt_vW_table(rows):
  lines = [f"{'W(N)':>5} {'V_mean':>8} {'V_std':>7} {'stand%':>7}"]
  for W, r in rows.items():
    lines.append(f"{W:>5} {r['V_mean']:>8.4f} {r['V_std']:>7.4f} {r['stand_frac']:>7.2f}")
  return "\n".join(lines)


# ── Task 2: 4-arm ramp table ─────────────────────────────────────────────────────────────────────
def four_arm(models, pull):
  return {
    "stand_only": rollout(models, pull, "W", W_switch=float("inf")),
    "rest_only": rollout(models, pull, "W", W_switch=0.0),
    "handoff": rollout(models, pull, "V", eps=EPS),
    "oracle": rollout(models, pull, "W", W_switch=W_ORACLE),
  }


# ── Task 3: trigger-delay sweep (forced W-switches) — the point-of-no-return data ─────────────────
SWEEP_W = [120, 150, 180, 210, 240, 270, 300, "never"]


def trigger_sweep(models, pull):
  out = {}
  for w in SWEEP_W:
    ws = float("inf") if w == "never" else float(w)
    out[str(w)] = rollout(models, pull, "W", W_switch=ws)
  return out


def fmt_arm_table(arms):
  lines = [f"{'arm':>11} {'afford':>7} {'pre':>6} {'h_end':>6} {'tip':>5} {'slam':>5} "
           f"{'swW_mu':>7} {'swW_sd':>7} {'pkF':>6} {'vz':>6}"]
  for name, r in arms.items():
    lines.append(
      f"{name:>11} {r['stand_frac']:>7.2f} {r['stand_frac_pre']:>6.2f} {r['h_end']:>6.2f} "
      f"{r['tip']:>5.2f} {r['slam']:>5.2f} "
      f"{r.get('switchW_mean', float('nan')):>7.1f} {r.get('switchW_std', float('nan')):>7.1f} "
      f"{r['peak_force_post']:>6.1f} {r['max_vz_post']:>6.2f}")
  return "\n".join(lines)


def fmt_sweep_table(sweep):
  lines = [f"{'switchW':>8} {'afford':>7} {'tip':>5} {'slam':>5} {'pkF_post':>9} {'vz_post':>8} {'h_end':>6}"]
  for w, r in sweep.items():
    lines.append(
      f"{w:>8} {r['stand_frac']:>7.2f} {r['tip']:>5.2f} {r['slam']:>5.2f} "
      f"{r['peak_force_post']:>9.1f} {r['max_vz_post']:>8.2f} {r['h_end']:>6.2f}")
  return "\n".join(lines)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--pull", type=float, default=None, help="single pull force_scale (else runs 0.2 and 0.5)")
  ap.add_argument("--sweep_pull", type=float, default=0.2, help="pull for the trigger-delay sweep")
  args = ap.parse_args()
  os.makedirs(OUT, exist_ok=True)

  models = _load_models()
  pulls = [args.pull] if args.pull is not None else [0.2, 0.5]

  results = {"config": dict(N=N, STEPS=STEPS, ramp=(RAMP_START, RAMP_LEN, W_MAX),
                            eps=EPS, hyst=HYST, W_oracle=W_ORACLE, cap=(CAP_BASE, CAP_SLOPE)),
             "value_vs_W": {}, "four_arm": {}, "trigger_sweep": {}}
  txt = ["E071 WEIGHT-LADDER HANDOFF FILTER — ramp demo + trigger-delay sweep",
         f"N={N} steps={STEPS} ramp: W=0 until t={RAMP_START}, linear 0->{W_MAX:.0f}N over {RAMP_LEN}, then hold",
         f"handoff trigger: V_stand < eps={EPS} for {HYST} consecutive steps -> permanent switch to REST",
         f"oracle switch W*={W_ORACLE:.0f}N.  slam cap = {CAP_BASE:.0f}+{CAP_SLOPE}*W(t)",
         "cols: afford=standing frac (full ep); pre=standing frac before switch; pkF=peak nonfoot force post-switch(N);"
         " vz=max |v_z| post-switch(m/s)", ""]

  # Task 1: V_stand vs W (the empirical certificate contraction; V>=0 == standing still certifiable)
  vW = value_vs_W(models, pull=0.2)
  results["value_vs_W"]["pull_0.2"] = vW
  txt.append("=== TASK 1: V_stand vs W under STAND policy @ pull 10N (after 150 steps) ===")
  txt.append(fmt_vW_table(vW))
  txt.append("")
  print(txt[-3]); print(txt[-2])

  for pull in pulls:
    arms = four_arm(models, pull)
    results["four_arm"][f"pull_{pull}"] = arms
    txt.append(f"=== 4-ARM RAMP @ pull {pull*50:.0f}N (force_scale {pull}) ===")
    txt.append(fmt_arm_table(arms))
    txt.append("")
    print(txt[-3]); print(txt[-2])

  sweep = trigger_sweep(models, args.sweep_pull)
  results["trigger_sweep"][f"pull_{args.sweep_pull}"] = sweep
  txt.append(f"=== TRIGGER-DELAY SWEEP @ pull {args.sweep_pull*50:.0f}N (forced W-switch) ===")
  txt.append(fmt_sweep_table(sweep))
  txt.append("")
  print(txt[-3]); print(txt[-2])

  with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
  with open(os.path.join(OUT, "README.txt"), "w") as f:
    f.write("\n".join(txt))
  print(f"\nwrote {OUT}/results.json + README.txt")
