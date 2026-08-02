"""E038 (KILL-TEST stage 2) — t_detect(B̂) vs t_detect(V) vs t_fail, with a FAIR B̂.

The premise of the B̂ direction is that a learned safety filter fails SILENTLY out-of-envelope. But B̂'s
strongest rival is the safety critic we already trained: if V collapses as early as B̂ empties, B̂ is redundant.
This measures all three clocks on the SAME trajectories:

  t_fail       first t with g < 0         (reach-avoid margin violated = actual failure)
  t_detect(V)  first t with V < 0         (the critic leaves its own certified set {V>=0})
  t_detect(B̂)  first probe with |B̂| == 0  (no admissible θ explains the data = envelope violated)

--- WHAT THE FIRST ATTEMPT GOT WRONG (both fixed here) ---
v1 reported |B̂|=0 at every probe INCLUDING in-distribution (8 kg), i.e. it flagged a payload the robot
handles perfectly. An always-empty estimator "detects" everything and is a false-positive machine, so its
apparent +1.7 s lead over V was an artefact. Two causes:

 (1) SCALE. The per-channel scale was the INSTANTANEOUS cross-candidate spread. Right after a re-init every
     candidate sits at the SAME written state, so spread ~ 0, was clamped to 1e-6, and the first-step
     residual exploded (~30 vs eps=0.22). Now pooled over the window + physical floor, and step 0 (a
     post-write transient, not dynamics) is skipped.
 (2) STRUCTURAL MISMATCH — the real one. Candidates were made by overwriting `body_mass` while INERTIA kept
     the base build's value, and by splitting mass UNIFORMLY over the 5 blocks, destroying the top-heavy
     profile. The observed trace came from a properly BUILT env. So observed and candidates differed by
     construction, not by θ: the θ-induced spread was 0.006 while the structural offset was ~0.05.
     Now a candidate is the base build rescaled: body_mass AND body_inertia are both multiplied by
     (θ_mass / base_total_mass), which is EXACT for fixed geometry with proportionally scaled density and
     PRESERVES the mass profile. Rigidity is still a direct jnt_stiffness write (exact — no inertia coupling).

 (3) EXCITATION. θ is only identifiable under excitation: at 0 N the 1.2 kg and 8.5 kg candidates were nearly
     indistinguishable at the base. E018 got δ*~0.14 s using a deliberate 50 N shake probe. Here we do NOT
     inject a probe into an already-failing robot — we run the regime that supplies excitation naturally
     (50 N adversarial pull), which is also the deployment-realistic case.

The IN-DIST SOUNDNESS CONTROL now runs automatically before the OOD verdict: at a mass INSIDE the candidate
grid, B̂ must stay NON-empty and concentrate near the true mass. If it does not, the OOD number is meaningless
and the script says so instead of reporting a lead time.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np, torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

R = "results/go2_payload_runs"; DEV = "cuda:0"; DT = 0.02
PULL = th.tensor([0., 1., 0.])

MASS = int(os.environ.get("E038_MASS", 16))          # OOD payload mass (kg) — outside the grid
CTRL_MASS = int(os.environ.get("E038_CTRL_MASS", 8))  # in-grid mass for the soundness control
FORCE = float(os.environ.get("E038_FORCE", 1.0))     # x50 N pull — supplies the excitation θ needs
ARM = os.environ.get("E038_ARM", "BLIND")
STEPS = 250
N_TRAJ = 4                                           # trajectories that get the (expensive) B̂ probe
N_STAT = 64                                          # cheap envs for t_fail / t_detect(V) statistics

RIG_RANGE, MASS_RANGE = (0.0, 320.0), (1.2, 8.5)     # the CERTIFIED envelope
N_RIG = N_MASS = 9
N_REP = 8
H = 16                                               # window (0.32 s) — E018's proven identifiability horizon
PROBE_EVERY = 10                                     # 0.20 s cadence
EPS = 0.22

ARMS = {"BLIND": (f"{R}/go2_payload_blind_gameplaysac", "blind"),
        "HISTORY": (f"{R}/go2_payload_history_gameplaysac", "history")}


def payload_idx(mj):
    m = mj.sim.mj_model
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    bnames = ["robot/payload_mount"] + [f"robot/payload_b{i}" for i in range(1, 5)]
    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in bnames]
    return (th.tensor([i for i in jids if i >= 0], device=DEV, dtype=th.long),
            th.tensor([i for i in bids if i >= 0], device=DEV, dtype=th.long))


def build_grid():
    rg = th.linspace(*RIG_RANGE, N_RIG, device=DEV); mg = th.linspace(*MASS_RANGE, N_MASS, device=DEV)
    Rg, Mg = th.meshgrid(rg, mg, indexing="ij")
    return th.stack([Rg.reshape(-1), Mg.reshape(-1)], dim=1)


def observe(d, pos0):
    return th.cat([d.projected_gravity_b, d.root_link_pos_w[:, 2:3], d.root_link_lin_vel_w,
                   d.root_link_ang_vel_w, d.root_link_pos_w[:, :2] - pos0], dim=1)


def load(run_dir, task, nenv, **kw):
    env = make_tensor(task, nenv, DEV, adversary=True, **kw)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    return env, m, nm


def critic_V(m, obs_n, act):
    qs = m.policy.critic(obs_n, act)
    return th.min(th.stack([q.squeeze(-1) for q in qs], 0), dim=0).values


def rollout(run_dir, task, nenv, record):
    env, m, nm = load(run_dir, task, nenv, end_criterion="timeout")
    dd = spec(task).dstb_dim; env.force_scale = FORCE * th.ones(nenv, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, dd).contiguous()
    robot = env.mj.scene["robot"]; obs = env.reset()
    pos0 = robot.data.root_link_pos_w[:, :2].clone()
    t_fail = th.full((nenv,), -1.0, device=DEV); t_v = th.full((nenv,), -1.0, device=DEV)
    tape = []
    for t in range(STEPS):
        obs_n = nm.normalize_obs(obs)
        with th.no_grad():
            a = th.clamp(m.policy._predict(obs_n, deterministic=True), -1, 1)
            V = critic_V(m, obs_n, th.cat([a, dstb], dim=1))
        if record:
            d = robot.data
            tape.append(dict(jp=d.joint_pos.clone(), jv=d.joint_vel.clone(),
                             root=th.cat([d.root_link_pos_w, d.root_link_quat_w,
                                          d.root_link_lin_vel_w, d.root_link_ang_vel_w], dim=1).clone(),
                             a=a.clone(), y=observe(d, pos0).clone()))
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        t_fail = th.where((g < 0) & (t_fail < 0), th.full_like(t_fail, t * DT), t_fail)
        t_v = th.where((V < 0) & (t_v < 0), th.full_like(t_v, t * DT), t_v)
    env.close()
    return t_fail, t_v, tape


class Grid:
    """Candidate bank on ONE env, built from `task` and RESCALED per candidate.

    A candidate = the base build with body_mass AND body_inertia both scaled by (θ_mass / base_total).
    Exact for fixed geometry with proportional density, and it preserves the top-heavy profile — unlike a
    uniform mass split, which silently changes the payload's shape as well as its weight.
    """

    def __init__(self, task):
        self.grid = build_grid(); self.n_theta = self.grid.shape[0]; self.n = self.n_theta * N_REP
        self.env = make_tensor(task, self.n, DEV, adversary=True, end_criterion="timeout")
        dd = spec(task).dstb_dim
        self.env.force_scale = FORCE * th.ones(self.n, device=DEV)
        self.dstb = (PULL / PULL.norm()).to(DEV)[None].expand(self.n, dd).contiguous()
        self.mj = self.env.mj; self.robot = self.mj.scene["robot"]
        self.jidx, self.bidx = payload_idx(self.mj)
        self.env.reset()
        mdl = self.mj.sim.model
        self.base_mass = mdl.body_mass[0, self.bidx].clone()            # (B,) profile-carrying
        self.base_total = float(self.base_mass.sum())
        self.has_inertia = hasattr(mdl, "body_inertia") and mdl.body_inertia is not None
        if self.has_inertia:
            self.base_inertia = mdl.body_inertia[0, self.bidx].clone()  # (B,3)
        self.stiff = self.grid[:, 0].repeat_interleave(N_REP)
        fac = (self.grid[:, 1] / self.base_total).repeat_interleave(N_REP)   # (n,)
        self.mass_rows = self.base_mass[None, :] * fac[:, None]              # (n,B)
        if self.has_inertia:
            self.inertia_rows = self.base_inertia[None] * fac[:, None, None]  # (n,B,3)

    def apply_theta(self):
        mdl = self.mj.sim.model
        mdl.jnt_stiffness[:, self.jidx] = self.stiff[:, None]
        mdl.body_mass[:, self.bidx] = self.mass_rows
        if self.has_inertia:
            mdl.body_inertia[:, self.bidx] = self.inertia_rows

    def probe(self, tape, traj, t0):
        """One windowed consistency test. Returns (|B̂|, min-residual, closest-candidate-mass)."""
        n = self.n
        jp = tape[t0]["jp"][traj][None].expand(n, -1).contiguous()
        jv = tape[t0]["jv"][traj][None].expand(n, -1).contiguous()
        rt = th.zeros(n, 13, device=DEV)
        rt[:, :2] = self.robot.data.root_link_pos_w[:, :2]   # keep each env's own world origin
        rt[:, 2:] = tape[t0]["root"][traj][2:][None]
        self.robot.write_joint_state_to_sim(jp, jv); self.robot.write_root_state_to_sim(rt)
        # write_*_to_sim is LAZY: without an explicit forward, `data` (and the state the next step rolls
        # from) still holds the PRE-write values. Verified: writing z+0.05 left data at the old z until
        # sim.forward() was called. Skipping this made every candidate roll from a stale state, producing a
        # θ-INDEPENDENT error ~3x the cross-candidate spread — which is why B̂ came out empty even in-dist.
        self.mj.sim.forward()
        self.apply_theta()
        pos0 = self.robot.data.root_link_pos_w[:, :2].clone()
        Y = []
        for k in range(H):
            self.apply_theta()
            a = tape[t0 + k]["a"][traj][None].expand(n, -1).contiguous()
            self.env.step_tensor(th.cat([a, self.dstb], dim=1))
            Y.append(observe(self.robot.data, pos0))
        Yc = th.nanmean(th.stack(Y).reshape(H, self.n_theta, N_REP, -1), dim=2)
        Yo = th.stack([tape[t0 + k]["y"][traj] for k in range(H)]).clone()
        Yo[:, -2:] = Yo[:, -2:] - tape[t0]["y"][traj][-2:]
        scale = Yc.std(dim=1).amax(0).clamp_min(1e-3)
        resid = (((Yc - Yo[:, None]) / scale).pow(2).mean(-1)).sqrt()
        worst = resid[1:].cummax(0).values[-1]
        inB = (worst <= EPS) & th.isfinite(worst)
        j = int(th.nan_to_num(worst, nan=float("inf")).argmin())
        return int(inB.sum()), float(worst.min()), float(self.grid[j, 1])

    def close(self):
        self.env.close()


def bhat_trace(gridbank, tape, traj):
    t_det, sizes, mins, closest = float("nan"), [], [], []
    for t0 in range(0, min(len(tape), STEPS) - H, PROBE_EVERY):
        k, rmin, cm = gridbank.probe(tape, traj, t0)
        sizes.append(k); mins.append(rmin); closest.append(cm)
        if k == 0 and t_det != t_det:
            t_det = t0 * DT
    return t_det, sizes, mins, closest


if __name__ == "__main__":
    run_dir, obs_key = ARMS[ARM]
    print(f"\nE038 THREE-TIMESTAMP KILL-TEST (fair B̂) — arm={ARM}, pull={FORCE*50:.0f}N, {STEPS*DT:.1f}s, no reset.")
    print(f"B̂: {N_RIG}x{N_MASS}={N_RIG*N_MASS} candidates x{N_REP} reps | window {H*DT:.2f}s every "
          f"{PROBE_EVERY*DT:.2f}s | eps={EPS} | candidates = base build rescaled (mass AND inertia)\n")

    # ---------- 1. SOUNDNESS CONTROL (in-distribution): B̂ must stay NON-empty ----------
    ctask = f"go2_payload_sweep_{obs_key}_{CTRL_MASS}"
    print(f"[1] IN-DIST SOUNDNESS CONTROL @ {CTRL_MASS}kg (inside grid {MASS_RANGE})")
    _, _, ctape = rollout(run_dir, ctask, 1, record=True)
    cg = Grid(ctask)
    print(f"    inertia rescaling available: {cg.has_inertia} | base build total payload mass "
          f"{cg.base_total:.2f}kg")
    _, csz, cmin, cclosest = bhat_trace(cg, ctape, 0)
    cg.close()
    print(f"    |B̂| trace : " + " ".join(f"{s:d}" for s in csz[:12]))
    print(f"    min-resid : " + " ".join(f"{r:.2f}" for r in cmin[:12]) + f"   (eps={EPS})")
    print(f"    closest θ mass: " + " ".join(f"{c:.1f}" for c in cclosest[:12]) + f"   (TRUE {CTRL_MASS}kg)")
    sound = sum(csz) > 0
    print(f"    => {'SOUND: B̂ retains candidates in-distribution' if sound else 'BROKEN: B̂ empty IN-DIST — the OOD number below would be meaningless'}\n")

    # ---------- 2. OOD regime: the three clocks ----------
    task = f"go2_payload_sweep_{obs_key}_{MASS}"
    print(f"[2] OOD REGIME @ {MASS}kg (grid tops at {MASS_RANGE[1]}kg)")
    tf, tv, _ = rollout(run_dir, task, N_STAT, record=False)
    fell = tf >= 0; vneg = tv >= 0
    med = lambda x, m: float(x[m].median()) if bool(m.any()) else float("nan")
    print(f"    [n={N_STAT}] t_fail {float(fell.float().mean())*100:5.1f}% median {med(tf, fell):.2f}s"
          f" | t_detect(V) {float(vneg.float().mean())*100:5.1f}% median {med(tv, vneg):.2f}s")
    both = fell & vneg
    if bool(both.any()):
        print(f"    V lead over failure (t_fail - t_V), median: {float((tf[both]-tv[both]).median()):+.2f}s")

    tf2, tv2, tape = rollout(run_dir, task, N_TRAJ, record=True)
    g = Grid(task)
    print(f"\n{'traj':>4} | {'t_fail':>7} | {'t_det(V)':>8} | {'t_det(B̂)':>9} | {'B̂-V':>7} | {'B̂-fail':>7}")
    rows = []
    for i in range(N_TRAJ):
        tdb, sizes, mins, _ = bhat_trace(g, tape, i)
        a = float(tf2[i]); b = float(tv2[i])
        a = a if a >= 0 else float("nan"); b = b if b >= 0 else float("nan")
        rows.append((a, b, tdb))
        print(f"{i:>4} | {a:7.2f} | {b:8.2f} | {tdb:9.2f} | {b-tdb:+7.2f} | {a-tdb:+7.2f}")
        print(f"       |B̂| " + " ".join(f"{s:d}" for s in sizes[:12])
              + "  | min-resid " + " ".join(f"{r:.2f}" for r in mins[:12]))
    g.close()
    arr = np.array(rows, dtype=float)
    with np.errstate(invalid="ignore"):
        print(f"\nmedian: t_fail {np.nanmedian(arr[:,0]):.2f}s | t_det(V) {np.nanmedian(arr[:,1]):.2f}s | "
              f"t_det(B̂) {np.nanmedian(arr[:,2]):.2f}s")
    if not sound:
        print("\n!! IN-DIST CONTROL FAILED — B̂ flags everything, so no lead-time claim can be made. !!")
    else:
        print("\nVERDICT: B̂ lead over V > 0 => B̂ sees the envelope violation before the critic doubts itself.")
        print("         <= 0 => the already-trained critic detects it as early; B̂ is REDUNDANT here.")
