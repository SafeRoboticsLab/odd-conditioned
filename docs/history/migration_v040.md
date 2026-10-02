> **Historical (2026-08-02).** The migration this plan describes was carried out; the repo now runs on safety_sb3 / robot-safety-sandbox 0.4.0. Kept for the reasoning behind the soundness reset.

# Migration to safety_sb3 0.4.0 + robot-safety-sandbox 0.4.0

*Analysis + plan, 2026-08-02. Status: PROPOSED (nothing migrated yet).*

## TL;DR

Two 0.4.0 releases land together. The API change is large but **mechanical and fully
documented** (rename table + sed script upstream). The consequential change is a
**certificate-soundness fix**: the SAC reach-avoid critic target wrongly included the
max-entropy bonus, which biases the value and breaks the `{V ≥ 0}` certificate —
measured precision `P(reach-avoid | V̂ ≥ 0)` **0.05 buggy → 0.98 fixed** against an HJ
oracle. **Every model we have is a reach-avoid SAC twin trained on the buggy value
function.** So this is not a code chore — it forces a **retrain of all four models**,
and it means **every value-based empirical result (E037/E038 and the V-as-detector
finding) must be re-measured**. B̂ (E018/E038 belief arm) is RL-free and unaffected.

## 1. What changed, and why it matters to us

### 1a. safety_sb3 0.3.0 → 0.4.0 — "the MAP rename" + soundness fix

Source: `safety-stable-baselines/RELEASE_NOTES.md` (the maintainers wrote it as a
migration guide). Local editable is already at `v0.4.0-1` (`origin/main`, feat/refactor).

**The rename (MAP = Mode · Algorithm · Players).** No shims, no aliases — old imports
raise `ImportError` by design. The classes we use:

| we use (v0.3.x) | 0.4.0 | what it is |
|---|---|---|
| `GameplaySAC` | **`ReachAvoidSAC2P`** | two-player reach-avoid SAC (blind/history/conditioned) |
| `ReachAvoidSAC` | **`ReachAvoidSAC1P`** | single-player reach-avoid SAC (descent) |
| `IsaacsPolicy` | **`TwoPlayerSACPolicy`** (now in `safety_sb3.policies`) | — |
| `TensorVecNormalize` | unchanged, still `safety_sb3.tensor_env` | ✅ our import survives |

Module files renamed too (`isaacs.py`→`sac_2p.py`, `reach_avoid_sac.py`→`sac_1p.py`,
`isaacs_policy.py`→`policies.py`, `isaacs_buffers.py`→`buffers_replay.py`, …). Also:
two-player log keys `isaacs/* → game/*`; default league dirs `isaacs_leaderboard →
sac_2p_leaderboard`.

**§7 The soundness fix (the reason this release is "important").** The SAC critic
target subtracted the entropy bonus inside the Bellman backup:

```
v0.3.x:  V' = min(Q1',Q2') − α_ctrl·logπ_ctrl(s') + α_dstb·logπ_dstb(s')   # WRONG for HJ
v0.4.0:  V' = min(Q1',Q2')                                                 # pure HJ
```

That entropy term is correct for *cumulative* RL (`r + γV'`) but not for a `min/max`
of margins — it biases the certificate. Confirmed the diff lands in `sac_2p.py`
(our two-player RA-SAC) **and** `sac_1p.py` (descent). **Retrain to benefit; a value
function trained on ≤ v0.3.x is unsound as a certificate.**

**Also:** checkpoints store the class path, so **v0.3.x `.zip` models will not load on
0.4.0** — independent of the soundness issue, we cannot even eval the old models under
0.4.0 without pinning v0.3.x. Combined with §7, the answer is: retrain.

**Not us:** the v0.2.0 reach-avoid *anchor* bug (`g` vs `min(l,g)`) hit only the **PPO**
family; the SAC family was already correct. We use SAC — so our models have the
entropy bug (fixed in 0.4.0) but never had the anchor bug. Also, a `min_alpha`/`max_alpha`
clamp fix in `ReachAvoidSAC1P` (descent) — only matters if we set non-default alpha bounds
(we don't currently).

### 1b. robot-safety-sandbox → 0.4.0

Our RSS is a **submodule** on branch `project/odd-conditioned-go2-payload`, based on
**v0.2.0 + ~23 commits** (it carries our go2_payload env/tasks and a merged v0.3.0
family-routed trainer). Upstream `origin/main` is now **v0.4.0-1**. The 0.4.0 tag/objects
are available locally in `../spar-migration/robot-safety-sandbox` (no GitHub fetch needed;
SSH auth is non-interactive-broken on this box anyway).

Upstream changes that touch our code:

- **`algo_name` is now the MAP formula** (081a690). `algo_name("go2_stabilize",
  adversary=True, family="off_policy") → "ReachAvoidSAC2P"` directly. *Lucky break:* our
  experiments call `getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO","SAC"))`
  — under 0.4.0 that yields `"ReachAvoidPPO2P".replace("PPO","SAC") = "ReachAvoidSAC2P"`,
  which **exists**. So the loader line survives verbatim, though we should clean it to
  `algo_name(task, adversary=True, family="off_policy")` and drop the `.replace`.
- **`TaskSpec`: `kind`/`default_algo` → `mode`** (64e1151). Required field, one of
  `safety` / `reach-avoid` / `cumulative`. Our two registrations use
  `default_algo="ReachAvoidPPO"` → both become **`mode="reach-avoid"`**. `warmstart_from`
  is gone from TaskSpec.
- **`make_tensor`, `spec`, `register`, `TaskSpec` still exported** from
  `robot_safety_sandbox` ✅.
- **Assets consolidated under `envs/assets/`** (f268263) — check our
  `assets_go2_payload/go2_payload_constants.py` base-asset path still resolves.
- **Filters decomposed** into fallback / monitor / intervention (d18eecb) + composition
  API. Only matters when we actually wire up ValueShield/QCBF (not yet).
- **registry.py and base.py were rewritten upstream** for MAP — and **we modified both**
  (+15 / +32 lines). These are the real merge-conflict sites.

## 2. Blast radius in our project

| surface | impact |
|---|---|
| `experiments/*.py` loader line (`algo_name(...).replace`) | survives by luck; clean up recommended |
| `experiments/*.py` model loads (`final_model.zip`, `ctrl_*.pt`) | **broken** — won't load; need retrained 0.4.0 models |
| `TensorVecNormalize.load` | ✅ unchanged |
| `run_exp.sh` (pins v0.3.0 clone) | repoint to 0.4.0, or retire in favor of a dedicated env |
| our RSS fork `tasks/go2_payload_stabilize.py` | `default_algo=` → `mode=`; rebase onto 0.4.0 registry |
| our RSS fork `envs/go2_payload_stabilize/env_cfg.py`, `assets_go2_payload/` | rebase onto 0.4.0 env framework; verify asset paths |
| our RSS fork edits to `registry.py`, `base.py` | **hand-reapply** — both rewritten upstream |
| `configs/go2_payload_*.yaml` | verify the 0.4.0 trainer's config schema is unchanged |
| **4 trained models** (blind, history, conditioned = RA-SAC2P; descent = RA-SAC1P) | **RETRAIN on 0.4.0** |
| **E037 / E038 / V-as-detector** results | **re-measure** — they read an unsound value function |
| **E018 / E038 B̂ arm** | ✅ RL-free, unaffected — stands |
| bifurcation (E014), dynamic-ODD (E021–E028) | policies from a biased critic — **re-confirm** after retrain |

## 3. Migration plan (phased; nothing here is executed yet)

### Phase 0 — infra: a dedicated, pinned env (ends the shadowing dance)
The shared `mjlab` conda env is owned by the other agent and flips between versions;
`run_exp.sh` shadows it. Cleanest fix now that we *want* 0.4.0 (which the shared env
already has): give the project its **own** env with both deps pinned to the 0.4.0 tags.
- Clone `safety-stable-baselines` @ `v0.4.0` (non-editable) and RSS submodule @ 0.4.0.
- New conda env `odd-v040`, `pip install --no-deps` both + the pinned mjlab stack.
- Retire `run_exp.sh` (or repoint it at the 0.4.0 clone) once the env is clean.

### Phase 1 — RSS fork onto 0.4.0 (the merge)
Recommend **port, not rebase** — our work is mostly *additive* files; a 23-commit rebase
through a registry rewrite is worse.
1. New branch off RSS `v0.4.0`: `project/odd-conditioned-go2-payload-v040`.
2. Copy our additive files unchanged: `envs/assets_go2_payload/`,
   `envs/go2_payload_stabilize/`, `tasks/go2_payload_stabilize.py`,
   `configs/go2_payload_*.yaml`.
3. Re-apply the small `registry.py` / `base.py` / `__init__.py` edits **by hand** against
   0.4.0 (diff our v0.2.0-based edits, port the intent).
4. Fix break sites: `default_algo="ReachAvoidPPO"` → `mode="reach-avoid"` (both
   registrations); drop `warmstart_from`; verify asset paths against `envs/assets/`.
5. `import robot_safety_sandbox; make_tensor("go2_payload_blind", 4)` smoke test until
   every go2_payload task registers and builds.
6. Point the submodule at the new branch; bump the RSS pyproject safety_sb3 pin to
   `@v0.4.0`.

### Phase 2 — our experiment code
1. Loader line: `algo_name(task, adversary=True, family="off_policy")`, drop `.replace`.
   (Survives either way, but make it honest.)
2. Any `IsaacsPolicy` / `isaacs/*` log-key / leaderboard-dir references → 0.4.0 names.
3. Do **not** touch the E018 B̂ estimator logic — RL-free, correct as-is.

### Phase 3 — retrain the four models on 0.4.0 (the real cost)
Local (RTX 4070, 12 GB) per the 2026-07-31 decision. The 50M history recipe was sized
for a 24 GB card — cut `num_envs` / `buffer_size` and time a short run first.
- `go2_payload_{blind,conditioned,history}` (ReachAvoidSAC2P), `go2_payload_descent`
  (ReachAvoidSAC1P).
- **Fix checkpoint cadence this time** (E036 lesson): save full `.zip` every ~2 M and keep
  best-on-a-**dynamic** validation score, never on in-dist adversarial `safe_rate`.
- Register each run in `experiments.md` before launch.

### Phase 4 — re-measure the results that read V
Re-run E037 (break sweep) and E038 (three-timestamp) on the retrained, now-sound critic.
Two things could move:
- **t_detect(V) should become meaningful** — V ≥ 0 now actually certifies reach-avoid
  (0.98 vs 0.05), so the V-as-detector baseline is finally reading a real certificate.
  Its lead time / false-positive rate may change substantially.
- The **B̂-vs-V comparison** — the whole point of the kill-test — must be re-run; the old
  numbers were against a biased V and are not citable.
Then re-confirm the qualitative bifurcation (E014) and dynamic-ODD (E021–E028) results.

## 4. Risks & open decisions

- **Cost.** Phase 3 is 4 retrains on a 12 GB card sharing the GPU with another agent.
  This is the dominant effort. Everything else is ≤ a day.
- **Results churn.** Be honest that most value-based conclusions revert to "unverified"
  until Phase 4 completes. The bright side: the V-as-detector result was *itself* on
  unsound footing, and the fixed critic may make it (or B̂) genuinely stronger. B̂'s
  standing is unchanged — it never depended on the learned value.
- **Decision — env strategy (Phase 0):** dedicated `odd-v040` env (recommended) vs. keep
  shadowing the shared env at 0.4.0. Dedicated ends a recurring failure mode.
- **Decision — retrain scope:** all four now, or just blind + history (the arms the
  detection story uses) and defer conditioned/descent. Recommend blind + history first,
  measure, then decide on the rest.
- **Left alone:** the stale 55-dim `heavy_sloshy` / `light_rigid` specialists — retrain
  only if E014 needs refreshing.
