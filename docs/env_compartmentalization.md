# Compartmentalizing odd-conditioned's Python environment

*Plan + setup, 2026-08-02. Follows the vault guide
`Multi-project workflow — shared conda base + per-project venv overlays`.*

> **STATUS 2026-08-02 (updated): DONE, and now on 0.4.0.** Both deps live under `external/`
> (safety_sb3 v0.4.0 at `external/safety-stable-baselines`; RSS 0.4.0 at
> `external/robot-safety-sandbox`). `run_exp.sh` has been **removed** — the workflow is now
> `source activate.sh` then normal `python experiments/<name>.py`. The mjlab compiled-dep gate
> below is **resolved**: go2_payload envs build + step under RSS 0.4.0 + base mjlab 1.1.1, so an
> overlay suffices (no dedicated conda env needed).

## Problem

Several projects on buzi-pc (RAAS, Digit, CLINC, SPAR, **odd-conditioned**) share the
one `mjlab` conda env and all depend on `safety_sb3` + `robot_safety_sandbox`. When any
project `pip install -e`'s its code into that shared env, its live edits leak into every
other project. We hit this repeatedly: the RAAS/dev agent's editable `safety_sb3` kept
overwriting our pin (v0.4.0 MAP rename → our `GameplaySAC` imports vanished mid-session),
which is why `run_exp.sh` exists as an ad-hoc PYTHONPATH shadow.

## Goal

Our `safety_sb3` and `robot_safety_sandbox` versions are **ours**, fixed regardless of
what other agents do to the shared base — and we leak **nothing** into base.

## The pattern (from the vault guide)

- **Shared base** (`mjlab` conda env): heavy/compiled deps only — torch, CUDA,
  mujoco-warp, **mjlab 1.1.1**, SB3-upstream — plus a pinned **non-editable** `safety_sb3
  v0.4.0` for casual consumers. We do **not** modify base. (RAAS already repinned base to
  this clean state on 2026-08-02.)
- **Our overlay** (in-project): a `.venv --system-site-packages` (inherits base's compiled
  deps) + a **PYTHONPATH prepend** of our own `safety_sb3` and RSS source trees.

**The load-bearing gotcha** (cost the guide's author the most time): a
`--system-site-packages` venv editable install **cannot** override a package that also
exists in base — the base's regular package is found by the default `PathFinder` before the
venv's PEP 660 editable finder. Since base *has* `safety_sb3`, only a **PYTHONPATH prepend**
gives our copy precedence. The `.venv` still earns its place: dist-info/metadata + an
isolated pip target. This is exactly (and only) what `run_exp.sh` was already doing by hand.

## What we point at (downstream-consumer style — our OWN copies, not the canonical checkouts)

| dep | our copy | note |
|---|---|---|
| `safety_sb3` | `DEVELOPMENT/safety-stable-baselines-v0.3.0` (our clone) | overridable via `ODD_SB3`; migration swaps to a v0.4.0 clone |
| `robot_safety_sandbox` | `external/robot-safety-sandbox` (our submodule fork) | the project's own submodule |
| `mjlab`, torch, warp | inherited from base `mjlab` env (1.1.1) | compiled — cannot be overlaid |

RAAS points its overlay at the *canonical* `safety-stable-baselines` / `safe_mjlab_zoo`
because RAAS maintains them. We are a **consumer**, so we point at our pinned clone +
submodule instead. Both repos build with `setuptools.build_meta`, so `--no-build-isolation`
installs work offline.

## Setup (this doc's companion commits)

1. `odd-conditioned/.venv` — `python -m venv --system-site-packages` off the mjlab env's
   python, with our `safety_sb3` + RSS editable-installed `--no-deps --no-build-isolation
   --force-reinstall` (metadata only; PYTHONPATH does the resolving). Already gitignored.
2. `activate.sh` — interactive entry: create the `.venv` if absent, activate it, prepend
   `PYTHONPATH`, set `MUJOCO_GL=egl`, verify + print resolutions.
3. `run_exp.sh` — non-interactive runner (what the agent uses): runs the `.venv` python with
   the same PYTHONPATH, and **hard-fails** unless `safety_sb3` resolves to our clone (with
   `GameplaySAC`) and `robot_safety_sandbox` to our submodule. Supersedes the old shadow.

## Verification (must all hold)

```
safety_sb3.__file__            -> .../safety-stable-baselines-v0.3.0/safety_sb3/...
robot_safety_sandbox.__file__  -> .../odd-conditioned/external/robot-safety-sandbox/...
torch, mjlab                    -> base mjlab env site-packages (inherited)
GameplaySAC present             -> True   (proves we did NOT get base's v0.4.0)
```
Plus: base `mjlab` env is unchanged (`pip show safety_sb3` in base still v0.4.0
non-editable), and no odd-conditioned code is installed into base.

## Relationship to the 0.4.0 migration

This is **Phase 0** of `migration_v040.md`, done first and independently. It compartmentalizes
at our *current* versions (safety_sb3 v0.3.0 + RSS v0.2.0-fork) so today's experiments/models
keep running unchanged. The migration then swaps the pinned versions *inside* the overlay:
point `ODD_SB3` at a fresh `safety-stable-baselines-v0.4.0` clone and bump the submodule — no
change to the compartmentalization mechanism.

## Open item — compiled-dep (mjlab) compatibility

Base has **mjlab 1.1.1**; our runs work on it today. RSS lists mjlab as a peer dep (no hard
pin) but the vault previously noted RSS wants **1.2.0**. If RSS **0.4.0** turns out to need
1.2.0, an overlay cannot fix a compiled dep — we'd either overlay our own editable mjlab (as
the Digit agent does) or move to a **dedicated conda env**. Gate this during the migration:
build a go2_payload env under RSS 0.4.0 + base mjlab 1.1.1; escalate only if it fails.
