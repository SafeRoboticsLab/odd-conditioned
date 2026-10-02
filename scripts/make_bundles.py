"""MAINTAINER tool: package the trained checkpoints and the reference outputs for sharing.

Builds two archives under dist/ (git-ignored) plus checksum manifests:

  odd-conditioned-weights-<tag>.tar.gz    the 16 trained reach-avoid PPO twins every experiment from
                                          E070 on loads, at their recorded repo-relative paths
                                          (results/<family>/<run>/checkpoints/model_49999872_steps.zip
                                          + the step-matched obs-norm stats + the run's config.yaml),
                                          plus the final models of the two warm-start sources.
                                          Extract at the repo root.
  odd-conditioned-reference-<tag>.tar.gz  the reference outputs (results.json / traj npz / figures /
                                          videos / per-experiment REPORT.md / the paper-draft package)
                                          from $ODD_ARTIFACTS. Extracts to reference/.

weights/MANIFEST.sha256 (tracked in git) lists the sha256 of every file in the weights archive, so a
download can be verified file-by-file by scripts/fetch_bundles.sh.

    python scripts/make_bundles.py --tag v1
"""
import argparse
import hashlib
import os
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEP_ZIP, STEP_NORM = "model_49999872_steps.zip", "tensornorm_50001920.pt"

# Every policy the E070+ scripts load (grep `results/` in experiments/E07*-E09*).
RUNS = [
    # weight ladder (T003, E070-E075): stand / rest experts, flat + high-CoM load, unified baselines
    "results/go2_weight_runs/go2_weight_stand_adv",
    "results/go2_weight_runs/go2_weight_rest_adv",
    "results/go2_weight_runs/go2_weight_stand_hi_adv",
    "results/go2_weight_runs/go2_weight_rest_hi_adv",
    "results/go2_weight_runs/go2_weight_unified_hi_adv",
    "results/go2_weight_runs/go2_weight_unified_disc_hi_adv",
    "results/go2_weight_runs/E075_recal/go2_weight_stand_hi_adv",     # the automaton's STAND expert
    # leg family (T004, E076) and compound leg-death-while-loaded (E078)
    "results/go2_leg_family_runs/go2_leg_stand_adv",
    "results/go2_leg_family_runs/go2_leg_rest_adv",
    "results/go2_compound_runs/go2_compound_stand_adv",
    "results/go2_compound_runs/go2_compound_rest_adv",
    # certified transitions (T006, E083/E087): get-up and descent funnels, all versions
    "results/go2_transition_runs/go2_getup_adv",
    "results/go2_transition_runs/getup_v2/go2_getup_adv",              # the automaton's GET-UP funnel
    "results/go2_transition_runs/go2_descend_adv",
    "results/go2_transition_runs/descend_v3/go2_descend_adv",
    "results/go2_transition_runs/descend_v4/go2_descend_adv",          # the automaton's DESCENT funnel
]


# Warm-start SOURCES: retraining getup_v2 / descend_v4 with their recorded `--load` needs these exact
# final models + normalizers (see docs/TRAINING.md).
WARMSTART = [
    "results/go2_transition_runs/go2_getup_adv",      # -> getup_v2  (E087)
    "results/go2_weight_runs/go2_weight_rest_hi_adv",  # -> descend_v4 (E083)
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def weight_files():
    files = []
    for run in RUNS:
        for rel in (f"{run}/checkpoints/{STEP_ZIP}", f"{run}/checkpoints/{STEP_NORM}", f"{run}/config.yaml"):
            if not os.path.exists(os.path.join(REPO, rel)):
                raise SystemExit(f"missing {rel}")
            files.append(rel)
    for run in WARMSTART:
        for rel in (f"{run}/final_model.zip", f"{run}/tensornormalize.pt"):
            if not os.path.exists(os.path.join(REPO, rel)):
                raise SystemExit(f"missing {rel}")
            files.append(rel)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--artifacts", default=os.environ.get("ODD_ARTIFACTS", "~/artifacts/odd-conditioned"))
    ap.add_argument("--skip-reference", action="store_true")
    a = ap.parse_args()
    dist = os.path.join(REPO, "dist")
    os.makedirs(dist, exist_ok=True)

    files = weight_files()
    manifest = os.path.join(REPO, "weights", "MANIFEST.sha256")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    with open(manifest, "w") as m:
        for rel in files:
            m.write(f"{sha256(os.path.join(REPO, rel))}  {rel}\n")
    wt = os.path.join(dist, f"odd-conditioned-weights-{a.tag}.tar.gz")
    with tarfile.open(wt, "w:gz") as tar:
        for rel in files:
            tar.add(os.path.join(REPO, rel), arcname=rel)
    print(f"weights   -> {wt}  ({os.path.getsize(wt) / 1e6:.0f} MB, {len(files)} files)")

    if not a.skip_reference:
        art = os.path.expanduser(a.artifacts)
        rt = os.path.join(dist, f"odd-conditioned-reference-{a.tag}.tar.gz")
        with tarfile.open(rt, "w:gz") as tar:
            tar.add(art, arcname="reference")
        print(f"reference -> {rt}  ({os.path.getsize(rt) / 1e6:.0f} MB)")

    with open(os.path.join(dist, "SHA256SUMS"), "w") as s:
        for f in sorted(os.listdir(dist)):
            if f.endswith(".tar.gz"):
                s.write(f"{sha256(os.path.join(dist, f))}  {f}\n")
    print(f"manifest  -> {manifest}")


if __name__ == "__main__":
    main()
