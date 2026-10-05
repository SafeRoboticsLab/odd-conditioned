"""Maintainer tool: pack the installed checkpoints into the publishable archive and rewrite the manifest.

    python scripts/pack_weights.py --tag v1     # -> dist/odd-conditioned-weights-v1.tar.gz, weights/MANIFEST.sha256

The archive holds checkpoints/<name>/{model.zip, tensornormalize.pt, config.yaml} for every policy in
odd_conditioned/policies.py, plus the warm-start finals of rest and getup_stage1. Commit the manifest; publish
the archive and point users at it (scripts/fetch_weights.sh).
"""
import argparse
import hashlib
import os
import sys
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from odd_conditioned.policies import POLICIES  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def files():
    out = []
    for name in POLICIES:
        out += [f"checkpoints/{name}/{f}" for f in ("model.zip", "tensornormalize.pt", "config.yaml")]
    for name in ("rest", "getup_stage1"):
        out += [f"checkpoints/{name}/final/{f}" for f in ("model.zip", "tensornormalize.pt")]
    out.append("checkpoints/getup_stage1/config.yaml")
    missing = [f for f in out if not os.path.exists(os.path.join(REPO, f))]
    if missing:
        raise SystemExit("missing:\n  " + "\n  ".join(missing))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    fs = files()
    os.makedirs(os.path.join(REPO, "dist"), exist_ok=True)
    os.makedirs(os.path.join(REPO, "weights"), exist_ok=True)
    with open(os.path.join(REPO, "weights", "MANIFEST.sha256"), "w") as m:
        for f in fs:
            m.write(f"{sha256(os.path.join(REPO, f))}  {f}\n")
    out = os.path.join(REPO, "dist", f"odd-conditioned-weights-{a.tag}.tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        for f in fs:
            tar.add(os.path.join(REPO, f), arcname=f)
    print(f"{len(fs)} files -> {out} ({os.path.getsize(out) / 1e6:.0f} MB); manifest -> weights/MANIFEST.sha256")


if __name__ == "__main__":
    main()
