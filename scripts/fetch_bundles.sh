#!/usr/bin/env bash
# Fetch + verify + install the trained checkpoints (and, optionally, the reference outputs).
#
#   bash scripts/fetch_bundles.sh                 # download weights from the GitHub release, verify, extract
#   bash scripts/fetch_bundles.sh --reference     # ... plus the reference outputs -> reference/
#   bash scripts/fetch_bundles.sh --from DIR      # use archives already on disk (scp / shared drive) instead
#
# Weights land at their recorded repo-relative paths (results/<family>/<run>/checkpoints/...), which is
# exactly where the experiment scripts look. Every extracted file is checked against
# weights/MANIFEST.sha256 (tracked in git), so a corrupted or mismatched download fails loudly.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${ODD_BUNDLE_TAG:-v1}"
GH_REPO="${ODD_BUNDLE_REPO:-SafeRoboticsLab/odd-conditioned}"
RELEASE="${ODD_BUNDLE_RELEASE:-bundles-$TAG}"
FROM=""; WANT_REF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$(cd "$2" && pwd)"; shift 2 ;;
    --reference) WANT_REF=1; shift ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

WEIGHTS="odd-conditioned-weights-$TAG.tar.gz"
REF="odd-conditioned-reference-$TAG.tar.gz"
mkdir -p "$REPO/dist"
cd "$REPO"

get() {   # get <archive name> -> dist/<name>
  local name="$1"
  if [ -f "dist/$name" ]; then return 0; fi
  if [ -n "$FROM" ]; then
    cp "$FROM/$name" "dist/$name"
  elif command -v gh >/dev/null 2>&1; then
    gh release download "$RELEASE" -R "$GH_REPO" -p "$name" -D dist
  else
    curl -fL -o "dist/$name" "https://github.com/$GH_REPO/releases/download/$RELEASE/$name"
  fi
}

get "$WEIGHTS"
echo "[fetch] extracting $WEIGHTS at the repo root"
tar xzf "dist/$WEIGHTS"
echo "[fetch] verifying checkpoints against weights/MANIFEST.sha256"
sha256sum --quiet -c weights/MANIFEST.sha256
echo "[fetch] weights OK ($(wc -l < weights/MANIFEST.sha256) files)"

if [ "$WANT_REF" = 1 ]; then
  get "$REF"
  tar xzf "dist/$REF"
  echo "[fetch] reference outputs -> $REPO/reference/  (paper draft: reference/PAPER-draft/REPORT.md)"
fi
