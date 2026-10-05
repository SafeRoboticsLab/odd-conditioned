#!/usr/bin/env bash
# Install and verify the published checkpoints into checkpoints/ (or $ODD_CHECKPOINTS).
#
#   bash scripts/fetch_weights.sh                       # download the published archive (Hugging Face)
#   bash scripts/fetch_weights.sh <archive.tar.gz | URL>
#
# Published at https://huggingface.co/buzinguyen/odd-conditioned-dev — the default URL is pinned to the upload
# commit, so it always serves exactly the files in weights/MANIFEST.sha256.
#
# The archive holds checkpoints/<name>/{model.zip, tensornormalize.pt, config.yaml} for every policy in
# odd_conditioned/policies.py, plus the two warm-start finals (rest/final, getup_stage1/final). Every file is
# checked against weights/MANIFEST.sha256 (tracked in git), so a corrupted or mismatched download fails loudly.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://huggingface.co/buzinguyen/odd-conditioned-dev/resolve/c923466bd1c68132b643f6806b9e8a2c532e4f1f/odd-conditioned-weights-v1.tar.gz"
SRC="${1:-${ODD_WEIGHTS_URL:-$URL}}"
DEST="${ODD_CHECKPOINTS:-$REPO/checkpoints}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
if [[ "$SRC" =~ ^https?:// ]]; then
  echo "[weights] downloading $SRC"
  curl -fL --progress-bar -o "$tmp/weights.tar.gz" "$SRC"
else
  cp "$SRC" "$tmp/weights.tar.gz"
fi
tar xzf "$tmp/weights.tar.gz" -C "$tmp"
echo "[weights] verifying against weights/MANIFEST.sha256"
(cd "$tmp" && sha256sum --quiet -c "$REPO/weights/MANIFEST.sha256")
mkdir -p "$DEST"
cp -r "$tmp/checkpoints/." "$DEST/"
echo "[weights] OK: $(wc -l < "$REPO/weights/MANIFEST.sha256") files -> $DEST"
