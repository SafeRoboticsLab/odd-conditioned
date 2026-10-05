#!/usr/bin/env bash
# Install and verify the published checkpoints into checkpoints/ (or $ODD_CHECKPOINTS).
#
#   bash scripts/fetch_weights.sh <archive.tar.gz | URL>
#   bash scripts/fetch_weights.sh                       # the URL in $ODD_WEIGHTS_URL
#
# The archive holds checkpoints/<name>/{model.zip, tensornormalize.pt, config.yaml} for every policy in
# odd_conditioned/policies.py, plus the two warm-start finals (rest/final, getup_stage1/final). Every file is
# checked against weights/MANIFEST.sha256 (tracked in git), so a corrupted or mismatched download fails loudly.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-${ODD_WEIGHTS_URL:-}}"
[ -n "$SRC" ] || { sed -n 2,9p "$0"; exit 1; }
DEST="${ODD_CHECKPOINTS:-$REPO/checkpoints}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
if [[ "$SRC" =~ ^https?:// ]]; then
  echo "[weights] downloading $SRC"
  curl -fL -o "$tmp/weights.tar.gz" "$SRC"
else
  cp "$SRC" "$tmp/weights.tar.gz"
fi
tar xzf "$tmp/weights.tar.gz" -C "$tmp"
echo "[weights] verifying against weights/MANIFEST.sha256"
(cd "$tmp" && sha256sum --quiet -c "$REPO/weights/MANIFEST.sha256")
mkdir -p "$DEST"
cp -r "$tmp/checkpoints/." "$DEST/"
echo "[weights] OK: $(wc -l < "$REPO/weights/MANIFEST.sha256") files -> $DEST"
