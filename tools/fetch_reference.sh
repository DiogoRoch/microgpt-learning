#!/usr/bin/env bash
# Fetch the source of truth: Karpathy's microgpt.py, pinned to a specific gist revision.
# The gist has no license, so we fetch at build/dev time and attribute rather than redistribute.
#
# Pinned revision: everything in golden/ was generated from this exact revision.
# If you bump GIST_REV you MUST regenerate golden/ (python3 tools/golden.py) and
# re-run the parity suite (npm test) before committing anything else.
set -euo pipefail

GIST_ID="8627fe009c40f57531cb18360106ce95"
GIST_REV="14fb038816c7aae0bb9342c2dbf1a51dd134a5ff"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/reference/microgpt.py"
mkdir -p "$ROOT/reference"

if [ -f "$DEST" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "reference/microgpt.py already present (FORCE=1 to refetch)"
  exit 0
fi

# Try the raw URL first; some egress policies block gist.githubusercontent.com,
# so fall back to cloning the gist as a git repo (gist.github.com) at the pinned rev.
if curl -fsSL "https://gist.githubusercontent.com/karpathy/${GIST_ID}/raw/${GIST_REV}/microgpt.py" -o "$DEST" 2>/dev/null; then
  echo "fetched via raw URL"
else
  echo "raw URL unreachable; falling back to git clone of the gist"
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  git clone --quiet "https://gist.github.com/${GIST_ID}.git" "$TMP/gist"
  git -C "$TMP/gist" checkout --quiet "$GIST_REV"
  cp "$TMP/gist/microgpt.py" "$DEST"
  echo "fetched via git clone @ ${GIST_REV}"
fi

echo "OK: $DEST ($(wc -l < "$DEST") lines)"
