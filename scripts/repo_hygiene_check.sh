#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[hygiene] scanning repository for duplicate/stale Python modules"

problem=0

# 1) Catch stale duplicate naming patterns like \"foo 2.py\"
stale_dupes="$(find src -type f -name '* [0-9].py' | sort || true)"
if [[ -n "$stale_dupes" ]]; then
  echo "[hygiene] FAIL: stale duplicate file names detected:"
  echo "$stale_dupes"
  problem=1
fi

# 2) Catch duplicate module stems within a package tree (excluding __init__).
dup_stems="$(
  find src -type f -name '*.py' \
    | sed 's#^src/##' \
    | awk -F/ '
        {
          file=$NF
          gsub(/\.py$/, "", file)
          if (file == "__init__") next
          key=$(NF-1)"/"file
          count[key]++
          paths[key]=(paths[key] ? paths[key] "\n  - " : "  - ") $0
        }
        END {
          for (k in count) {
            if (count[k] > 1) {
              print k
              print paths[k]
            }
          }
        }
      ' || true
)"
if [[ -n "$dup_stems" ]]; then
  echo "[hygiene] FAIL: duplicate module stems detected in package subtrees:"
  echo "$dup_stems"
  problem=1
fi

if [[ "$problem" -ne 0 ]]; then
  exit 1
fi

echo "[hygiene] PASS"
