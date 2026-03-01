#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[security] running hygiene checks"

# 1) Ensure .env is not tracked.
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "[security] FAIL: .env is tracked in git"
  exit 1
fi

# 2) Ensure no legacy wrong scaffold remains.
if [ -d "${ROOT_DIR}/agent_skills" ]; then
  echo "[security] FAIL: legacy agent_skills directory still exists"
  exit 1
fi

# 3) Top-level markdown hygiene (single source docs only).
ALLOWED_TOP_LEVEL_DOCS_REGEX='^(README\.md|PRODUCT_GAP_TRACKER\.md|TECHNICAL_BUILD_TRACKER\.md)$'
if find "${ROOT_DIR}" -maxdepth 1 -type f -name "*.md" -print \
  | sed "s#${ROOT_DIR}/##" \
  | rg -v "$ALLOWED_TOP_LEVEL_DOCS_REGEX" >/tmp/datada_doc_hygiene_scan.txt; then
  echo "[security] FAIL: stale top-level markdown docs detected"
  cat /tmp/datada_doc_hygiene_scan.txt
  exit 1
fi

# 4) Secret pattern scan on tracked files only.
TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

git ls-files > "$TMP_FILE"

SECRET_PATTERNS='(sk-ant-api|sk-proj-|OPENAI_API_KEY\s*=|ANTHROPIC_API_KEY\s*=|HG_OPENAI_API_KEY\s*=|HG_ANTHROPIC_API_KEY\s*=|-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----)'

if xargs -a "$TMP_FILE" rg -n --pcre2 "$SECRET_PATTERNS" --color never >/tmp/datada_secret_scan.txt 2>/dev/null; then
  echo "[security] FAIL: potential secret/key material found"
  cat /tmp/datada_secret_scan.txt
  exit 1
fi

# 5) Key/cert file extension guard in tracked files.
if git ls-files | rg -n '\.(pem|p12|key)$' >/tmp/datada_keyfile_scan.txt; then
  echo "[security] FAIL: tracked key/cert file detected"
  cat /tmp/datada_keyfile_scan.txt
  exit 1
fi

echo "[security] PASS"
