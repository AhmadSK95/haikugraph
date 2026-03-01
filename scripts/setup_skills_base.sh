#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${ROOT_DIR}/skills/base"
META_DIR="${ROOT_DIR}/skills/meta"

mkdir -p "${BASE_DIR}" "${META_DIR}"

install_skill() {
  local repo="$1"
  local skill="$2"
  echo "[skills] add ${repo}:${skill}"
  (cd "${BASE_DIR}" && npx skills add "${repo}" --skill "${skill}" --agent codex --yes --copy)
}

install_skill "vercel-labs/skills" "find-skills"
install_skill "openai/skills" "spreadsheet"
install_skill "openai/skills" "pdf"
install_skill "openai/skills" "doc"
install_skill "openai/skills" "figma-implement-design"
install_skill "404kidwiz/claude-supercode-skills" "csv-data-wrangler"
install_skill "jamesrochabrun/skills" "query-expert"
install_skill "rmyndharis/antigravity-skills" "sql-optimization-patterns"
install_skill "obra/superpowers" "dispatching-parallel-agents"
install_skill "obra/superpowers" "systematic-debugging"
install_skill "obra/superpowers" "verification-before-completion"

# Stable projection to a predictable folder.
mkdir -p "${BASE_DIR}/skills"
rm -rf "${BASE_DIR}/skills"
mkdir -p "${BASE_DIR}/skills"
if [ -d "${BASE_DIR}/.agents/skills" ]; then
  cp -R "${BASE_DIR}/.agents/skills/." "${BASE_DIR}/skills/"
fi

{
  echo "generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "base_dir=${BASE_DIR}"
  echo ""
  echo "installed_dirs:"
  find "${BASE_DIR}/skills" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort | sed 's/^/- /'
} > "${META_DIR}/skills_install_snapshot.txt"

echo "[skills] done. snapshot=${META_DIR}/skills_install_snapshot.txt"
