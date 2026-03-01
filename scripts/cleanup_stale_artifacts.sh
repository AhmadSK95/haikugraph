#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Remove wrong scaffolding from previous iteration.
rm -rf "${ROOT_DIR}/agent_skills"
rm -f "${ROOT_DIR}/scripts/setup_skill_base.sh"

# Ensure required skill workspace exists.
mkdir -p "${ROOT_DIR}/skills/base" "${ROOT_DIR}/skills/meta" "${ROOT_DIR}/skills/reports"

echo "cleanup complete"
