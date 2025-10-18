#!/usr/bin/env bash

# Utility script to package the current NLP workspace into a fresh Git repository.
# The resulting repository can then be pushed to GitHub/GitLab and cloned elsewhere.

set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPORT_ROOT="${1:-${WORKSPACE_ROOT}/build/nlp_repo}"

if [[ -d "${EXPORT_ROOT}" ]]; then
  echo "[export-nlp-repo] Removing existing export directory: ${EXPORT_ROOT}"
  rm -rf "${EXPORT_ROOT}"
fi

echo "[export-nlp-repo] Copying workspace (excluding Git metadata)…"
# Rsync keeps permissions and skips .git directories if any exist.
rsync -a \
  --exclude='.git' \
  --exclude='**/__pycache__' \
  "${WORKSPACE_ROOT}/" "${EXPORT_ROOT}/"

cd "${EXPORT_ROOT}"

echo "[export-nlp-repo] Initialising standalone Git repository…"
git init >/dev/null
git add . >/dev/null
git commit -m "Initial import of NLP workspace" >/dev/null

cat <<'EOF'

[export-nlp-repo] Done!

The workspace snapshot now lives in:
  ${EXPORT_ROOT}

Next steps to publish it:
  1. Create an empty remote repository (e.g. on GitHub).
  2. cd ${EXPORT_ROOT}
  3. git remote add origin <REMOTE_URL>
  4. git push -u origin master   # or main, depending on your remote

Teammates can then clone the remote to replicate this NLP environment.
EOF
