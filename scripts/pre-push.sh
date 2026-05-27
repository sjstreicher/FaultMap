#!/usr/bin/env bash
# Runs the CI checks locally before pushing to main.
# Mirrors .github/workflows/ci.yml so a green local run = green CI.
#
# Install with:
#   ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push
#
# Bypass once with `git push --no-verify`.

set -euo pipefail

protected_branch="refs/heads/main"
run_checks=0

while read -r local_ref local_sha remote_ref remote_sha; do
    if [[ "$remote_ref" == "$protected_branch" ]]; then
        run_checks=1
    fi
done

if [[ "$run_checks" -eq 0 ]]; then
    exit 0
fi

echo "[pre-push] Pushing to main — running CI checks locally"

if ! command -v uv >/dev/null 2>&1; then
    echo "[pre-push] uv not found in PATH" >&2
    exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "[pre-push] Syncing dependencies"
uv sync --extra dev --extra test --extra types --extra docs --quiet

echo "[pre-push] ruff format --check"
uv run ruff format --check .

echo "[pre-push] ruff check"
uv run ruff check .

echo "[pre-push] pyright"
uv run pyright faultmap/

echo "[pre-push] pytest"
uv run pytest tests/

echo "[pre-push] sphinx-build"
uv run sphinx-build -b html docs/ docs/_build/html -W --keep-going

echo "[pre-push] all checks passed"
