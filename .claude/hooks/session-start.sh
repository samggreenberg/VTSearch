#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Rebase the working branch onto origin/dev. The harness cuts new branches
# off `main` (the GitHub default), so without this the session starts
# without commits already merged to `dev`. See CLAUDE.md "Branch Policy".
cd "$CLAUDE_PROJECT_DIR"
current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
if [ -z "$current_branch" ] || [ "$current_branch" = "HEAD" ]; then
  echo "‼ session-start: detached HEAD or no branch; skipping dev rebase." >&2
elif [ "$current_branch" = "dev" ] || [ "$current_branch" = "main" ]; then
  echo "ℹ session-start: on $current_branch; skipping dev rebase." >&2
elif ! git diff-index --quiet HEAD -- 2>/dev/null; then
  echo "‼ session-start: working tree dirty; skipping dev rebase to avoid clobbering changes." >&2
else
  echo "ℹ session-start: fetching and rebasing $current_branch onto origin/dev..." >&2
  if git fetch origin --prune 2>&1 | sed 's/^/  /' >&2 \
      && git rebase origin/dev 2>&1 | sed 's/^/  /' >&2; then
    echo "✓ session-start: rebased $current_branch onto origin/dev." >&2
  else
    echo "‼ session-start: rebase failed; aborting and leaving branch as-is." >&2
    git rebase --abort 2>/dev/null || true
  fi
fi

# Install lightweight dev tools only — linter and formatter.
# Heavy dependencies (PyTorch, transformers, etc.) are installed lazily
# by ensure-test-deps.sh the first time tests or the app are run.
pip install ruff -q

# Surface missing or placeholder HF_TOKEN loudly so gated-model downloads
# (DINOv3 etc.) don't fail later with confusing 401s. Don't hard-fail the
# session — non-gated work should still proceed.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "‼ HF_TOKEN is not set. Gated Hugging Face downloads (e.g. DINOv3) will fail." >&2
  echo "  Set it in .claude/settings.local.json under {\"env\": {\"HF_TOKEN\": \"hf_...\"}}." >&2
elif [ "$HF_TOKEN" = "PASTE_YOUR_HF_TOKEN_HERE" ]; then
  echo "‼ HF_TOKEN is still the placeholder value. Replace it in .claude/settings.local.json with a real token from https://huggingface.co/settings/tokens." >&2
fi
