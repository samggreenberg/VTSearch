#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
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
