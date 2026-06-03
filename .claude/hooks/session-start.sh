#!/bin/bash
# Note: deliberately NOT using `set -u`. The hook used to start with
# `set -euo pipefail` and `cd "$CLAUDE_PROJECT_DIR"`; if CLAUDE_PROJECT_DIR
# was unset (which happens in some harness configurations), nounset would
# error out before the rebase ever ran, and the harness silently swallowed
# the failure. Removing -u plus the fallback below keeps the rebase running
# even when the env var is missing.
set -eo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Resolve project dir defensively. CLAUDE_PROJECT_DIR is set by the harness
# but has been observed unset in practice; fall back to the script's repo.
project_dir="${CLAUDE_PROJECT_DIR:-}"
if [ -z "$project_dir" ]; then
  project_dir=$(cd "$(dirname "$0")/../.." && pwd)
fi
cd "$project_dir"

# Helper: print to BOTH stdout (so Claude sees it in session context via
# the SessionStart hook output) and stderr (so it shows in container logs).
notify() {
  echo "$1"
  echo "$1" >&2
}

# Rebase the working branch onto origin/dev. The harness cuts new branches
# off `main` (the GitHub default), so without this the session starts
# without commits already merged to `dev`. See CLAUDE.md "Branch Policy".
current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
skip_reason=""
if [ -z "$current_branch" ] || [ "$current_branch" = "HEAD" ]; then
  skip_reason="detached HEAD or no branch"
elif [ "$current_branch" = "dev" ] || [ "$current_branch" = "main" ]; then
  notify "ℹ session-start: on $current_branch; skipping dev rebase."
elif ! git diff-index --quiet HEAD -- 2>/dev/null; then
  skip_reason="working tree dirty (would clobber changes)"
else
  notify "ℹ session-start: fetching origin..."
  if ! git fetch origin --prune 2>&1 | sed 's/^/  /' >&2; then
    skip_reason="fetch failed"
  elif git merge-base --is-ancestor origin/dev HEAD 2>/dev/null; then
    notify "✓ session-start: $current_branch already includes origin/dev; nothing to do."
  elif ! git rev-parse --verify --quiet "refs/remotes/origin/$current_branch" >/dev/null; then
    # No origin/<branch>: a brand-new branch the harness just cut off `main`
    # (the GitHub default). At session start, before Claude has pushed
    # anything, EVERY commit unique to this branch relative to `dev` was
    # inherited from `main` — there is no Claude work to preserve. That
    # inherited history can include `main`-only cleanup/revert commits (e.g.
    # "revert the accidental main merges") that *conflict* when replayed onto
    # `dev`, because they try to undo work that legitimately lives on `dev`.
    # `git cherry` flags such a revert `+` ("not patch-equivalent to dev"),
    # which sends the cherry heuristic below down the rebase path and produces
    # phantom conflicts — exactly the speed bump this branch was created to
    # fix. Since nothing pushed is at risk, hard-reset to origin/dev
    # unconditionally instead of attempting a doomed rebase.
    notify "ℹ session-start: $current_branch has no origin counterpart (fresh branch cut off main); hard-resetting to origin/dev (inherited main-only commits carry no Claude work)..."
    if git reset --hard origin/dev 2>&1 | sed 's/^/  /' >&2; then
      notify "✓ session-start: hard-reset $current_branch to origin/dev."
    else
      skip_reason="hard-reset to origin/dev failed"
    fi
  elif [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$current_branch")" ]; then
    # The branch exists on origin and HEAD differs from it: it carries real
    # pushed work that a reset/rebase here could orphan. Stay conservative and
    # leave it to Claude to reconcile by hand.
    skip_reason="local $current_branch differs from origin/$current_branch (pushed work would be orphaned)"
  else
    # In sync with origin/<branch> but behind dev. Decide rebase vs hard-reset.
    # `git cherry origin/dev HEAD` flags each unique-to-branch commit as `+`
    # (genuinely new) or `-` (patch-equivalent to a commit already on dev).
    # All-`-` means a rebase would just produce phantom conflicts against the
    # already-merged work; hard-reset is the safe and correct move. Any `+`
    # line means there's real pushed work to preserve, so fall back to a
    # normal rebase.
    cherry_out=$(git cherry origin/dev HEAD 2>/dev/null || echo "")
    if [ -n "$cherry_out" ] && ! echo "$cherry_out" | grep -q '^+'; then
      dup_count=$(echo "$cherry_out" | grep -c '^-' || true)
      notify "ℹ session-start: $current_branch has $dup_count commit(s), all patch-equivalent to commits already on origin/dev; hard-resetting to origin/dev (no real work would be lost)..."
      if git reset --hard origin/dev 2>&1 | sed 's/^/  /' >&2; then
        notify "✓ session-start: hard-reset $current_branch to origin/dev."
      else
        skip_reason="hard-reset to origin/dev failed"
      fi
    else
      notify "ℹ session-start: rebasing $current_branch onto origin/dev..."
      if git rebase origin/dev 2>&1 | sed 's/^/  /' >&2; then
        notify "✓ session-start: rebased $current_branch onto origin/dev."
      else
        skip_reason="rebase failed (aborted, branch left as-is)"
        git rebase --abort 2>/dev/null || true
      fi
    fi
  fi
fi

# When the auto-rebase didn't happen, make it impossible for Claude to miss.
# Echoed to stdout so it appears as session-context for the assistant.
if [ -n "$skip_reason" ]; then
  cat <<EOF
‼ session-start: DID NOT rebase onto origin/dev ($skip_reason).

ACTION REQUIRED before editing any code:
  1. Save / commit / stash any in-progress work.
  2. Run: git fetch origin --prune && git rebase origin/dev
  3. Resolve conflicts (or hard-reset to origin/dev if local commits are stale duplicates).
  4. Re-run any complexity / lint / test analysis AFTER the rebase; the pre-rebase
     view of the codebase is stale and conclusions drawn from it will be wrong.
EOF
fi

# Install lightweight dev tools only: linter and formatter.
# Heavy dependencies (PyTorch, transformers, etc.) are installed lazily
# by ensure-test-deps.sh the first time tests or the app are run.
pip install ruff -q

# Surface missing or placeholder HF_TOKEN loudly so gated-model downloads
# (DINOv3 etc.) don't fail later with confusing 401s. Don't hard-fail the
# session; non-gated work should still proceed.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "‼ HF_TOKEN is not set. Gated Hugging Face downloads (e.g. DINOv3) will fail." >&2
  echo "  Set it in .claude/settings.local.json under {\"env\": {\"HF_TOKEN\": \"hf_...\"}}." >&2
elif [ "$HF_TOKEN" = "PASTE_YOUR_HF_TOKEN_HERE" ]; then
  echo "‼ HF_TOKEN is still the placeholder value. Replace it in .claude/settings.local.json with a real token from https://huggingface.co/settings/tokens." >&2
fi
