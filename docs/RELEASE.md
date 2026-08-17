# Release runbook: `dev` → `main`

This is the procedure the **Dev2Main** Routine follows to promote `dev` to `main`. It lives in the repo (not in Claude settings) so it's versioned, PR-reviewable, and run by reference: the Routine prompt is a thin pointer at this file.

> **Override for this procedure only:** the final release PR's `base` is
> **`main`**, not `dev`. This is the one sanctioned exception to CLAUDE.md's
> "never open a PR to `main`" rule — it applies solely to the release PR
> opened in step 5, and only when running this runbook.

Work through the steps in order.

## 1. Vulture dead-code audit

Run:

```
vulture vtsearch/ app.py tests/ .vulture-whitelist.py --min-confidence 60 \
  --exclude '*/vtsearch/schemas/*,*/vtsearch/settings_models.py' \
  --ignore-decorators '@*.route,@*.before_request,@*.after_request,@*.errorhandler,@*.teardown_request,@*.context_processor,@bp.*,@app.*,@pytest.fixture,@pytest.mark.*,@fixture,@*.fixture' \
  --ignore-names 'Meta,model_config,_keys_to_ignore_on_load_unexpected,test_*,Test*,setup_method,teardown_method,setup_class,teardown_class,pytest_*,pytestmark,__enter__,__exit__,__package__'
```

Vulture is intentionally **not** a CI gate (false positives against the plugin-discovery pattern), so a non-clean exit doesn't block the promotion — but every finding gets triaged:

- **Genuinely unused** → delete the symbol and any imports/references that fall out.
- **Used reflectively** → add it to `.vulture-whitelist.py` with a one-line comment explaining the indirect use.

## 2. Land the cleanup

Commit the triage changes on the current branch **before** opening the PR — not as a follow-up. If vulture was clean, skip this step.

## 3. Write the release summary

Run `git fetch origin --prune`, then summarize
`git log origin/main..origin/dev --no-merges --reverse`.

**Format:** categorized bullets under these headings (collapse any with zero items): **Features**, **Bug fixes**, **Performance**, **Refactors / internals**, **Dev tooling & docs**.

**Constraints:**

- Hard cap **1000 characters**. If you'd exceed it, drop or combine the lowest-impact items.
- No PR numbers, no `#1234`, no commit SHAs, no author handles, no branch names, no issue references. The summary is for end-users.
- If vulture was clean, append a single line: `vulture: clean.` Otherwise:
  `vulture: <N> findings triaged.`

Use the summary verbatim as the PR body (step 5) and also output it in chat, written in plaintext, so the MD formatting can be copied.

## 4. Rebuild the punch-card graphic

`scripts/punchcard/punchcard.py` is a **pure renderer**: it reads the hand-maintained data file `scripts/punchcard/pr_merges.txt` (one `<pr_number>|<merged_at_utc_iso8601>` line per merged PR) and rewrites the PNG. It does **not** generate the data file — you refresh that yourself first.

- Append a line for each PR merged into `dev` since the last release. These are the same PRs you enumerate for the release range in step 3 / step 6; take each one's number and its merge timestamp (`merged_at`, UTC ISO 8601) and add `<pr_number>|<merged_at>` to `scripts/punchcard/pr_merges.txt`. The file is sorted-unique by PR number, so keep it that way.
- Run `python scripts/punchcard/punchcard.py`, which rewrites `scripts/punchcard/vtsearch_pr_punchcard.png`.
- Commit the regenerated PNG and the updated `pr_merges.txt`.

## 5. Open the release PR

- **Title:** `Release: dev → main (YYYY-MM-DD)` using today's date.
- **Base:** `main`. **Head:** `dev`.
- **Body:** the step-3 summary, verbatim.

## 6. Close the issues shipped in this release

Now that the release PR is open, close the GitHub issues whose fixes are included in this `dev → main` batch. This is the counterpart to the per-fix rule in CLAUDE.md: individual fix PRs link their issue with a `Closes #N` keyword but leave it **open** (their merge to `dev` can't auto-close it), and this step is what finally closes it once the fix reaches `main`.

**Find the candidate issues** from this release's PRs:

- List the pull requests merged into `dev` within this release range — the same `origin/main..origin/dev` window used for the summary in step 3 (inspect the merge commits in that range to get the PR numbers).
- For each such PR, read its body and collect **every** issue it references, keeping track of which keyword introduced each reference. Sort them into two buckets:
  - **Closing** — `Closes #N`, `Fixes #N`, `Resolves #N` (case-insensitive).
  - **Non-closing** — `Refs #N`, `Part of #N`, or a bare `#N` mention.
- Then add a third bucket, the **orphan backstop**: list the repo's still-open issues and check each one's comments for a pointer at a PR in this release range (`Addressed in #M`, `Fixed in #M`, and similar). Collect any whose pointer names a PR in the range that never referenced it back. (To keep this cheap, it's enough to check issues updated since the previous release.)

Non-closing references are **not** silently skipped. A PR that finishes an issue but writes `Refs #N` would otherwise orphan it permanently: this step skips it, and because no later release re-examines an already-merged PR, nothing ever revisits it — the issue stays open forever while its fix is live in `main`. Real incident: #2940, #2930 and #2951 each shipped in the 2026-08-12 release under `Refs`, with an "Addressed in #M" comment on the issue, and all three stayed open. So the non-closing and orphan buckets get **reconciled** rather than dropped.

**Reconcile each issue in the non-closing and orphan buckets.** Read the issue (body *and* comments) alongside the PR, then close it only when **both** hold:

- The PR (or a comment on the issue pointing at it) claims to address the issue **without qualification** — e.g. an `Addressed in #M` comment, or a PR body that plainly does everything the issue body asks.
- Neither the PR body nor any later comment names work still owed **by that issue**. Scope the PR explicitly deferred into a plan file or a separate issue is no longer owed here and does not make it partial; likewise, an issue that was rescoped narrower counts as finished if the PR does all of what remains.

Anything else stays open — a genuinely partial `Refs` is doing its job.

**Then, for each issue to be closed (closing bucket, plus the reconciled ones):**

- Skip it if it is already closed. Never reopen or re-close.
- Close it with `state_reason: completed`.
- **Strip the `dev` label in the same write.** `dev` means "fixed on `dev`, NOT yet on `main`" (see CLAUDE.md), and this close is the moment that stops being true. Pass `labels` explicitly with every label the issue keeps (`claude`, `experiment`, …) minus `dev`; `labels` *replaces* the whole set, so passing `[]` would wipe the rest. A `PreToolUse` hook blocks a `completed` close that keeps `dev` or omits the array.
- Add a one-line comment noting it shipped to `main` in today's release and linking the fix PR (e.g. `Shipped to main in the 2026-07-14 release — fixed
  in #M.`). When the PR used a non-closing keyword, say so in that comment, so the mislabel is visible on the issue rather than silently corrected.

**Report the reconciliation in chat**, briefly: which issues came from the closing bucket, which were closed after reconciliation (and under which PR keyword), and which non-closing references were deliberately left open. This is the only place a crossed wire between a PR keyword and an issue comment becomes visible, so do not collapse it to a bare count. If no qualifying issues are found, state that and do nothing.

## 6b. Refresh the `dev` label between releases

Step 6 is what *removes* `dev`. What puts it on is `scripts/reconcile-dev-labels.py`, and that runs **between** releases, not only at one: the label's whole job is to make the awaiting-release view (`is:issue is:open label:dev`) correct while the release is still weeks away. Run it whenever you want that view refreshed, and once more right after step 6 to confirm nothing was left behind.

The script is a pure function from data to plan — it does no network I/O, because the GitHub REST API is unreachable from a Claude session (`GITHUB_TOKEN` is present but returns 403; GitHub access is intermediated by the MCP server). So gather the data first, then pipe it in:

1. List the PRs merged into `dev` since the last release — the same `origin/main..origin/dev` window as step 3 — and read each one's **body**.
2. List the repo's issues with their `labels` and `state`, and fetch each one's **comments** in chronological order (the API default).
3. Assemble them into one JSON object and run the script:

```json
{
  "release_prs": [{"number": 3128, "body": "... Closes #3077 ..."}],
  "issues": [
    {"number": 3077, "state": "open", "labels": ["claude"],
     "comments": [{"body": "Addressed in #3128"}]}
  ]
}
```

```
python scripts/reconcile-dev-labels.py --input plan-input.json
```

It prints four buckets. **Apply `ADD` and `REMOVE` directly** — they are unambiguous. **Do not apply `NEEDS REVIEW`**; read those issues yourself. An issue lands there when a fix pointer is not the newest comment, which is genuinely undecidable from the outside: the later comment may be a maintainer saying "thanks" or the reporter saying the fix does not work. Tagging would bury a dispute; skipping would silently drop the issue out of the awaiting-release view. That is the same "not silently skipped" principle step 6 applies to non-closing references.

Add `--check` to make it exit non-zero when anything needs attention, and `--json` for machine-readable output.

## 7. Prune plan pointers for the closed issues

Per CLAUDE.md's "Issues vs `docs/plans/`: one item, one home" invariant, plan files reference shipped issues by a one-line checkbox pointer (`- [ ] #N — title`) rather than duplicating their bodies. When an issue closes, its pointer is stale and should go.

For **every** issue closed in step 6, grep `docs/plans/` for its number:

```
grep -rn '#<number>' docs/plans/
```

For each hit, delete that pointer line (or check its box, `- [x]`, if the umbrella deliberately keeps a shipped-slice ledger — prefer deletion unless the surrounding plan clearly does the latter). Leave the `<!-- item-sep -->` sentinels around it in place, per the plan-file policy. If the deletion empties a plan entirely and no follow-ups remain, delete the plan file (after absorbing any lasting design notes into the permanent docs), as the plan-file policy directs. Commit these prunes.

This is what makes issue-dismissal trickle back automatically: because plans hold only pointers (never bodies), pruning is always a safe one-line deletion.

**Whenever this step deletes a plan file, also grep the source tree for it** — not just `docs/plans/`. Module docstrings and inline comments cite plan files by path far more often than other plan files do, and `docs/plans/` alone misses all of them:

```
grep -rl 'docs/plans/<deleted-name>\.md' --include="*.py" --include="*.ts" --include="*.sh" --include="*.md" --include="*.json" --include="*.html" .
```

Fix every hit in the same commit: repoint it at the permanent doc the rationale was folded into, or drop the pointer outright when the surrounding prose is already self-contained (the common case). See CLAUDE.md's plan-file policy for the full rule; issue #2982 is the incident that motivated it — 94 source files had gone dangling this way across 13 deleted plans before anyone grepped for them.
