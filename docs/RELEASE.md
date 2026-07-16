# Release runbook: `dev` → `main`

This is the procedure the **Dev2Main** Routine follows to promote `dev` to `main`. It lives in the repo (not in Claude settings) so it's versioned, PR-reviewable, and run by reference: the Routine prompt is a thin pointer at this file.

> **Override for this procedure only:** the final release PR's `base` is
> **`main`**, not `dev`. This is the one sanctioned exception to CLAUDE.md's
> "never open a PR to `main`" rule — it applies solely to the release PR
> opened in step 6, and only when running this runbook.

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

Use the summary verbatim as the PR body (step 6) and also output it in chat, written in plaintext, so the MD formatting can be copied.

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

**Find the issues to close** from this release's PRs:

- List the pull requests merged into `dev` within this release range — the same `origin/main..origin/dev` window used for the summary in step 3 (inspect the merge commits in that range to get the PR numbers).
- For each such PR, read its body and collect every issue it references with a **closing** keyword: `Closes #N`, `Fixes #N`, or `Resolves #N` (case-insensitive). **Skip** non-closing references (`Refs #N`, `Part of #N`)
  — those are partial and must stay open.

**Then, for each collected issue that is still open:**

- Close it with `state_reason: completed`.
- Add a one-line comment noting it shipped to `main` in today's release and linking the fix PR (e.g. `Shipped to main in the 2026-07-14 release — fixed
  in #M.`).

Do not close any issue that isn't linked by a closing keyword from a PR in this batch, and don't reopen or re-close issues already closed. If no qualifying issues are found, state that in chat and do nothing.

## 7. Prune plan pointers for the closed issues

Per CLAUDE.md's "Issues vs `docs/plans/`: one item, one home" invariant, plan files reference shipped issues by a one-line checkbox pointer (`- [ ] #N — title`) rather than duplicating their bodies. When an issue closes, its pointer is stale and should go.

For **every** issue closed in step 6, grep `docs/plans/` for its number:

```
grep -rn '#<number>' docs/plans/
```

For each hit, delete that pointer line (or check its box, `- [x]`, if the umbrella deliberately keeps a shipped-slice ledger — prefer deletion unless the surrounding plan clearly does the latter). Leave the `<!-- item-sep -->` sentinels around it in place, per the plan-file policy. If the deletion empties a plan entirely and no follow-ups remain, delete the plan file (after absorbing any lasting design notes into the permanent docs), as the plan-file policy directs. Commit these prunes.

This is what makes issue-dismissal trickle back automatically: because plans hold only pointers (never bodies), pruning is always a safe one-line deletion.
