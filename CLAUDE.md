# VTSearch

See [README.md](README.md) for the product one-liner and pitch — the README is the authoritative source (it is hash-matched and served raw), so this file does not restate it.

Architecture, state model, plugin systems, auth, and the directory map all live in **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**. This file holds the testing rules and the policy/gotchas that must be in context every turn.

## Ask Questions via `AskUserQuestion`: NOT prose (CRITICAL, READ FIRST)

This is the **#1 rule** in this repo. Read it on every turn. If you only read one section of CLAUDE.md, read this one.

When you have a question for the user (to disambiguate requirements, choose between approaches, confirm scope, or surface a non-obvious tradeoff), **ask it via the `AskUserQuestion` tool**. Do not guess silently. Do not bury the question in prose at the end of a response. A 10-second clarification beats a 10-minute wrong-direction implementation, and a one-click answer beats a typed answer every time.

**Always ask via the `AskUserQuestion` tool when the question fits its shape** (a discrete choice with a small number of options). Do not leave dangling questions like "Want me to go with approach A or approach B?" at the end of a prose response; those are easy to miss and force the user to type out an answer that could have been a single click. The tool also captures the choice cleanly in the transcript.

This applies *especially* to end-of-investigation "what scope should I take next?" prompts: when a research/investigation turn ends by offering Phase 1 / Phase 2 / smaller scope, the scope choice goes through `AskUserQuestion`, **not** into the prose summary. The investigation findings stay in prose; the "what next?" question is a tool call.

Use plain prose questions only when the answer is genuinely open-ended (e.g. "What should this field be named?") and a multiple-choice list would be artificial.

### Trip-wire: scan your turn before sending

Before sending a turn, scan its last paragraph for any of these phrases:

- "Want me to …?"
- "Should I …?"
- "Do you want … or …?"
- "Let me know if …"
- "(a) … and/or (b) …?"
- "Recommend I …?"

If you see one, **stop**: that sentence is an `AskUserQuestion` call you almost emitted as prose. Convert it into the tool call before sending; even if you're confident the user will say yes, even if the options feel obvious, even if you've already invested effort in the prose summary. The cost of the extra tool call is zero; the cost of a missed or typed-out answer is a wasted round-trip.

This rule has **no exceptions for "quick" yes/no follow-ups.** Yes/no offers belong in the tool too (with `["Yes", "No"]` options); they are exactly the case where a one-click reply beats a typed reply. A pure progress update with no question at the end is fine; an update that ends in an offer is not.

## A bare `#N` prompt is a complete instruction (CRITICAL)

When the user's whole prompt is just a number — `#3421`, or bare `3421` — it names a GitHub issue or pull request **in this repo**, and it is already a full instruction. Do not ask what to do with it; do not answer with a summary of the issue and stop. Resolve the number first (GitHub draws issues and PRs from one sequence, so a given number is one or the other, never both), then follow the matching workflow below.

**This applies on every surface.** Claude Code on the web, the desktop app, the laptop CLI — the convention is repo policy, not a per-session preference, so it holds wherever this file is loaded.

### Post a link to it, right after the assignment

The number the user typed is not clickable, so hand them one that is. Right after step 1's assignment for an issue — or, for a pull request, which has no assignment step, as soon as the number resolves — write one line of text to the user linking the item:

```markdown
Working on [#3421](https://github.com/samggreenberg/VTSearch/issues/3421): <title>
```

- **Now, not in the summary.** It goes out ahead of the evaluation, while you are still reading, so the user has the link from the first minute of the session; saved for the end-of-turn message, it is one link among many.
- **Same form as every other reference** (see "Every `#N` you write to the user is a link" below): the `issues/N` URL serves a PR too, and the title is the GitHub title verbatim, from the lookup that resolved the number. Write it as prose, never inside a code span, where a link renders as literal brackets.
- **Every surface**, unlike the retitle below: a terminal renders the link as well as the web app does.
- **Not a gate.** It is a line of text, not a tool call, and nothing waits on it.

### In a cloud session, retitle it after the issue or PR

A session whose whole opening prompt is `#3421` gets an auto-title that says nothing, and the session list is where the user goes to find this work again. So **when the `set_session_title` tool is available** (the claude-code-remote MCP server, present in cloud sessions on claude.ai/code), rename the session to the item's GitHub title as soon as the number resolves — for an issue, right after step 1's assignment. The lookup that resolved the number already returned the title, so there is nothing extra to fetch; take the session id from `get_session` called with no arguments.

- **Format:** `#3421: <title>`, with the GitHub title verbatim. The number leads so the list scans and searches by it, and it stays bare: a session title is plain text (see "Where `#N` stays bare" below).
- **Cosmetic, never a gate.** Rename once, at the start. If the call fails, carry on without retrying; nothing in the workflow waits on it.
- **Where the tool is absent** (the laptop CLI, a local desktop session), skip this silently. If it is listed only as a deferred tool name, load it with `ToolSearch` first.

### If `#N` is an issue

1. **Assign `samggreenberg` immediately.** The moment the number resolves to an issue — before you read it, before you evaluate it, before any analysis — write `assignees: ["samggreenberg"]`. Evaluation is itself work on the issue, and it is exactly the window in which a second session picks up the same number. Do this even if you go on to disagree with the issue; if you end up walking away, clear the assignee then. Then post the link to it (see "Post a link to it, right after the assignment" above).
2. **Read it, then evaluate it.** Decide whether the premise is correct and whether you agree with the solution it proposes. This is a real gate, not a formality: an issue can be stale, already fixed, based on a misreading, or right about the symptom and wrong about the fix.
3. **If you disagree — or the right scope is unclear — stop and ask** via `AskUserQuestion` before writing any code. Say what the issue claims, what you found instead, and what you'd do differently.
4. **If you agree, do the work off a fresh `dev`:** `git fetch origin --prune && git checkout -B <branch> origin/dev`. Never build on whatever the branch happened to be pointing at.
5. **If the issue carries the `experiment` label, do the work entirely on the GRID, on a fresh worktree.** See the `grid-experiments` skill for how those runs are launched and monitored. Worktrees go under `/expscratch/$USER/worktrees/vts-<issue>`, never commit in the shared `/exp/$USER/projects/VTSearch` checkout or the deploy clone (a hook refuses it), and prune a worktree after its PR merges: see "Worktrees on the GRID" in that skill.
6. **Run the rest of the issue lifecycle**, per "Issue lifecycle" below: open the PR with `base=dev` and a closing keyword; comment `Addressed in #M` on the issue; add `solved` and clear the assignee; leave the issue open for the `dev`→`main` sweep to close.

### If `#N` is a pull request

It is conflicted, and the ask is to **resolve the conflicts so the PR can merge into `dev`** — nothing more. Post the link to it first (see "Post a link to it, right after the assignment" above), then fetch, merge `origin/dev` into the PR's head branch, resolve, run the tests, and push. Do not rewrite the branch's history (no rebase, amend, or force-push) and do not fold in unrelated changes while you are there; a reviewer should see conflict resolution and nothing else.

## Every `#N` you write to the user is a link (CRITICAL)

When you mention a GitHub issue or PR in a message to the user, write it as a markdown link to GitHub — **every occurrence, not just the first one in the turn:**

```markdown
[#3421](https://github.com/samggreenberg/VTSearch/issues/3421)
```

Keep `#3421` as the link text; the number is what the user reads and what they'd search for. Only the target is new.

**You never need to know whether `N` is an issue or a PR.** GitHub redirects `/issues/N` to `/pull/N`, so the `issues` URL is correct for both; there is no lookup to do first, and "not sure which it is" never justifies leaving a reference bare.

**Why every one, rather than the first mention.** The reference the user wants to click is rarely the first — it is the one in the summary table at the bottom, or the sentence saying *this is the issue that's still open* — and a mix of linked and bare copies implies a distinction that does not exist.

**This applies on every surface Claude talks to the user on** — the web app, Claude Code in the terminal, the desktop app, a cloud session's final message — as repo policy, like the bare-`#N` convention above. Within a message it covers prose, end-of-turn summaries, bullet lists and tables alike, repeats inside a single sentence included.

**What decides is the direction of the message, not the surface it's typed on.** Anything travelling *to the user* gets links; anything travelling *to GitHub* — or into git — does not, for the reasons in the four carve-outs below. A surface you haven't seen before is on the user's side of that split unless it's one of those four.

### Where `#N` stays bare

Four places, and the reason is the same each time: the link either doesn't render or actively breaks something.

- **Inside code fences and inline code spans.** A markdown link renders as literal brackets there. `git log --grep '#3421'` is a command, not a reference.
- **Anything written to GitHub** — issue comments, PR titles and bodies, review replies. GitHub **autolinks `#N` natively** in those fields, so a markdown link adds a second URL to something that was already clickable. Write `Addressed in #3421`, as the sections below already say.
- **Closing keywords, specifically.** `Closes #3421` — never `Closes [#3421](https://github.com/samggreenberg/VTSearch/issues/3421)`. That keyword is parsed by GitHub and by `scripts/release/reconcile-solved-labels.py`; both expect the bare form, and this is the one place where dressing up a reference can silently cost an issue its close. See "Linking the fix PR" under "Issue lifecycle" below for what rides on that keyword.
- **Commit messages, branch names, and a cloud session's title.** No renderer, so a link is just noise in `git log` or the session list.

Tracked markdown in the repo — this file, `docs/`, `docs/plans/` — is **out of scope** and keeps its bare `#N` convention (plan pointers like `- [ ] #2355 — …` stay exactly as documented below). This rule is about messages, not files.

### Not every `#` followed by digits is a reference

Link what actually names an issue or PR in this repo. Leave alone: ordinals and rankings (this file's own "**#1 rule**" at the top is the canonical false positive), list positions, heading anchors, CSS ids, and any `#N` that came out of a code sample. If you can't say which issue a number names, it probably isn't one.

While you're at it, prefer the `#N` spelling over "PR 3984" or "issue 3421" in the first place — the `#` form is what the trip-wire below catches, and a prose number is a reference that will never get linked because nothing looks for it.

### Trip-wire: scan your turn before sending

Before sending, scan the turn for `#` followed by digits. **Every hit outside a code span is a link** — not the first hit, every hit. If you find a bare one, fix it before sending; you are already in the message, and re-editing it costs nothing, whereas the user reaching for a URL bar costs a round-trip.

## Branch Policy (CRITICAL)

- **Always base work on `dev`.** The `.claude/hooks/session-start.sh` SessionStart hook fetches `origin --prune` and then lands the working branch on `origin/dev` automatically in remote sessions. The harness cuts the working branch off `main` (the GitHub default), so this is required to pick up work already merged to `dev`. The GitHub default stays `main` so new users land on the stable branch: `dev` is Claude's starting point, not the public default.
  - **The hook can *hard-reset*, not only rebase — read its output.** It picks one of four outcomes and says which: `already includes origin/dev; nothing to do`; **hard-reset** to `origin/dev` (no `origin/<branch>` counterpart, i.e. a fresh branch cut off `main` whose unique commits carry no Claude work, or `git cherry` shows every unique commit is patch-equivalent to one on `dev`); **rebase** onto `origin/dev` (in sync with its origin counterpart and carrying genuinely new pushed commits); or a **skip**. A hard-reset discards the branch's prior commits by design and loses nothing of yours, but do not assume commits you saw in `git log` before the hook ran are still there.
  - If the hook prints `‼ session-start: DID NOT rebase onto origin/dev` (dirty tree, detached HEAD, fetch failed, a reset/rebase that failed, or a local branch that differs from its pushed origin counterpart), run `git fetch origin --prune && git rebase origin/dev` yourself before making any changes.
- **All pull requests MUST target `dev`**, never `main`.
- **Claude must NEVER open or merge a PR into `main`, except as the Dev2Main release.** `main` is protected and changes only through that release: the runbook in `docs/RELEASE.md` opens the `dev` → `main` PR (step 5) and, once a full `./run-tests.sh` on the tip passes, merges it (step 8, owner, 2026-10-06). No other session touches `main`.
- When creating a PR, always use `--base dev` (e.g., `gh pr create --base dev ...` or the equivalent MCP tool parameter).
- If your feature branch was forked from `main` instead of `dev`, rebase or merge onto `dev` before opening a PR.

## Git Fetch Hygiene

Before comparing branches (`git log a..b`, `git diff a...b`, etc.), always run `git fetch origin --prune` first. Do **not** trust `origin/<branch>` refs after a partial fetch like `git fetch origin main`; that only updates the branch you named, leaving other remote-tracking refs stale and producing misleading diffs.

## Auto-PR

When you're done with your changes, open a PR targeting `dev`. Do not ask; just create it. Always pass `base=dev` explicitly (the GitHub PR-creation URL printed by `git push` defaults to `main`).

This standing instruction **is** the explicit request that the remote-environment harness rule ("do not create a pull request unless the user explicitly asks for one") defers to: that rule only suppresses *unsolicited* PRs, and a durable, repo-committed instruction satisfies its carve-out. So the two do not conflict: in this repo, finishing your changes is your cue to open the PR (base `dev`) without further prompting.

## Merging a PR

**Use a normal merge commit: `gh pr merge <n> --merge`.** Never `--squash`, never
`--rebase`.

`scripts/release/release-prs.py` can read a squash too, but that is a backstop, not
permission — a subject-line parse is a weaker thing for a release to rest on than
a merge commit, and two costs are unrecoverable by any script:

- **Ancestry.** A squash-merged branch is no longer an ancestor of `dev`, so a
  follow-up PR from it re-shows its *entire* diff rather than the new commits.
  Sessions here routinely keep working on a branch after its PR lands, which is
  exactly when that bites.
- **Structure.** Every commit collapses into one. The messages survive in the
  squash body, but `cherry picked from` provenance and the shape of the work do
  not.

None of it is repairable afterwards: fixing the history means rewriting a shared,
protected branch. So the check belongs **before** the merge. If you are ever
unsure what this repo does, ask the repo rather than the last commit you happened
to read — a two-commit sample is not a convention:

```
git log --pretty=%s -100 origin/dev | grep -c "^Merge pull request"          # merge commits
git log --pretty=%s -100 origin/dev | grep -cE "\(#[0-9]+\) \(#[0-9]+\)$"   # squashes
```

**Merging is not part of Auto-PR.** Open the PR without being asked; merge it
only when the user says so. The Dev2Main release is the standing exception: its
runbook merges its own housekeeping PRs into `dev` and the release PR into `main`
because the owner asked it to.

## Issue lifecycle (CRITICAL)

Every issue goes through the same five stages, and the table says what each one owes. Two labels are chosen when you file; the other two signals are **statuses** that must stay true as the work moves:

- **`solved`** means there is no problem-solving left: a fix PR carries it, and all that remains is merges (into `dev`, then into `main` at the next release). An open PR counts; do not wait for the merge. A closed issue never carries it.
- **Assignee `samggreenberg`** means somebody is working the issue right now. A stale assignee makes free work look taken, so it goes on early and comes off promptly.

Why each rule exists, the tracker views the four signals keep honest, and how the backstop script decides are in [`docs/ISSUES.md`](docs/ISSUES.md).

| Stage | What you write | What catches a miss |
|---|---|---|
| **Filing** | Search for a duplicate first (see *Duplicates*). Labels at creation, never as a follow-up edit: `claude` always, `experiment` when it applies (see *The filing labels*). A recommended model in the body (see *A recommended model*). | `.claude/hooks/require-issue-labels.py` blocks a create missing `claude`, or missing `experiment` when the issue reads like a run, on `issue_write` and `gh issue create` alike |
| **Starting work** | `assignees: ["samggreenberg"]` the moment you know which issue you are on: **before you read and evaluate it** (for a bare `#N` prompt that is step 1 above), whoever filed it, you included. If evaluation ends with you walking away, `assignees: []`. | Nothing: this half has no backstop |
| **Opening the fix PR** | `Closes #N` in the PR body, or `Refs #N` if partial (see *Linking the fix PR*). Comment `Addressed in #M` on the issue. Then **one write** with `labels` = the issue's existing labels plus `solved`, and `assignees: []`. | The hook blocks an update whose `labels` include `solved` without `assignees: []` |
| **Fix falls through** (the PR closes unmerged, or review finds the fix wrong) | Strip `solved`: the one removal a fix session makes itself. Re-assign only if you are the one picking it back up. | `scripts/release/reconcile-solved-labels.py` |
| **Closing** | Not yours. The Dev2Main sweep ([`docs/RELEASE.md`](docs/RELEASE.md) step 6) closes it `completed` once the fix reaches `main`, passing `labels` minus `solved` and `assignees: []`. The only exception is a duplicate found after its fix shipped (see *Duplicates*). | The hook blocks a `completed` close that keeps `solved` or omits `labels` |

**Write mechanics.** On `issue_write`, `labels` and `assignees` each **replace the whole set**. `[]` wipes it, so a write that touches labels lists every label the issue keeps (read the issue first if you don't know them). **Omitting** a field leaves it untouched, so an assignee-only write never disturbs labels and needs no read. Through `gh`, adding `solved` needs `--remove-assignee` in the **same** `gh issue edit` (`gh issue edit <n> --add-label solved --remove-assignee samggreenberg`); `--add-assignee` or a second chained edit does not count (#4144). A `gh issue close` on an issue that carries `solved` is blocked until you strip it: `gh issue edit <n> --remove-label solved && <your close command>`. The `gh` lookups let the call through whenever they cannot reach GitHub, so they catch mistakes but guarantee nothing. `scripts/release/reconcile-solved-labels.py` is the backstop for every rule here except assigning: it covers `solved`, clearing the assignee, and the `experiment` marker (`docs/RELEASE.md` step 6b).

### The filing labels: `claude` and `experiment`

Pass them as `labels: ["claude", …]` on `issue_write` or `--label claude,experiment` on `gh issue create`. If a label is missing from the repo, applying it creates it, so never skip one because it doesn't exist yet.

- **`claude` means Claude wrote the issue.** It is not a topic tag. Every session files through the owner's account, so this label is the only thing that marks authorship, and human-filed issues are the ones without it. Never put it on an issue a human wrote, and never leave it off one you wrote.
- **`experiment` means the issue cannot be closed without a run**: a GRID/SLURM sweep, an eval study, a calibration. It is a gate, not a topic. The test is whether a competent implementer could close it from a laptop with the test suite, or would need measured results first.
  - Apply it to new eval arms, sweeps and re-runs, calibration rows that must be measured, "is X better than Y?" questions, and research ideas whose first real step is a measurement.
  - Do not apply it to bug fixes, docs, refactors, plumbing, or code whose spec is already measured and decided.
  - It is orthogonal to `claude`. Runs are launched per the `grid-experiments` skill.
- **When the reason isn't obvious from the title, put a marker in the body** so the gate survives the label being dropped: `<!-- experiment: <reason> -->`, for example `<!-- experiment: the file being ported lives on the GRID, so a container can only invent its SBATCH headers -->`. It is a hint, never required, and its text is never parsed. Its twin, `<!-- not-an-experiment: <reason> -->`, opts out of the hook's experiment heuristic. That reason **is** read, and a reason that describes a run is refused.
- The hook also blocks an `issue_write` update whose `labels` drop `experiment`. It looks the issue up through `gh`, so this guard is only awake where `gh` works. To remove the label on purpose, run `gh issue edit <n> --remove-label experiment`.

### Linking the fix PR

- **The PR body's keyword is load-bearing.** The Dev2Main sweep decides what to close by reading PR bodies, not issue comments. A `Refs #N` on a PR that actually finishes the issue orphans it for good, because no later release looks at an already-merged PR again. Default to `Closes` (or `Fixes` / `Resolves`), with one keyword per issue (`Closes #12, closes #15`), never a comma-list after a single `Closes`.
- **Use `Refs #N` / `Part of #N` only when you can name the work still owed on *that* issue.** Rescoping counts as finishing: if you narrowed the body and the PR does everything left, it's `Closes`. Scope moved into a plan file or a separate issue is no longer owed by this one.
- **The issue comment must agree with the keyword:**

  | The PR… | PR body | Issue comment | `solved` |
  |---|---|---|---|
  | fully resolves the issue | `Closes #N` | `Addressed in #M` | add it |
  | does part of it | `Refs #N` / `Part of #N` | `Partially addressed in #M — still open: <what's left>` | leave it off |

  Name the PR number, never only a commit SHA: the backstop script matches `#M`, and a SHA-only pointer lands in its `NEEDS REVIEW` bucket. A SHA is fine as an addition ("Addressed in #3051 (`de9ae81ac`)"). If the comment and the keyword disagree, one of them is wrong; fix it before merging.
- **Leave the issue open.** GitHub auto-closes an issue only when the PR merges into `main`, and our PRs target `dev`. That is intended: the issue stays open until its fix ships. Link, comment and label, and the `solved` label keeps the issue out of the pick-up queue in the meantime.

### Duplicates

**Search before filing.** Run `search_issues` on the *symptom* (a failing test's name, an error string, a file path), not on your diagnosis. Two sessions describe the same bug differently but name the same test. An issue is only ever resolved by something that **names its number**, so a duplicate nobody links stays open forever while its fix is live (#2911). When two issues turn out to be the same bug:

- **Fixing both at once:** put a closing keyword for each in the PR body, and comment `Addressed in #M` on both.
- **Found before any fix:** close the **newer** one with `state_reason: "duplicate"` and `duplicate_of: <older number>`. The older one survives, and the fix PR names it. If the newer one has the better write-up, fold that into the survivor's body; don't reverse the direction.
- **Found after the fix shipped:** comment `Addressed in #M` on the orphan and close it `completed` yourself, stripping `solved` and clearing the assignee. This is the one close a per-fix session makes. The sweep only looks at `origin/main..origin/dev`, so a fix already on `main` is out of its reach.

If you know an issue is resolved and nothing on GitHub says so, write the pointer yourself.

### A recommended model

Every issue you file includes a bolded line near the top, right after the summary, such as `**Recommended Claude model: Sonnet.**`, with a short clause on *why* when it isn't obvious. It routes the task to the cheapest model that will do it well.

- **The ladder, cheapest to most capable: Haiku → Sonnet → Opus → Fable.** Fable is the *most* capable and most expensive, never a cheap rote tier. "Step up" means toward Fable.
- **Name the family, never a version.** Write `Opus`, not `Opus 5.5`, because an issue outlives the lineup it was filed under. Read the versions older issues carry (`Opus 4.8`, `Sonnet 5`) as the family.
- **Size to the hardest part, not the average:**
  - **Haiku:** rote, find-and-replace-shaped edits with a clear spec and no design judgment.
  - **Sonnet:** normal feature and bugfix work with bounded reasoning.
  - **Opus:** regression-prone refactors, subtle concurrency or reactivity, cross-cutting design, or anything where a plausible wrong answer is costly.
  - **Fable:** only long-horizon or research-grade work that genuinely exceeds Opus.

  When tiers mix, name the split, for example "Sonnet for the bulk; Opus for `foo.component.ts`."
- Pointers in a plan umbrella may append the model after the title as a routing hint. This is optional.

This recommendation is about the implementer's model. It has nothing to do with which model **you** report being.

## Issues vs `docs/plans/`: one item, one home (CRITICAL)

GitHub Issues and plan files are **not two copies of the same list.** They hold different kinds of work, and the split is enforced by a single invariant that makes the two stores impossible to desync:

> **No task's body lives in two places.** A plan file may *reference* an issue by number, but must never duplicate the issue's content. There is nothing to "sync" because nothing is stored twice.

**Division of labor:**

- **GitHub Issues own every concrete, independently-shippable task** — bugs, papercuts, small self-contained features. Issues are browsable, closable, PR-linkable, and swept by the Dev2Main Routine, which is exactly what discrete tasks want. A concrete bug belongs in an issue **alone**, never also as a plan-file bullet.
- **`docs/plans/` owns design narrative** — architecture, rationale, sequencing, the "why" and "shape" of multi-step efforts. Reserve plan files for work where the prose earns its keep. A plan that has decayed into a bag of independent one-line bullets *wants* to be N issues; promote it.

**Promoting a plan item to an issue:** when a slice inside a plan becomes concrete enough to ship on its own, file the issue, then **delete that item's body from the plan.** If the plan is a genuine umbrella that needs to show "these slices belong together," leave a **one-line checkbox pointer** in place of the body — never the full text:

```markdown
- [ ] #2355 — Fill in missing demo media counts
```

The pointer carries the issue number (the durable link) and a short human-readable title (so the umbrella is legible at a glance). It does **not** restate the issue's body, file pointers, or fix notes — those live in the issue, which is now the single source of truth. Check the box (or delete the line) when the issue closes.

**Dismissing an issue as unwarranted:** close it as `not_planned` with a one-line comment explaining why. If any plan file points at it (grep `docs/plans/` for `#<number>`), prune that pointer line in the same motion. Because plans carry only pointers — not bodies — this is always a trivial one-line deletion, whether the issue was dismissed or merged. (The Dev2Main Routine does the same prune for the issues it closes.)

## Work still owed goes in an issue, never in a closing paragraph (CRITICAL)

When a session ends with something left to do — a re-run, a correction to an old comment, a check that needs the GRID, a failure you did not fix — **file a GitHub issue for it**, following the filing row of "Issue lifecycle" (search for a duplicate, label it, recommend a model), and link that issue in your summary. Do **not** write it into the end-of-turn message as a task for the user ("once this merges, you can re-run X from your laptop"). A to-do in the last paragraph of a long summary is easy to miss, cannot be tracked or closed, and gives the user work they never agreed to.

**Before filing, check that the work is still owed.** Search the tracker, and read the threads the work comes from. A follow-up has often been done already (#3884's session asked the user for a correction that #3877's closing comment already recorded). If nothing is owed, say so in one line and file nothing.

The split: a **decision** for the user goes through `AskUserQuestion`, and **work** goes into an issue (or, when it is design narrative, a plan under `docs/plans/`). Neither goes into prose.

## Plan files (`docs/plans/`) track FUTURE work only

A plan file describes **work still owed**: a proposed feature, or the open parts of one in progress. Plans are **not an archive of completed work.** Git history and merged PRs are the record of what already landed; the plan is what someone reads to pick up what's left.

**When you ship, prune what you finished:**

- **Fully shipped, nothing left → delete the file.** Do not leave it behind marked "done" / "shipped" / "kept as reference." First fold any durable design rationale into `docs/ARCHITECTURE.md` / `docs/EXTENDING.md` (or their siblings) where permanent docs belong.
- **Partly shipped → delete the shipped narrative, keep only what's still owed.** Remove "What shipped" sections, resolved-finding catalogs, phase-by-phase ship logs, strikethrough-completed checklists, and completion dates. What remains is the open work.
- **Keep past context only when future work needs it.** If the remaining work can't be understood without some of what already shipped, keep a *short* "Background" note at the top — a paragraph, not a changelog. That is the single exception to "no records of past work."
- **Grep the source tree before deleting, not just `docs/plans/`.** Module docstrings and inline comments cite plan files by path (`See docs/plans/<name>.md`) far more often than other plan files do. Before deleting `docs/plans/<name>.md`, run `grep -rl 'docs/plans/<name>\.md' --include="*.py" --include="*.ts" --include="*.sh" --include="*.md" --include="*.json" --include="*.html" .` (adjust extensions to what the repo actually cites from) and fix every hit in the same commit: either repoint it at the permanent doc the rationale was folded into, or drop the pointer outright when the surrounding prose is already self-contained (the common case). A dangling citation left in source is a documentation regression, not a harmless leftover: it is the "why is this code shaped like this" pointer a future maintainer follows, and it now leads nowhere (`scripts/gates/check-docs.py` fails on one).

**Follow-ups go in the plan file, not the PR body.** When you identify deferred scope or known limitations, record them as open work in the relevant plan under `docs/plans/` (the one that scoped the feature). Do **not** stash them in the PR description as the only record — PRs close, get archived, and stop surfacing in normal discovery. The PR body describes what landed; the plan tracks what's still owed.

**Name plan items; never renumber them.** Identify each item in a plan by a stable, descriptive **name** (a bolded title), not by its position in a `1., 2., 3.` list: renumbering the survivors after a deletion is a gratuitous merge conflict whenever two efforts ship different items at once. So write the open-work list as a plain **bulleted** list (`- **Name** — …`); refer to items by name in prose and PRs. If you inherit a numbered plan, treat the numbers as arbitrary stable labels: when you delete an item, **leave every surviving item's number exactly as it is** — gaps (`1, 3, 4`) are fine and expected — or, better, drop the numbers to bullets as you touch the file. The only edit a shipped slice makes is deleting its own item; it never touches another item's label or number.

**Minimize churn to avoid merge conflicts.** When your effort ships a slice, make the **smallest edit that removes the work you finished** — delete the completed items (leaving the rest untouched, per "never renumber" above), don't reflow or restructure the surrounding prose, don't rewrite a status header into a ship narrative, don't append a completion log.

**Separate every item with a `<!-- item-sep -->` sentinel, and never delete a sentinel.** Deleting your own item is *not* enough on its own. Git merges by diff hunk, not by item: when two parallel efforts delete **adjacent** items, their deletions abut with no unchanged line between them, and git flags a conflict even though each side only deleted (items *far apart* merge cleanly; adjacency is the trigger, not "same file"). A permanent separator between every pair of items guarantees that an unchanged line survives between any two deletions, giving git the common context it needs to auto-merge. So:

- Every plan item is preceded (or followed — pick one and be consistent within a file) by a lone `<!-- item-sep -->` line on its own, blank-line-separated from the bullets on either side. The sentinel renders as nothing.
- When you ship a slice, **delete only your item's own lines; leave the sentinels above and below it in place.** Deleting a sentinel re-creates the adjacency problem for the next pair.
- Empty runs of back-to-back sentinels (left behind by deleted items) are harmless and expected — sweep them opportunistically when you're editing the file for another reason, never as a churn-only commit that would itself conflict with in-flight deletions.
- A plan that predates this convention gets sentinels added the next time someone touches it; until then its parallel-deletion conflicts stay trivial (both sides only delete) and are resolved by taking both deletions.

If a plan fully ships and no follow-ups remain, deleting it (after absorbing any lasting design notes into the permanent docs) is the expected outcome, not an oversight.

## PR Activity Subscription (do not ask)

Never ask the user whether to subscribe to PR activity, and never call `subscribe_pr_activity`. The user does not want Claude to watch PRs or respond to review comments / CI. This overrides the default GitHub Integration instruction to offer PR subscription after creating a PR.

## Claude.ai Projects: one project, one repository (CRITICAL)

A [project](https://code.claude.com/docs/en/claude-projects) at claude.ai/code is one
coordinating conversation plus N parallel cloud sessions ("threads"), each on its own
branch with its own PR. Projects are scoped to a **stream of work**, not to a
repository, so one repo can back several projects and one project can clone several
repos. VTSearch uses **one project, holding this repository alone.**

**Never add a second repository to a VTSearch project.** This is not a preference; it
silently disables the three hooks this repo depends on. Threads read permission rules,
hooks and `env` only from the `.claude/settings.json` *in the directory the thread
starts in* — inside the repository when the project has one, and **above the clones
when it has several, where no repository's file is read for them.** Our
`.claude/settings.json` carries the SessionStart hook that lands the working branch on
`origin/dev` (see Branch Policy) plus the `ensure-test-deps-gate.py` and
`require-issue-labels.py` `PreToolUse` gates. With one repository all three run; with
two they are gone, with nothing in the thread saying so. A thread whose task needs
another repo can add it to *itself* — that is the supported path, and it leaves the
project single-repo.

**Creating a project needs settings this repository cannot enforce.** Three project defaults (branching from `main`, auto-fix watching of a thread's own PR, every thread on Opus) contradict rules in this file; the override text to paste into **Project settings > Memory > Project instructions**, and how to verify it took, are in [`docs/claude-projects.md`](docs/claude-projects.md). Inside a thread, the rules here already apply: base on `dev`, never watch the PR you opened, size the model to the work.

**What does not belong in a thread.** Anything reaching the GRID: the
`grid-experiments` skill drives SLURM over `ssh grid`, which a cloud sandbox cannot
do, and the docs put work needing machine-only services in a local session. So an
`experiment`-labelled issue is laptop work. A thread *can* do the analysis half —
reading finished cells, writing a `REPORT.md` — once the run exists.

Plan limits are shared with every other session and a project spends them faster;
the enforced ceiling is 200 new threads per day across all projects.

## Versioning (do NOT bump by hand)

`vtsearch.__version__` is the UTC timestamp of `HEAD`'s commit (ISO 8601, Z-terminated), computed from git at import time in `vtsearch/__init__.py`. There is no tracked version constant to bump; every commit on `dev` automatically becomes the new version, and parallel branches cannot collide on a hand-edited version line. Do not add a `VERSION` file, do not write a hand-bumped string into `vtsearch/__init__.py`, and do not include version bumps in feature PRs. For Docker images (where `.git` is excluded from the build context), the host passes `--build-arg VTSEARCH_VERSION=$(TZ=UTC git log -1 --format=%cd --date=format-local:%Y-%m-%dT%H:%M:%SZ HEAD)` and the Dockerfile bakes it into `vtsearch/_version.txt` (gitignored). If git is unavailable and the baked file is missing, the version falls back to `0.0.0-unknown`.

**The frontend bundle carries the same stamp**, written by `frontend/scripts/build-stamp.mjs` so a stale SPA can be detected at startup (the mechanism is in [`docs/FRONTEND.md`](docs/FRONTEND.md#the-build-stamp-and-the-skew-check)). Nothing there is hand-edited; if you change how the Python version is derived, change `build-stamp.mjs` to match.

**`vtscore.__version__` is different.** The library uses independent semver, tracked as a hand-edited constant in `vtscore/__init__.py` (currently `0.1.0`). Bump it only when cutting an actual `vtscore` release, and add a matching entry to `vtscore/CHANGELOG.md`. Do *not* include `vtscore` version bumps in unrelated feature PRs: `vtsearch` is continuously deployed (every commit = new version), while `vtscore`'s external consumers expect stable, semver-tagged releases.

## Backwards Compatibility

**Which surface breaks decides how much care it's owed.** The two cases pull in opposite directions, so name the surface before deciding.

### Saved data and internal surfaces: break freely

Persisted artifacts (settings files, detector JSON, dataset pickles, cached sidecars, exported files) and anything internal to the app (routes talking to our own SPA, module layout, function signatures inside `vtsearch`) may be broken without ceremony. VTSearch does not produce long-lived exports that have to survive version to version, and both halves of the app ship together. Do **not** add migration shims, feature flags, legacy re-exports, or compatibility layers to preserve old behavior here. Just make the clean change, and mention the break to the user so they're aware.

### Extension-facing surfaces: don't break casually

A third-party developer may have written code against our plugin ABCs (`ResultsExporter` and its permanent alias `LabelsetExporter`, `DatasetImporter`, `MediaConverter`, and their siblings), the registry sentinels (`EXPORTER`, `IMPORTER`, …), the `vtscore.*` entry-point groups, or the public `vtscore` library API. Breaking those forces someone outside this repo, who never agreed to it, to refactor working code on our schedule.

This is not an absolute bar — a genuinely better contract can be worth it — but it must be a deliberate decision, not a side effect:

- **Prefer additive.** A new method alongside the old one, with the base class delegating so an existing plugin keeps working, is usually available and usually cheap. Reach for it first.
- **Keep the cheap escape hatches.** A module-level alias for a renamed class, or a default implementation that delegates to the old method, costs a line or two and preserves every out-of-tree caller. Unlike a data migration, these do not rot.
- **When a hard break is right, make it loud, not silent.** A plugin that no longer satisfies the contract should be rejected with a message naming what's missing — never half-work.
- **Say so where extension authors will see it:** an `[Unreleased]` entry in `vtscore/CHANGELOG.md`, plus the relevant guide under `docs/EXTENDING-plugins.md` / `vtscore/docs/extending/`.
- **Raise it with the user before committing to it.** An extension-facing break is a decision to make on purpose, out loud.

### "Dead code" in `vtscore/` is a claim you cannot verify by grepping

Out-of-tree extensions import `vtscore` symbols this repository cannot see, so a
repo-wide grep proves a symbol is unused *by us* — never that it is unused. Treat
"no callers found" as a prompt to classify, not a licence to delete:

- **Delete freely:** private (`_`-prefixed) symbols, `vtsearch/` app-tier internals,
  frontend code, tests, and one-off scripts.
- **Keep the name, retire the body:** anything exported from a `vtscore` package
  `__init__`, documented under `vtscore/docs/`, a plugin ABC method, a registry or
  `register_*` function, an entry-point-facing name, or a public module-level
  constant. Collapse it to a thin delegation, or deprecate it with an
  `[Unreleased]` note in `vtscore/CHANGELOG.md`.
- **Public-but-undocumented is still public.** A name without a leading underscore
  is importable even when it is absent from `__all__` and from the docs.

A zero-registrant extension point is the shape of a *working* extension point, not
a dead one: `register_*` hooks, ABC methods no shipped subclass overrides, and
"backwards-compatible getter" helpers all look unused from inside this repo by
design. Removing one is a deliberate library break — raise it with the user first,
per the bullets above.

## Frontend Scope: Desktop Only

VTSearch is a desktop web app. **Do not design, implement, or test for mobile or narrow viewports.** No responsive breakpoints, no touch-targeted controls, no mobile-only layouts, no concerns about portrait orientation. If a design discussion raises "what about mobile?", the answer is "we don't care." When evaluating a layout, assume a standard desktop viewport and skip mobile considerations entirely.

## Render and attach slide decks (when you change `slides/`)

**Any session that changes the slide codebase must end by rendering the affected decks and attaching the changed pages in the conversation** (via `SendUserFile`), so the user can see the result in-browser without checking out the branch and running the build themselves. "Changes the slide codebase" means any edit under `slides/` that can change what a rendered deck looks like — fragments, `.deck` manifests, the theme, figures, or the build/render tooling. Prose that renders nothing (`README.md`, `STYLE.md`) is the one exception: there is no output to show, so attaching pages identical to the last render is noise rather than evidence.

- **Which decks:** every deck whose output the change affects. A fragment edit affects the decks whose manifests name it (grep `slides/decks/*.deck`); a theme, `build.py`, or `render.sh` change affects all decks. When in doubt, render more rather than fewer.
- **How:** `cd slides && ./render.sh <deck> pdf` for each affected deck. The cloud container has a browser; if Marp can't find it, use `CHROME_PATH=/opt/pw-browsers/chromium`. When presenter notes or the speaker pipeline changed, also render `./render.sh <deck> pdf --speaker` and attach that variant's pages too.
- **Attach only the pages the change touched — never the whole deck.** A full deck PDF is far over the 30 MiB attachment limit (~75 MiB audience, ~36 MiB speaker), and the user wants the slides that moved anyway: cut an excerpt holding the affected section (or the affected slides plus one either side) from each rendered PDF and attach that. The PyMuPDF recipe and how to find the page range are in [`slides/README.md`](slides/README.md#attaching-an-excerpt-claude-sessions).
- **When:** at the end of the session, from the final state of the branch (after the last commit that touches `slides/`), so what the user sees is what the PR ships. Attach with a one-line caption naming the deck(s), the section or pages the excerpt holds, and which page numbers are new or moved.
- Rendered PDFs and their excerpts stay out of git (`slides/_out/` is gitignored; excerpts go in the session scratchpad), and are never published by hand from a session: `.github/workflows/publish-slides.yml` re-renders every deck on each push to `dev` that touches `slides/` and uploads it to the rolling `slides-latest` release (see `slides/README.md` § Publishing).

## Screenshot reshoots: queue them, don't shoot them (when you change the GUI)

The user docs (`docs/user/assets/`) and the slide deck's UI figures (`slides/figs/ui-*.webp`) are screenshots of the app, rendered by Playwright harnesses: `scripts/screenshots/refresh.sh` and `slides/figs/src/shoot-ui-figs.mjs`. **A session that changes the GUI does not run either one**: re-rendering costs far more than most GUI changes do, and the next change moves the same shots again anyway. Instead:

- **Queue what your change moves.** Add one new file per change under `docs/reshoot-queue/` — never a row in a shared table, so parallel PRs can't conflict on it — naming the shots your change alters: a docs shot by its manifest id, a slide figure as `slides:<group>`. The format, and how to find the ids, are in [`docs/reshoot-queue/README.md`](docs/reshoot-queue/README.md). A rough list is fine: the drain re-renders the whole docs set, so a missed shot is still caught.
- **Dev2Main drains the queue** at every release ([`docs/RELEASE.md`](docs/RELEASE.md#4b-drain-the-screenshot-reshoot-queue)), before the release PR opens, so `main` never ships a stale screenshot. `dev`'s screenshots may lag its GUI in between; that is the trade.
- **A brand-new shot is the exception.** A change that adds a shot to the manifest captures it in the same session with `scripts/screenshots/refresh.sh <new-id>`, because a doc cannot embed an image that doesn't exist and the wiring check requires both theme files on disk. Only *re*shoots are deferred.

The wiring check (`scripts/screenshots/wiring-check.py`, a `run-tests.sh` gate) fails on a queued id that names no manifest shot and no slide group, so the queue can't rot. The full screenshot system (manifest, harness, determinism knobs, embedding convention) is documented in [`scripts/screenshots/README.md`](scripts/screenshots/README.md).

## No Persisted Vectors or MLPs (CRITICAL)

**Embeddings and trained model weights are in-memory artifacts only.** Never serialize them to disk, to `data/settings.json`, to detector JSON files, or to any other persistent store. Origins are the canonical persisted form: the system rederives `origin → file → embedding → detector head` on demand.

This rule applies to all detector code:

- Detector JSON files store `LabeledElement`s with origin info, never embeddings or model weights.
- In-memory caches are fine and encouraged: `DetectorContext.label_embeddings`, `DetectorContext.model`, etc.: they live for the lifetime of the process and are repopulated from origins on the next start.
- New features that cache vectors must use a process-scoped data structure (e.g. a field on `DetectorContext`), not a file or settings key.
- Embedder version drift is impossible by construction because every load resolves+re-embeds against the active embedder.

The single exception is **dataset pickle files**, which are by design a snapshot of media + their embeddings; they ARE the dataset, not a cache. That exception extends to **derived caches of a pickle's own contents, written beside that pickle** — today the `<stem>.embmat.npy` / `<stem>.embids.npy` embedding-matrix sidecar (`vtscore/embedding/matrix.py`) and the coverage atlas cached inside the pickle itself. These put nothing on disk that the pickle does not already hold durably, and they only qualify when all four hold: the payload is a pure function of the pickle's medias, it is validated against the live id set on read (a mismatch rebuilds rather than being adopted), it is swept with the pickle by `registry.unregister_dataset`, and losing it costs only time. A cache that fails any of those is a persisted vector, not a derived cache.

If a feature seems to require persisting a vector or a trained head, push back: either re-derive on demand, or change the design.

## The Eval Default Arm IS the App (CRITICAL)

`vtscore.eval` exists to measure **deviations** from the shipped algorithm, which only means something if its **default arm** *is* the shipped algorithm. When the app moves and the harness doesn't, every later experiment silently measures a detector nobody uses. So: **an app-side algorithm change is not finished until the eval framework has caught up.** The full design (delegated / ported / default-resolution mirrors, gate output, reason codes) is in [`docs/EVAL.md`](docs/EVAL.md#the-eval-default-arm-is-the-app); the rules:

- **Prefer delegation** (the harness calling the app's function, as `MaxPatchStyle` does) over copying, every time — it is the only fix that can't rot. The two kinds of code that can't delegate are **ported** app logic (unreachable TypeScript, or lock-guarded interactive caches) and **default resolution** (`style=None`, `blend_schedule=None`, … resolving to the app's current default).
- **The gate:** `scripts/gates/check-eval-app-sync.py` (a `./run-tests.sh` gate) digests both sides of every mirror. `app-changed` means reconcile the harness copy to the app; `harness-changed` means re-read the two against each other (#2923 drifted that way with the gate green). After reconciling — or confirming nothing is owed — re-pin with `python scripts/gates/check-eval-app-sync.py --update`. **Re-pinning without looking at the other side defeats the entire gate.**
- **When you add a new mirror** (any new place the harness copies app logic or tracks an app default), add a `Mirror(...)` entry to `MIRRORS` in that script and run `--update`. Record intentional differences in `divergence=`; a `ported` mirror may never opt out of the harness pin via `no_harness_pin=`; list every top-level name a multi-symbol harness side spans (`file.py::GOOD_TARGET,BAD_TARGET`).

Named experiment arms (`whole_image`, `max_patch_hac`, …) are supposed to differ and are out of scope; this rule is about the **default** arm only.

## Fix All Errors (CRITICAL)

When you run a build, typecheck, linter, or test suite, **fix every error and failure you see; not only the ones you introduced**. Do not dismiss errors as "pre-existing", "unrelated to my change", or "not my fault" and move on. Do not announce them and ask the user to triage. The user does not want to scan your output for problems you decided to ignore.

This applies to:
- TypeScript errors from `tsc` / `npm run build:prod` (including in `*.spec.ts` files).
- Frontend unit-test failures from the Vitest suite (`cd frontend && npm run test:ci`, also run by `./run-tests.sh` and `./run-tests.sh frontend`).
- Angular build warnings of any kind, including `anyComponentStyle` budget warnings (e.g. `▲ [WARNING] ... exceeded maximum budget`). `run-tests.sh` treats every `▲ [WARNING]` line from `build:prod` as a hard test failure, so do not just bump budgets to silence them: fix the underlying bloat (split the component, extract shared styles, or remove dead rules). Bumping a budget is only acceptable when the size is genuinely justified, and requires the user's explicit approval.
- Python test failures from `./run-tests.sh` and `pytest` runs.
- Linter errors from `ruff check` (including the flake8-bandit `S` ruleset), formatting drift from `ruff format --check`, typos from `codespell`, documentation drift from `scripts/gates/check-docs.py`, dependency issues from `deptry`, known CVEs from `pip-audit`, type errors from `pyright`, and OpenAPI snapshot drift. All of these are `./run-tests.sh` gates (the linters, snapshot checks and pyright run before pytest and stop the run on failure; pip-audit runs alongside it). There is no CI backstop (the repo's three GitHub Actions workflows only publish and gate nothing), so `./run-tests.sh` is the source of truth — do not push a change without running it.
- Any other diagnostics surfaced by tooling you invoke.

If a failure is genuinely outside the scope of the current task (e.g. a flaky network test, a failure in unrelated infrastructure you cannot reproduce), explicitly call it out in your end-of-turn summary with one sentence explaining why you did not fix it, and file an issue for it unless one already exists (see "Work still owed goes in an issue" above). The default is **fix it**; skipping requires justification.

## Nested-modal back buttons (Back vs Cancel)

Any modal that switches between an outer view and an inner view (importer picker → importer form, exporter picker → exporter form, new-detector → media picker, etc.) **must** render a left-aligned back chevron at the top of the inner view so the user can return to the outer view without dismissing the modal. The standard markup is:

```html
<button class="btn btn--secondary btn--sm back-btn" (click)="back()" title="Return to ...">&larr; Back</button>
```

The `.back-btn` rule in `frontend/src/scss/_components.scss` provides the shared styling (`align-self: flex-start`, smaller font, tighter padding). Do not introduce a new variant class, a chevron icon component, or a right-aligned placement; keep the `&larr; Back` text label and the existing class combination.

**Back vs Cancel; these are not interchangeable.** Pick the word that matches the actual semantic:

- **`&larr; Back`** (top-left of the inner view, via `.back-btn`) means *navigate to the previous view*. It returns the user to where they came from (the outer view of the same modal, or the parent modal that opened this one), without committing the current step. Use it for any retreat action, including in child modals like `vt-clipper-chooser` that are opened from a parent modal: from the user's POV they are "going back" to the parent, so the affordance reads as Back even though the implementation dismisses a separate dialog.
- **`Cancel`** (in the footer alongside the primary action) means *abandon the entire dialog*. Use it only at the leaves of a flow, where the alternative to the primary action is to throw the whole thing away: typically the outermost view of a top-level modal (the importer/exporter picker, the new-detector main form, etc.).

A flow can legitimately carry both: a nested view shows `← Back` at the top to step back one view, while the outer view's footer shows `Cancel` to dismiss the whole modal. What it should *not* do is use the word "Cancel" for an action that is really navigation back to a parent view.

**Persistent-tab pickers are an intentional exception.** The rule's trigger is a view that *replaces* its outer view (the picker vanishes, the form takes over). A picker whose tabs stay **persistently visible** above the selected form never hides an outer view, so there is nothing to navigate "back" to and no `.back-btn` belongs on it — the user switches by clicking another tab. The Add-Dataset importer picker (`vt-source-picker`'s `.tab-bar` + `.importer-subtab-bar`) is the canonical case, and its divergence from the New-detector › Trained flow's picker→form→`← Back` shape is by design: do not add a `.back-btn` to a persistent-tab picker to "align" it. (Its footer `Cancel` is still correct — it dismisses the whole modal, per the Back-vs-Cancel rule above.)

## Commands

- **Run tests (CPU, fast)**: `./run-tests.sh` (every gate listed in [`docs/TESTING.md`](docs/TESTING.md#what-run-testssh-gates), then pytest; mandatory before pushing)
- **Run tests by group**: `./run-tests.sh core`, or several at once, `./run-tests.sh core sorting api` (see [Test Groups](docs/TESTING.md#test-groups)). A group run keeps the cheap gates but **skips the heavy whole-repo gates**, and says so: pyright, pip-audit and the vulture whitelist check (`VTSEARCH_FULL_GATES=1` forces those three back on), and the frontend gates unless the group is `core` (build + `npm audit`, when the branch touches `frontend/`) or `frontend` (build, audit and Vitest, always). Run it *before* the full run, not instead of it: the pass cache carries its passes into the full run
- **Slides-, markdown- and frontend-only changes**: `./run-tests.sh slides` (~8s) is the *complete* gate for a change confined to `slides/`, and refuses to run if the branch touches anything else. A bare `./run-tests.sh` on a branch that changes only tracked markdown, or nothing outside `frontend/`, narrows itself and announces what it skipped (`./run-tests.sh docs` asserts markdown-only explicitly); `VTSEARCH_FULL_GATES=1` opts out. What each path runs: [Test Groups](docs/TESTING.md#test-groups)
- **Re-run everything from scratch**: `VTSEARCH_FRESH=1 ./run-tests.sh` ignores the incremental-pyright baseline and the pass cache. For a suspected flaky pass or a release gate, not as a habit: on an identical tree a recorded pass is as good as a repeated one
- **Run tests with coverage**: `VTSEARCH_COVERAGE=1 ./run-tests.sh` (opt-in; adds ~10-20% overhead)
- **Run tests with extra args**: `./run-tests.sh core -- -x --tb=long` (args after `--` go to pytest)
- **Run library-tier tests only (Flask-blocked)**: `./run-tests.sh vtscore-clean` (runs `tests_lib/` with `flask`, `werkzeug` and `flask_smorest` refused at import; skips every other gate)
- **Run tests (CPU, full)**: `bash .claude/hooks/ensure-test-deps.sh && python -m pytest tests/ tests_lib/ -q --tb=short -m 'not gpu'`
- **Run slow tests only**: `python -m pytest tests/ tests_lib/ -q --tb=short -m slow` (both trees — the slow tests do not all live under `tests/`; see [Test Markers](docs/TESTING.md#test-markers))
- **Run GPU tests**: `python -m pytest tests_lib/gpu/test_gpu.py -q --tb=short -m gpu` (requires CUDA GPU; downloads models on first run)
- **Run all tests (CPU + GPU)**: `python -m pytest tests/ tests_lib/ -q --tb=short -m ''`
- **Regenerate the OpenAPI snapshot** (after any route/schema change, to clear the drift gate): `cd frontend && npm run regenerate-openapi-snapshot` (equivalently `python scripts/gates/dump_openapi.py > frontend/openapi.json`), then commit `frontend/openapi.json`
- **Start app**: `bash .claude/hooks/ensure-test-deps.sh && python app.py` (or `python app.py --local` for dev)
- **CLI autodetect**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json>` (add `--exporter <name> …` to write results, or `--importer <name> …` in place of `--dataset`; see [`docs/CLI.md`](docs/CLI.md))
- **Check eval/app sync**: `python scripts/gates/check-eval-app-sync.py` (also a `./run-tests.sh` gate; re-pin with `--update` after reconciling the harness)
- **Check for a stale tree**: `python scripts/gates/check-phantom-base.py` (also the first `./run-tests.sh` gate; fails when the branch deletes files it never created, or carries none of what `dev` gained across two or more consecutive commits. Override a deliberate deletion with `VTSEARCH_ALLOW_DELETIONS=1`, a deliberate revert with `VTSEARCH_ALLOW_REVERTS=1`)
- **Check documentation**: `python scripts/gates/check-docs.py` (a `./run-tests.sh` gate; what it validates is in [`docs/TESTING.md`](docs/TESTING.md#the-stages). Pure invariants — nothing to re-pin. Fix the doc, or add an allowlist entry with a reason if the path is runtime-generated or a deliberately fictional example)
- **Check extension docs**: `python scripts/gates/check-extension-docs.py` (a `./run-tests.sh` gate: the extension guides must name only members that exist on the plugin ABCs. Fix the doc, or register a new contract section in `SECTIONS`)
- **Regenerate doc inventories**: `python scripts/gates/gen-docs-inventories.py` (fills the `<!-- BEGIN GENERATED: ... -->` regions in the docs from the live registries — embedders, plugin families, demo datasets; `--check` is a `./run-tests.sh` gate, so registry changes require rerunning this and committing the result)
- **Install deps**: `bash scripts/install.sh` (auto-detects CPU vs GPU; pass `cpu`/`gpu` to force, or a `cuXYZ` tag to override the GPU wheel, e.g. `bash scripts/install.sh cu126`)
- **Build frontend**: `cd frontend && npm install && npm run build:prod` (builds Angular app to `static/`)
- **Frontend dev server**: `cd frontend && npm start` (proxies `/api/*` to Flask at localhost:5000)
- **Frontend audit**: `python scripts/gates/npm-audit-gate.py` (the `./run-tests.sh` gate: `npm audit` over `frontend/`, minus the advisories in its `WAIVERS` table, which have no patched release anywhere; bare `cd frontend && npm audit` still lists those)
- **Frontend unit tests**: `cd frontend && npm run test:ci` (headless Vitest + jsdom, no browser; also run by `./run-tests.sh` and `./run-tests.sh frontend`). `npm test` is the watch-mode variant.
- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Spell check**: `codespell --toml pyproject.toml`
- **Dependency check**: `python -m deptry .`
- **Dead code audit** (manual, pre-release): `python scripts/gates/vulture-audit.py` — the script owns the invocation. The **findings** are not a gate (see `docs/RELEASE.md` step 1 for the triage rules).
- **Check the vulture whitelist**: `python scripts/gates/vulture-audit.py --check-whitelist` (a `./run-tests.sh` lane; fails when an entry in `.vulture-whitelist.py` suppresses no finding). So adding an entry is not free: whitelist what is genuinely reflective, delete what is genuinely dead.

## Testing reference

The gate chain, the test-group table, markers, fixture isolation and the flaky-test patterns live in [`docs/TESTING.md`](docs/TESTING.md). Read it before adding a test or a gate. The rules it spells out, in brief:

- **A full `./run-tests.sh` is mandatory before pushing.** A group run skips the whole-repo gates; the only exceptions are the self-policing `slides` and `docs` scopes. Keep the gate table in `docs/TESTING.md` in step with `run-tests.sh` in the same commit (`scripts/gates/check-docs.py` fails when their gate names disagree).
- **Put a test in `tests_lib/` if it touches no app-tier module**, otherwise in `tests/`. In `tests_lib/`, use the folder of the `vtscore` package it exercises: the folders mirror `vtscore/`, and `LIB_GROUPS` in `tests_shared/pytest_plumbing.py` maps each to its group. `tests_lib/` must be import-clean of Flask and of `vtsearch` entirely; add a library seam rather than reaching across the tier. Import helpers tier-qualified (`from tests.helpers import …` / `from tests_lib.helpers import …`).
- **Do not add per-file autouse reset fixtures or end-of-test cleanup** — the conftests reset all global state before every test. Add a new library-tier global's reset to `tests_shared/state_reset.py`. Only `medias` needs save/restore (try/finally) if a test empties it.
- **A test that reads a repo doc** goes in `tests_lib/meta/` or is registered in `tests_shared/markdown_surface.py`.
- **No flakiness:** seed every RNG (`np.random.default_rng(42)`); synchronise threads with `threading.Event`, never `time.sleep()`; simulate cancellable work with an unbounded loop that exits only on cancellation, never a bounded one.
- **Naming a group re-opens `slow` and `gpu`**; spell the filter out after `--` to restore the default exclusions.

## Test Workflow (IMPORTANT)

Testing can crash the session. To avoid losing work, follow this workflow:

1. **Commit and push before running tests.** Before running `pytest` or any test command, commit all current changes and push to your working branch. Use a message like `"WIP: pre-test checkpoint"` if the work isn't finalized yet. **Re-rendered images are the exception:** leave screenshots and slide figures out of a WIP checkpoint and commit them once, when final. PRs merge without squashing, so every commit carrying an image keeps it in history for good, and losing an uncommitted render to a crash costs only a re-render.
2. **Start the run in the foreground with the maximum timeout, and never *launch* it with `run_in_background`.** The test command has a slow startup phase: `ensure-test-deps.sh` installs dependencies (~1-2 min on first run), then `conftest.py` imports `app.py` and generates test media/embeddings before any tests execute. There may be no output for several minutes; this is normal, and is not a sign that output capture is broken.

   **Pass `600000` ms (10 minutes) — the Bash tool's maximum.** The first full run of a session is roughly pyright plus pytest, about 10 minutes warm; a re-run after a small fix is roughly pytest alone, and a full run after a group run skips what that group passed. A cold container adds ~3 minutes of dep install, and a cooled page cache can make the first run several times slower ([`docs/TESTING.md`](docs/TESTING.md#what-run-testssh-gates) has the measurements). A run will often pass the cap — that is fine: the harness then moves the run to the background rather than killing it; wait for the completion notification and read the output file it names. What matters is that you *started* it in the foreground so the harness tracks it.

   Two consequences worth knowing before you run it:
   - **Do not pipe the run through `tail`/`grep`.** If the harness backgrounds a pipeline, nothing flushes until the whole pipeline ends, so the output file sits empty and you can't watch progress. Run the script bare and read the tail of the output file afterwards.
   - **A run that outlives the tool's cap is not a timeout.** The script has its own 30-minute wall-clock cap (`VTSEARCH_TEST_TIMEOUT`) and prints a distinctive `TESTS TIMED OUT` banner when *it* fires. Absent that banner, the run is still healthy. To stay well inside 10 minutes, run one group at a time (typically well under a minute warm; see Commands) before the full run.
3. **If tests fail and fixes are needed**, make the fixes, then commit and push again before re-running tests.
4. **Repeat** until tests pass. Every cycle of fixes should be committed and pushed before the next test run.

This ensures work is recoverable if the session crashes during a test run.

## Reading Test Results (IMPORTANT)

A `./run-tests.sh` run prints its verdict as its very last output, in a `====`-bordered block:
- `RUN PASSED (all gates green; pytest summary above)` → all good (the `slides`, markdown-only and frontend-only runs print their own `RUN PASSED (<scope>-only; …)` variant, and a full run on a branch that touches no `frontend/` path says `RUN PASSED (all gates green; frontend build, npm audit and Vitest skipped: …)`, which is still the full gate — those three had nothing to judge)
- `RUN PASSED, EXCEPT N GATE(S) THAT DID NOT RUN:` + a list → everything that ran is green, but this is **not** the full gate (a group run's skipped pyright/pip-audit/vulture, or frontend gates skipped for missing `node_modules`)
- `RUN FAILED: pytest, pip-audit` → those gates failed; each failing gate's `TESTS BLOCKED` banner and log tail were printed above

A failure in a serial stage (the cheap gates, the frontend build, pyright) stops the run early with a `TESTS BLOCKED: <reason>` block and no verdict banner.

Because the free lanes (pip-audit, the vulture whitelist check, `npm audit`, Vitest) report after pytest, pytest's own summary block (`ALL <n> TESTS PASSED (<k> skipped, total: <t>)` / `TESTS FAILED: 2 failed, ...`) sits *above* the lane report in a full run — it is still the place to read pytest's counts, and it is the last output of a bare `pytest` invocation. A `Pass cache: N tests not re-run …` line, or the headline `ALL N TESTS ALREADY PASSED ON THIS TREE (nothing re-run)`, means an earlier run on this identical tree already passed those tests: the run is green ([pass cache](docs/TESTING.md#two-records-a-run-leaves-for-the-next-one); `VTSEARCH_FRESH=1` re-runs them).

**ONLY look at these summary blocks** (bordered by `====` lines) to determine pass/fail. Many test names contain the word "error" (e.g., `test_memory_errors.py`, `TestErrorResponseFormat`). These test **error-handling behavior**; they are not failures.

**Do NOT scan test names or output for the word "error" to detect failures.** A line like:
```
tests/datasets/test_memory_errors.py::TestPickleMemoryError::test_importer_background_oom_reports_error PASSED
```
means the test **passed**; "error" is part of its name.

## Environment Notes (Claude Code on the web)

- **Chromium *is* available — check before assuming it isn't.** The cloud container ships a Playwright chromium under `PLAYWRIGHT_BROWSERS_PATH` (`/opt/pw-browsers`). When a question is "what does the browser actually do here?", go and look rather than reasoning from the spec (#2898 shipped two rounds of frontend fixes without anyone opening a tab).
  - The container's chromium revision does **not** necessarily match the one the `playwright` npm pin wants, so a bare `chromium.launch()` can fail with `Executable doesn't exist` even though a perfectly good browser is present. Do **not** run `npx playwright install` (the environment sets `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1`). Use `scripts/screenshots/launch.mjs`'s `launchChromium()`, which tries the pinned build and falls back to whatever is in the browsers directory; `CHROMIUM_PATH` overrides both.
  - This does not change the **test suite**, which stays deliberately browser-free: the frontend unit suite runs on **Vitest + jsdom** (the Angular 21 `@angular/build:unit-test` builder), and the Python backend tests (`./run-tests.sh`) never needed a browser. Keep it that way — a browser-dependent gate would be slow and machine-sensitive. The browser is for *investigation* and for the screenshot harness, not for `run-tests.sh`.
  - A jsdom stub is not a browser. `window.open` is the cautionary example: the Vitest specs returned a truthy fake handle, so they cheerfully passed against code that could never work in Chrome. When a behaviour depends on real browser semantics (popups, user activation, navigation, focus), verify it in chromium *as well as* in the unit suite.

## More docs

- [`docs/SETUP.md`](docs/SETUP.md) — prerequisites, virtualenv, dependency install, frontend build, Docker, running the tests. The SLURM walkthrough lives beside its scripts, in [`scripts/slurm/README.md`](scripts/slurm/README.md).
- [`docs/TESTING.md`](docs/TESTING.md) — the `run-tests.sh` gate chain, test groups and markers, fixture isolation, flaky-test patterns.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — key concepts (media items, votes, media types, origins), directory map, dependency graph, plugin systems, state management (multi-dataset / multi-detector contexts, proxies, `X-Dataset-Id` / `X-Detector-Id` headers), auth, origin tracking.
- [`docs/FRONTEND.md`](docs/FRONTEND.md) — Angular SPA architecture: feature-area boundaries, the service layer, the **zoneless change-detection rules**, active dataset/detector propagation, the generated OpenAPI client, component/modal conventions. Read this before changing frontend state or reactivity.
- [`docs/API.md`](docs/API.md) and `docs/api/*.md` — REST API reference.
- [`docs/CLI.md`](docs/CLI.md) — CLI flags and autodetect workflow.
- [`docs/ML.md`](docs/ML.md) — training/scoring details.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — production/offline deployment, env vars, data directory.
- [`docs/SETTINGS.md`](docs/SETTINGS.md) — the server (`data/settings.json`) and per-user (`user_settings.json`) settings files, key by key.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) — symptoms and fixes, including GPU driver / CUDA / cuML installs.
- [`docs/EVAL.md`](docs/EVAL.md) — the `vtscore.eval` harness and calibration experiments.
- [`docs/EXTENDING.md`](docs/EXTENDING.md) + [`docs/EXTENDING-plugins.md`](docs/EXTENDING-plugins.md) + [`docs/EXTENDING-media.md`](docs/EXTENDING-media.md) — how to add plugins.
- [`vtscore/docs/README.md`](vtscore/docs/README.md) — the library tier's own doc set (quickstart, concepts, per-package reference, tutorials, FAQ).
- [`slides/STYLE.md`](slides/STYLE.md) — house rules for every slide deck (no running footer, real subscripts, colour reserved for meaning, the 20px type floor, the opening outline); [`slides/README.md`](slides/README.md) is the build mechanics.
- [`docs/plans/`](docs/plans/) — future-work design docs (open/proposed; shipped work is pruned out, not archived); check here before adding a "Phase N" feature.
- [`docs/ISSUES.md`](docs/ISSUES.md) — the reasons behind the issue-lifecycle rules: what each label and the assignee mean, the tracker views they keep honest, how the hook and the backstop script enforce them, and the incidents behind each.
- [`docs/RELEASE.md`](docs/RELEASE.md) — the `dev` → `main` release runbook (the procedure the Dev2Main Routine follows: vulture audit, release summary, punch-card refresh, reshoot-queue drain, changelog dating, release PR, issue close-out, label audit, experiment-queue sweep, plan-pointer prune, merge).
- [`docs/branch-protection.md`](docs/branch-protection.md) — who can land on `main` vs `dev`, the branch protection in force on each, and why `dev` survives the Dev2Main release PR now that merged head branches are auto-deleted.
- [`docs/style-guide.md`](docs/style-guide.md) — frontend SCSS conventions (the styling half of [`docs/FRONTEND.md`](docs/FRONTEND.md)).
- [`CHANGELOG.md`](CHANGELOG.md) — curated record of notable user-facing app changes ([`vtscore/CHANGELOG.md`](vtscore/CHANGELOG.md) is the library's).
