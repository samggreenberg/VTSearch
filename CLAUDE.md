# VTSearch

Trainable media search tool. Searches collections of audio, images, text, video, and documents using a **detector**: a small trained ranker that scores each item by how well it matches. Two ways to search: **train a new detector** (vote good/bad on a handful of items; a linear (logistic) head learns to rank the rest) or **use an existing detector** (saved or imported). Trained detectors are reusable across compatible datasets. Text queries (LAION-CLAP, SigLIP, X-CLIP, E5 embeddings) seed either flow or work as a quick stand-alone search. Flask + Angular + PyTorch.

Architecture, state model, plugin systems, auth, and the directory map all live in **`docs/ARCHITECTURE.md`**. This file holds the testing rules and the policy/gotchas that must be in context every turn.

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

## Branch Policy (CRITICAL)

- **Always base work on `dev`.** The `.claude/hooks/session-start.sh` SessionStart hook fetches `origin --prune` and then lands the working branch on `origin/dev` automatically in remote sessions. The harness cuts the working branch off `main` (the GitHub default), so this is required to pick up work already merged to `dev`. The GitHub default stays `main` so new users land on the stable branch: `dev` is Claude's starting point, not the public default.
  - **The hook can *hard-reset*, not only rebase — read its output.** It picks one of four outcomes and says which: `already includes origin/dev; nothing to do`; **hard-reset** to `origin/dev` (either because the branch has no `origin/<branch>` counterpart — a fresh branch the harness just cut off `main`, whose unique commits are all inherited `main`-only history carrying no Claude work — or because `git cherry` shows every unique commit is patch-equivalent to one already on `dev`); **rebase** onto `origin/dev` (the branch is in sync with its origin counterpart and carries genuinely new pushed commits); or a **skip**. A hard-reset discards the branch's prior commits by design; that is expected at session start and nothing of yours is lost, but do not assume commits you saw in `git log` before the hook ran are still there.
  - If the hook prints `‼ session-start: DID NOT rebase onto origin/dev` (dirty tree, detached HEAD, fetch failed, a reset/rebase that failed, or a local branch that differs from its pushed origin counterpart), run `git fetch origin --prune && git rebase origin/dev` yourself before making any changes.
- **All pull requests MUST target `dev`**, never `main`.
- **Claude must NEVER open a PR that merges into `main`.** The `main` branch is protected and only updated by human maintainers.
- When creating a PR, always use `--base dev` (e.g., `gh pr create --base dev ...` or the equivalent MCP tool parameter).
- If your feature branch was forked from `main` instead of `dev`, rebase or merge onto `dev` before opening a PR.

## Git Fetch Hygiene

Before comparing branches (`git log a..b`, `git diff a...b`, etc.), always run `git fetch origin --prune` first. Do **not** trust `origin/<branch>` refs after a partial fetch like `git fetch origin main`; that only updates the branch you named, leaving other remote-tracking refs stale and producing misleading diffs.

## Auto-PR

When you're done with your changes, open a PR targeting `dev`. Do not ask; just create it. Always pass `base=dev` explicitly (the GitHub PR-creation URL printed by `git push` defaults to `main`).

This standing instruction **is** the explicit request that the remote-environment harness rule ("do not create a pull request unless the user explicitly asks for one") defers to. The harness rule only suppresses *unsolicited* PRs; a durable, repo-committed instruction to auto-open one satisfies its "unless the user explicitly asks" carve-out. So the two do not conflict: in this repo, finishing your changes is your cue to open the PR (base `dev`) without further prompting.

## Linking a fix PR to its GitHub issue

When your change resolves a GitHub issue, **link the PR back to that issue** so the two are connected on GitHub:

- Put a closing keyword in the **PR body**: `Closes #N` (or `Fixes #N` / `Resolves #N`). This populates GitHub's "Linked issues" sidebar and the issue timeline. Reference every issue the PR resolves; use one keyword per issue (`Closes #12, closes #15`), not a comma-list after a single `Closes`.
- If the PR only *partly* addresses an issue, use a non-closing reference instead — `Refs #N` / `Part of #N` — so it links without implying the issue is done.
- Also drop a one-line comment on the issue pointing at the PR (e.g. "Addressed in #M"), so someone reading the issue sees the fix even before it merges.

**The PR keyword is the load-bearing signal, and the issue comment must agree with it.** The `dev`→`main` sweep (step 6 of `docs/RELEASE.md`) decides what to close by reading **PR bodies**, not issue comments — so the keyword you pick is what determines whether the issue ever gets closed. `Refs #N` on a PR that actually finishes the issue is not a harmless understatement: the sweep reads it as "still partial", skips the issue, and *nothing revisits it in a later release*. The issue stays open forever while its fix is live in `main`.

So make the two statements match, and default to `Closes` whenever the PR finishes the issue:

| The PR… | PR body | Issue comment |
|---|---|---|
| fully resolves the issue | `Closes #N` | `Addressed in #M` |
| does part of it, more is still owed | `Refs #N` / `Part of #N` | `Partially addressed in #M — still open: <what's left>` |

Use `Refs` only when you can name the work that remains on **that issue**. Rescoping counts as finishing: if you narrowed the issue's body and the PR does all of what's left, that's `Closes`. Deferred scope that has been moved into a plan file or a separate issue is no longer owed by this issue, so it doesn't make the PR partial either.

If you catch yourself writing "Addressed in #M" on the issue while the PR body says `Refs`, one of the two is wrong — fix it before merging.

**Do not close the issue yourself.** The closing keyword will **not** auto-close the issue on merge, because GitHub only auto-closes keyword-linked issues when the PR merges into the **default branch** (`main`), and our PRs target **`dev`**. That is intentional: an issue stays open while its fix lives only on `dev`, and is closed only once the fix reaches `main` (i.e. is actually shipped to users). The `dev`→`main` merge Routine is what sweeps the now-shipped issues closed (with `state_reason: completed`); a per-fix Claude session must not close the issue early. So: link and comment, but leave the issue open.

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

**Dismissing an issue as unwarranted:** close it as `not_planned` with a one-line comment explaining why. If any plan file points at it (grep `docs/plans/` for `#<number>`), prune that pointer line in the same motion. Because plans carry only pointers — not bodies — this is always a trivial one-line deletion, whether the issue was dismissed or merged. (The Dev2Main Routine reconciles this automatically when it sweeps issues; a manual close should do the same pointer-prune by hand.)

This is why closing an issue by hand didn't previously "trickle back": the item was **duplicated as a body** in the plan instead of **referenced as a pointer.** Store it once, point at it from anywhere else, and dismissal is just deleting the pointer.

## Label every issue you file (CRITICAL)

**Every GitHub issue you create must carry the `claude` label**, and the `experiment` label when it applies. Apply them at creation time (`labels: ["claude", …]`), not as a follow-up edit. If a label is missing from the repo, applying it via the issues API creates it automatically — do not skip a label because it doesn't exist yet.

A third label, **`dev`**, is a release *status* rather than something you choose when filing — it is applied and removed by the release machinery. Read its section below before closing any issue.

### `claude` — who filed it

`claude` means **this issue was written by Claude, not by a human.** It is not a topic tag and has nothing to do with what the issue is about (nearly every issue here concerns Claude-adjacent work; that is never why the label goes on).

The label exists because **authorship is otherwise invisible.** Claude sessions file issues through the repo owner's GitHub account, so a Claude-written issue and a human-written one have the *same author* — there is no `author:` query that separates them. The label is the only signal, which is why the burden sits on Claude:

- **Claude labels its own issues.** Humans file issues without ceremony and are never asked to remember a `human` tag.
- **Human-filed issues are the ones *without* the label:** `is:issue is:open -label:claude`.

That asymmetry is the whole design. It only works if Claude is exhaustive: a Claude-filed issue that slips through unlabeled doesn't just lose a tag, it **silently contaminates the human-issue view** — the query that a human uses to find their own thinking now returns Claude's. Never apply `claude` to an issue a human wrote, and never omit it from one you wrote.

### `experiment` — does closing it require a run?

`experiment` means **this issue cannot be resolved without running an experiment** — a GRID/SLURM sweep, an eval study, a calibration run. It marks a *gate*, not a topic: the test is "could a competent implementer close this from a laptop with the test suite, or do they need measured results first?"

- Apply it to: new eval arms, sweeps and re-runs, calibration rows that must be measured, "is X better than Y?" questions, and research ideas whose first real step is a measurement.
- Do **not** apply it to: bug fixes, docs, refactors, plumbing, or code changes whose spec is already measured and decided.
- It is orthogonal to `claude` — a human-filed research idea gets `experiment` alone; a Claude-filed sweep gets both.

The point is scheduling: `label:experiment` is the queue of work that needs machine time booked, and `-label:experiment` is what can be picked up right now. See the `grid-experiments` skill for how those runs are actually launched.

### `dev` — fixed on `dev`, not yet on `main`

`dev` means **the fix has merged to `dev` but has not shipped to `main`.** Unlike `claude` and `experiment`, it is **not applied at creation time** and is never something you decide when filing — it is a *status* the release machinery maintains.

It exists to make an otherwise invisible state visible. A fix PR targets `dev`, and GitHub only auto-closes a keyword-linked issue when the PR merges into the **default** branch (`main`), so a fixed issue stays open until the release sweep closes it. Without this label there is no way to tell "waiting for the next release" apart from "nobody has started it". The awaiting-release view is:

```
is:issue is:open label:dev
```

**The label is transient, not a historical fact.** It goes on when the fix merges to `dev`, and comes off in the same write that closes the issue (`docs/RELEASE.md` step 6). A closed issue must never carry it: by then the fix is on `main` too, so the label would assert something false, and a reopened issue would wrongly appear in the awaiting-release view. `is:open` already filters the closed pile for free, so letting it linger would buy nothing.

Two rules bind you directly:

- **Closing an issue strips `dev`.** Pass `labels` explicitly on a `completed` close. Note that `labels` *replaces* the whole set, so list every label the issue keeps (`claude`, `experiment`, …) and omit `dev` — passing `[]` would wipe the rest. A `PreToolUse` hook blocks a close that keeps the label or omits the array.
- **Do not apply it by hand from a fix session.** When you open a fix PR the merge hasn't happened yet, so the issue is not on `dev` and labeling it would be a lie for as long as the PR sits unmerged. Applying it is `scripts/reconcile-dev-labels.py`'s job.

That script encodes `docs/RELEASE.md` step 6's resolution logic — closing keywords vs. `Refs`, `Partially addressed in #M` vs. `Addressed in #M`, and the ambiguity of a comment posted *after* a fix pointer — and reconciles the label in both directions. It is a pure function from data to plan: the GitHub REST API is unreachable from a Claude session (`GITHUB_TOKEN` is present but 403s, since GitHub access is intermediated by the MCP server), so gather the PR and issue data with the `github` MCP tools and pipe it in. See `docs/RELEASE.md` for the recipe.

**A comment after the fix pointer is never guessed at.** If someone comments below an `Addressed in #M` pointer, the script reports the issue as needing review rather than tagging or skipping it. The later comment might be a maintainer saying "thanks" or the reporter saying the fix doesn't work; tagging would bury a dispute, and skipping would silently drop the issue out of the awaiting-release view. Ambiguity gets surfaced, not resolved by a coin flip.

## Recommend a Claude model in every issue you file

**Every GitHub issue you create must include a recommended Claude model** for whoever picks it up, sized to the work. This lets a task be routed to the cheapest model that will do it well — a Haiku-tier mechanical edit shouldn't burn Opus, and a regression-prone refactor shouldn't be handed to a model that will botch it.

**Capability ladder (cheapest/weakest → most capable/most expensive): Haiku 4.5 → Sonnet 5 → Opus 4.8 → Fable 5.** Fable 5 is Anthropic's *most* capable model (and its most expensive), reserved for the hardest, longest-horizon work — it is **not** a cheap rote tier. Haiku 4.5 is the fast, cheap tier for mechanical edits. "Step up" always means moving toward Fable; "step down" toward Haiku.

- Add a bolded line near the top of the body (right after the difficulty/summary), e.g. `**Recommended Claude model: Sonnet 5.**` — with a short clause on *why* when it isn't obvious.
- Size to the hardest part of the issue, not the average. Rough guide: **Haiku 4.5** for rote, schematic-/find-replace-shaped edits with a clear spec and no design judgment; **Sonnet 5** for normal feature/bugfix work with bounded reasoning; **Opus 4.8** for regression-prone refactors, subtle concurrency/reactivity, cross-cutting design, or anything where a wrong-but-plausible answer is costly; **Fable 5** only for the most demanding, long-horizon, or research-grade work that genuinely exceeds Opus. Use the current model names (Haiku 4.5, Sonnet 5, Opus 4.8, Fable 5).
- When one issue mixes tiers (a mechanical bulk plus one gnarly file), name the split: e.g. "Sonnet 5 for the bulk; Opus 4.8 for `foo.component.ts`."
- If a plan file's umbrella lists issue pointers, it's fine (not required) to append the recommended model in parentheses after each pointer title, as a routing hint.

This is a recommendation for the *implementer's* model choice; it has nothing to do with the model identifier you report when asked which model **you** are.

## Plan files (`docs/plans/`) track FUTURE work only

A plan file describes **work still owed**: a proposed feature, or the open parts of one in progress. Plans are **not an archive of completed work.** Git history and merged PRs are the record of what already landed; the plan is what someone reads to pick up what's left.

**When you ship, prune what you finished:**

- **Fully shipped, nothing left → delete the file.** Do not leave it behind marked "done" / "shipped" / "kept as reference." First fold any durable design rationale into `docs/ARCHITECTURE.md` / `docs/EXTENDING.md` (or their siblings) where permanent docs belong.
- **Partly shipped → delete the shipped narrative, keep only what's still owed.** Remove "What shipped" sections, resolved-finding catalogs, phase-by-phase ship logs, strikethrough-completed checklists, and completion dates. What remains is the open work.
- **Keep past context only when future work needs it.** If the remaining work can't be understood without some of what already shipped, keep a *short* "Background" note at the top — a paragraph, not a changelog. That is the single exception to "no records of past work."
- **Grep the source tree before deleting, not just `docs/plans/`.** Module docstrings and inline comments cite plan files by path (`See docs/plans/<name>.md`) far more often than other plan files do, and those citations are invisible if you only check for inbound *plan* pointers. Before deleting `docs/plans/<name>.md`, run `grep -rl 'docs/plans/<name>\.md' --include="*.py" --include="*.ts" --include="*.sh" --include="*.md" --include="*.json" --include="*.html" .` (adjust extensions to what the repo actually cites from) and fix every hit in the same commit: either repoint it at the permanent doc the rationale was folded into, or drop the pointer outright when the surrounding prose is already self-contained (the common case — most citations duplicate content the docstring already states, so the plan was never load-bearing there). A dangling `docs/plans/<name>.md` citation left in source is a documentation regression, not a harmless leftover: it is exactly the "why is this code shaped like this" pointer a future maintainer follows, and it now leads nowhere.

**Follow-ups go in the plan file, not the PR body.** When you identify deferred scope or known limitations, record them as open work in the relevant plan under `docs/plans/` (the one that scoped the feature). Do **not** stash them in the PR description as the only record — PRs close, get archived, and stop surfacing in normal discovery. The PR body describes what landed; the plan tracks what's still owed.

**Name plan items; never renumber them.** Identify each item in a plan by a stable, descriptive **name** (a bolded title — the items already carry one), not by its position in a contiguous `1., 2., 3.` list. A numbered list forces every deletion to renumber the survivors, and that renumber is a gratuitous merge conflict whenever two efforts ship different items at once. So write the open-work list as a plain **bulleted** list (`- **Name** — …`); refer to items by name in prose and PRs. If you inherit a numbered plan, treat the numbers as arbitrary stable labels: when you delete an item, **leave every surviving item's number exactly as it is** — gaps (`1, 3, 4`) are fine and expected, and are strictly better than a renumber. Better yet, drop the numbers to bullets as you touch the file. The only edit a shipped slice makes is deleting its own item; it never touches another item's label or number.

**Minimize churn to avoid merge conflicts.** Because there is no ship-status to update, a plan touched by parallel efforts changes far less than it used to. Keep it that way: when your effort ships a slice, make the **smallest edit that removes the work you finished** — delete the completed items (leaving the rest untouched, per "never renumber" above), don't reflow or restructure the surrounding prose, don't rewrite a status header into a ship narrative, don't append a completion log.

**Separate every item with a `<!-- item-sep -->` sentinel, and never delete a sentinel.** Deleting your own item is *not* enough to avoid a merge conflict on its own. Git merges at the granularity of diff hunks, not "items": when two parallel efforts delete **adjacent** items, their deletions abut with no surviving unchanged line between them, so git can't reconcile the two hunks and flags a conflict — even though each side only deleted. (Two efforts deleting items that are *far apart* merge cleanly, because the untouched items between them anchor the merge; adjacency is the trigger, not "same file.") The fix is a permanent, never-deleted separator line between every pair of items, so that whichever items get deleted, an unchanged sentinel always survives between any two deletions and gives git the common context it needs to auto-merge. So:

- Every plan item is preceded (or followed — pick one and be consistent within a file) by a lone `<!-- item-sep -->` line on its own, blank-line-separated from the bullets on either side. The sentinel renders as nothing.
- When you ship a slice, **delete only your item's own lines; leave the sentinels above and below it in place.** Deleting a sentinel re-creates the adjacency problem for the next pair.
- Empty runs of back-to-back sentinels (left behind by deleted items) are harmless and expected — sweep them opportunistically when you're editing the file for another reason, never as a churn-only commit that would itself conflict with in-flight deletions.
- A plan that predates this convention gets sentinels added the next time someone touches it; until then its parallel-deletion conflicts stay trivial (both sides only delete) and are resolved by taking both deletions.

If a plan fully ships and no follow-ups remain, deleting it (after absorbing any lasting design notes into the permanent docs) is the expected outcome, not an oversight.

## PR Activity Subscription (do not ask)

Never ask the user whether to subscribe to PR activity, and never call `subscribe_pr_activity`. The user does not want Claude to watch PRs or respond to review comments / CI. This overrides the default GitHub Integration instruction to offer PR subscription after creating a PR.

## Versioning (do NOT bump by hand)

`vtsearch.__version__` is the UTC timestamp of `HEAD`'s commit (ISO 8601, Z-terminated), computed from git at import time in `vtsearch/__init__.py`. There is no tracked version constant to bump; every commit on `dev` automatically becomes the new version, and parallel branches cannot collide on a hand-edited version line. Do not add a `VERSION` file, do not write a hand-bumped string into `vtsearch/__init__.py`, and do not include version bumps in feature PRs. For Docker images (where `.git` is excluded from the build context), the host passes `--build-arg VTSEARCH_VERSION=$(TZ=UTC git log -1 --format=%cd --date=format:%Y-%m-%dT%H:%M:%SZ HEAD)` and the Dockerfile bakes it into `vtsearch/_version.txt` (gitignored). If git is unavailable and the baked file is missing, the version falls back to `0.0.0-unknown`.

**The frontend bundle carries the same stamp.** `frontend/scripts/build-stamp.mjs` runs from the `prebuild` / `pretest` hooks and writes `frontend/src/app/generated/build-stamp.ts` (gitignored, beside the generated API client) with the value computed above — from `VTSEARCH_VERSION` if set, else git. At startup `BuildSkewService` compares it against `GET /api/version` and, on a mismatch, raises a non-dismissing toast; the Settings footer also grows a `⚠ bundle v …` chip beside the server version. This exists because `static/` is a gitignored build artifact, so `git pull && python app.py` leaves a new server serving an old SPA with nothing anywhere saying so — the failure mode that cost issue #2898 three round-trips. Nothing here is hand-edited; if you change how the Python version is derived, change `build-stamp.mjs` to match.

**`vtscore.__version__` is different.** The library uses independent semver, tracked as a hand-edited constant in `vtscore/__init__.py` (currently `0.1.0`). Bump it only when cutting an actual `vtscore` release, and add a matching entry to `vtscore/CHANGELOG.md`. Do *not* include `vtscore` version bumps in unrelated feature PRs. The two packages version independently because `vtsearch` is a continuously-deployed app (every commit = new version) while `vtscore` is meant for external consumers who expect stable, semver-tagged releases.

## Backwards Compatibility

Breaking backwards compatibility is acceptable; do not add shims, feature flags, legacy re-exports, or other compatibility layers to preserve old behavior. Just make the clean change. When a change does break backwards compatibility, mention it to the user so they're aware.

## Frontend Scope: Desktop Only

VTSearch is a desktop web app. **Do not design, implement, or test for mobile or narrow viewports.** No responsive breakpoints, no touch-targeted controls, no mobile-only layouts, no concerns about portrait orientation. If a design discussion raises "what about mobile?", the answer is "we don't care." When evaluating a layout, assume a standard desktop viewport and skip mobile considerations entirely.

## Screenshot reshoots (when you change the GUI)

User-facing docs embed screenshots captured by a Playwright harness that **needs a real browser**. The cloud container *does* have one (see "Environment Notes" below), so the default when you change a framed GUI surface is to **reshoot in the same session**: run `scripts/screenshots/refresh.sh`, review `git diff docs/user/assets/`, and commit the regenerated PNGs. **Do not let that drift go silently unrecorded.**

Only when a reshoot genuinely isn't possible — no browser, or the shot needs a fixture the harness doesn't build — add the affected shot id(s) to the **reshoot queue**: `docs/user/screenshots-reshoot-queue.md`. Each id must match an entry in `docs/user/screenshots.manifest.ts`; the wiring check (`scripts/screenshots/wiring-check.py`, gated in `run-tests.sh`) fails if a queued id has no matching shot, so the queue can't rot. To know whether a change touches a shot, scan the manifest's `embeddedIn` / `caption` fields for the surface you modified (e.g. a modal, a panel, a toolbar).

If you *do* have a browser this session, drain the queue instead of growing it: run `scripts/screenshots/refresh.sh`, review `git diff docs/user/assets/`, commit the regenerated PNGs, and delete the drained rows. The full system (manifest, harness, determinism knobs, embedding convention) lives in `docs/plans/user-docs-screenshots.md`.

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

`vtscore.eval` exists to measure **deviations** from the shipped algorithm. That only means something if its **default arm** *is* the shipped algorithm. When the app's algorithm moves and the harness doesn't, every experiment run after that point is measuring a detector nobody uses — and the damage is silent and retroactive, because the numbers still look fine. So: **an app-side algorithm change is not finished until the eval framework has caught up.**

Most of the harness is safe by construction because it **delegates** — `MaxPatchStyle` calls `pool_box_from_media` / `bad_negative_vecs` / `media_score_rows` rather than re-deriving them, so it cannot drift. Prefer delegation over copying every time; it is the only fix that can't rot. Two kinds of code can't delegate, and those are the ones that bite:

- **Ported** — app logic re-implemented in the harness because the original is unreachable (it lives in TypeScript) or unusable (wrapped in interactive, lock-guarded, single-detector caches). `vtscore/eval/autopilot_flow.py` is the whole of this category today.
- **Default resolution** — where the harness resolves "no explicit arm" to whatever the app currently defaults to (`style=None` → `max_patch` on a patch dataset; `blend_schedule=None` → `production_schedule_for(...)`). When the app's default changes, the harness keeps handing out the old one *under the name "default"*.

**The gate:** `scripts/check-eval-app-sync.py` pins a digest of every mirrored app surface (Python and TypeScript), and `./run-tests.sh` fails when one moves. It tells you which harness code to reconcile. After reconciling — or after confirming nothing is owed — re-pin:

```
python scripts/check-eval-app-sync.py --update
```

Digests ignore comments, docstrings, and formatting, so only real logic changes trip it. **Re-pinning without looking at the harness defeats the entire gate**; the digest is a prompt to check, not a checkbox.

**When you add a new mirror** (any new place the harness copies app logic or tracks an app default), add a `Mirror(...)` entry to `MIRRORS` in that script and run `--update`. If the harness *intentionally* differs from the app at that point, put the reason in `divergence=` — the text is printed whenever that mirror trips, so the next person reconciling it knows which differences are deliberate. Named experiment arms (`whole_image`, `max_patch_hac`, …) are supposed to differ and are out of scope; this rule is about the **default** arm only.

## Fix All Errors (CRITICAL)

When you run a build, typecheck, linter, or test suite, **fix every error and failure you see; not only the ones you introduced**. Do not dismiss errors as "pre-existing", "unrelated to my change", or "not my fault" and move on. Do not announce them and ask the user to triage. The user does not want to scan your output for problems you decided to ignore.

This applies to:
- TypeScript errors from `tsc` / `npm run build:prod` (including in `*.spec.ts` files).
- Frontend unit-test failures from the Vitest suite (`cd frontend && npm run test:ci`, also run by `./run-tests.sh` and `./run-tests.sh frontend`).
- Angular build warnings of any kind, including `anyComponentStyle` budget warnings (e.g. `▲ [WARNING] ... exceeded maximum budget`). `run-tests.sh` treats every `▲ [WARNING]` line from `build:prod` as a hard test failure, so do not just bump budgets to silence them: fix the underlying bloat (split the component, extract shared styles, or remove dead rules). Bumping a budget is only acceptable when the size is genuinely justified, and requires the user's explicit approval.
- Python test failures from `./run-tests.sh` and `pytest` runs.
- Linter errors from `ruff check` (including the flake8-bandit `S` ruleset), formatting drift from `ruff format --check`, typos from `codespell`, documentation drift from `scripts/check-docs.py`, dependency issues from `deptry`, known CVEs from `pip-audit`, type errors from `pyright`, and OpenAPI snapshot drift. All of these run as the first steps of `./run-tests.sh`, so the test loop catches them before pytest. There is no CI backstop: VTSearch has no GitHub Actions workflows; `./run-tests.sh` is the source of truth, so do not push a change without running it.
- Any other diagnostics surfaced by tooling you invoke.

If a failure is genuinely outside the scope of the current task (e.g. a flaky network test, a failure in unrelated infrastructure you cannot reproduce), explicitly call it out in your end-of-turn summary with one sentence explaining why you did not fix it. The default is **fix it**; skipping requires justification.

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

**Persistent-tab pickers are an intentional exception.** The rule's trigger is a view that *replaces* its outer view (the picker vanishes, the form takes over). A picker whose navigation chrome stays **persistently visible** while the selected form renders below it never hides an outer view, so there is nothing to navigate "back" to and no `.back-btn` belongs on it — switching selection is done by clicking a different tab in the always-present bar. The Add-Dataset importer picker (`vt-source-picker`'s `.tab-bar` + `.importer-subtab-bar`) is the canonical case: its category/importer tabs remain on screen with the source form beneath them, so it correctly diverges from the New-detector › Trained flow's picker→form→`← Back` shape. Do not add a `.back-btn` to a persistent-tab picker to "align" it; the divergence is by design. (The footer `Cancel` on such a picker is still correct — it dismisses the whole modal, per the Back-vs-Cancel rule above.)

## Commands

- **Run tests (CPU, fast)**: `./run-tests.sh` (runs every gate listed under "What `run-tests.sh` gates" below, then pytest)
- **Run tests by group**: `./run-tests.sh core`, `./run-tests.sh sorting`, `./run-tests.sh api` (see Test Groups below; every invocation runs the cheap serial gates — linters, doc checks, snapshot drift — first, but a group run **skips the heavy whole-repo gates** (pyright, pip-audit, and the frontend gates unless the group is `core`/`frontend`) to keep the inner loop fast; it says so in its output. `VTSEARCH_FULL_GATES=1` forces them. A **full** `./run-tests.sh` runs everything and is mandatory before pushing. `core` and `frontend` additionally run the frontend build + `npm audit`, and `frontend` alone also runs the Vitest unit suite)
- **Run tests with coverage**: `VTSEARCH_COVERAGE=1 ./run-tests.sh` (opt-in; adds ~10-20% overhead)
- **Run multiple groups**: `./run-tests.sh core sorting api`
- **Run tests with extra args**: `./run-tests.sh core -- -x --tb=long` (args after `--` go to pytest)
- **Run library-tier tests only (Flask-blocked)**: `./run-tests.sh vtscore-clean` (runs `tests_lib/` via a meta-path import hook that refuses `flask`, `werkzeug`, `flask_smorest`; proves the library tier is import-clean. This mode `exec`s straight into the checker, so it deliberately skips the linter and frontend gates)
- **Run tests (CPU, full)**: `bash .claude/hooks/ensure-test-deps.sh && python -m pytest tests/ tests_lib/ -q --tb=short -m 'not gpu'`
- **Run slow tests only**: `python -m pytest tests/ tests_lib/ -q --tb=short -m slow` (both trees — the slow tests do not all live under `tests/`; see Test Markers)
- **Run GPU tests**: `python -m pytest tests_lib/gpu/test_gpu.py -q --tb=short -m gpu` (requires CUDA GPU; downloads models on first run)
- **Run all tests (CPU + GPU)**: `python -m pytest tests/ tests_lib/ -q --tb=short -m ''`
- **Regenerate the OpenAPI snapshot** (after any route/schema change, to clear the drift gate): `cd frontend && npm run regenerate-openapi-snapshot` (equivalently `python scripts/dump_openapi.py > frontend/openapi.json`), then commit `frontend/openapi.json`
- **Start app**: `bash .claude/hooks/ensure-test-deps.sh && python app.py` (or `python app.py --local` for dev)
- **CLI autodetect**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json>`
- **CLI autodetect + exporter**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --dataset <file.pkl> --settings <settings.json> --exporter server_json_file --filepath results.json`
- **CLI autodetect + importer**: `bash .claude/hooks/ensure-test-deps.sh && python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio --settings <settings.json>`
- **Check eval/app sync**: `python scripts/check-eval-app-sync.py` (also a `./run-tests.sh` gate; re-pin with `--update` after reconciling the harness)
- **Check documentation**: `python scripts/check-docs.py` (also a `./run-tests.sh` gate; validates relative links, `#anchors`, backticked repo paths, absolute-path leaks, `docs/plans/*.md` citations anywhere in the tree, and code fences. Pure invariants — nothing to re-pin. Fix the doc, or add an allowlist entry with a reason if the path is runtime-generated or a deliberately fictional example)
- **Regenerate doc inventories**: `python scripts/gen-docs-inventories.py` (fills the `<!-- BEGIN GENERATED: ... -->` regions in the docs from the live registries — embedders, plugin families, demo datasets; `--check` is a `./run-tests.sh` gate, so registry changes require rerunning this and committing the result)
- **Install deps**: `bash scripts/install.sh` (auto-detects CPU vs GPU; pass `cpu`/`gpu` to force, or a `cuXYZ` tag to override the GPU wheel, e.g. `bash scripts/install.sh cu121`)
- **Build frontend**: `cd frontend && npm install && npm run build:prod` (builds Angular app to `static/`)
- **Frontend dev server**: `cd frontend && npm start` (proxies `/api/*` to Flask at localhost:5000)
- **Frontend audit**: `cd frontend && npm audit` (checks for known vulnerabilities in dependencies)
- **Frontend unit tests**: `cd frontend && npm run test:ci` (headless Vitest via the `@angular/build:unit-test` builder + jsdom; no browser needed). Also run by `./run-tests.sh` (full suite) and `./run-tests.sh frontend` (frontend-only gate: build + audit + Vitest). `npm test` is the watch-mode variant.
- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Spell check**: `codespell --toml pyproject.toml`
- **Dependency check**: `python -m deptry .`
- **Dead code audit** (manual, pre-release): see `.vulture-whitelist.py` for the full invocation (60% confidence, with marshmallow/pydantic field directories excluded and pytest/Flask/dunder noise filtered). Run before each release; not a CI gate.

## What `run-tests.sh` gates

There is no CI: a **full** `./run-tests.sh` is the only gate, and it still runs every check. **This list is derived from `run-tests.sh`; when you add or remove a gate there, update it here in the same commit.** The run is staged: cheap gates run serially and stop at the first failure with a `TESTS BLOCKED: ...` banner naming which one; the heavy, mutually independent gates then run **concurrently with pytest**, each runs to completion, and every failure is reported (so one pass surfaces every problem instead of one per rerun). A final `RUN PASSED` / `RUN FAILED: <gates>` banner closes the run.

Wrapping everything: a wall-clock cap (`VTSEARCH_TEST_TIMEOUT`, default **1800s = 30 min**, `0` opts out for a deliberately long run) and `.claude/hooks/ensure-test-deps.sh` (minutes on a cold container, near-instant after).

**Stage 1 — cheap gates, serial, fail-fast (~10s total, every invocation):**

| Gate | Command it runs | Notes |
|------|-----------------|-------|
| Lint | `ruff check .` | |
| Format | `ruff format --check .` | Fix with `ruff format .`. |
| Spelling | `codespell --toml pyproject.toml` | |
| Documentation | `scripts/check-docs.py` | Pure invariants over every tracked markdown file: relative links, `#anchors` (GitHub slug rules), backticked repo paths, absolute-path leaks, `docs/plans/*.md` citations **anywhere in the tree**, and broken code fences. Nothing to re-pin; fix the doc, or add an allowlist entry with a reason. |
| Dependencies | `python -m deptry .` | |
| OpenAPI snapshot drift | `scripts/dump_openapi.py` diffed against `frontend/openapi.json` | The generated TS client is built from this snapshot. Regenerate with `npm run regenerate-openapi-snapshot` and commit the result. |
| Doc inventories | `scripts/gen-docs-inventories.py --check` | Regenerate with `python scripts/gen-docs-inventories.py` and commit the result. |
| Dockerfiles | `scripts/check-dockerfiles.py` | |
| User-docs screenshot wiring | `scripts/screenshots/wiring-check.py` | Browser-free; the pixel-diff (`check.sh`) stays a manual chore. Also what makes the reshoot queue un-rottable. |
| vtscore package docs | `scripts/check-vtscore-docs.py` | |
| Eval/app sync | `scripts/check-eval-app-sync.py` | Re-pin with `--update` **after** reconciling the harness. |

**Stage 2 — frontend production build, serial (full run and the `core` / `frontend` groups):** `cd frontend && npm run build:prod`. Any `▲ [WARNING]` line is a hard failure. Runs *before* pytest because some tests serve the built bundle out of `static/`. Skipped with a notice if `frontend/node_modules` is absent.

**Stage 3 — heavy gates, concurrent with pytest (pytest streams in the foreground; lane results print after it):**

| Gate | Command it runs | When | Notes |
|------|-----------------|------|-------|
| Types | `pyright` (pinned via `PYRIGHT_PYTHON_FORCE_VERSION`) | Full run only | Scope is `pyrightconfig.json`. |
| Known CVEs | `pip-audit` | Full run only | Audits the resolved venv, not the requirements files. `PIP_AUDIT_IGNORE` in the script lists advisories with no upstream fix; re-audit and remove an entry once a patched release exists. |
| Frontend audit | `cd frontend && npm audit --omit=dev` | Full run, `core`, `frontend` | Prod deps only — dev-only advisories don't ship. |
| Frontend unit tests | `cd frontend && npm run test:ci` | Full run or `frontend` **only** — deliberately off the fast `core` path | Headless Vitest. |
| Python tests | `pytest tests/ tests_lib/ -n auto --dist loadgroup` | Every run except a `frontend`-only group | |

**Group runs skip the whole-repo stage-3 gates** (pyright, pip-audit, and the frontend gates unless the group asks for them) so the edit/test loop stays in the seconds — the skip is announced in the output, and `VTSEARCH_FULL_GATES=1` forces the complete chain on a group run. Stage 1 runs on every invocation. This is a deliberate trade: the fast inner loop may miss a type error or CVE, which is why **a full `./run-tests.sh` remains mandatory before pushing.**

## Test Groups

Tests are grouped by folder under `tests/` and `tests_lib/`. Each folder is a pytest marker; `./run-tests.sh <group>` runs all tests in `tests[_lib]/<group>/`. New tests inherit their group from the folder they're added to.

| Group | Description |
|-------|-------------|
| `core` | Basic app functionality (audio, medias, votes, inclusion, settings, frontend, torch config) |
| `api` | API contracts, error handling, security, dashboard, embed |
| `sorting` | Sort algorithms, diversity, safe thresholds, enriched text sort |
| `datasets` | Dataset loading, splitting, dedup, parallel/chunked/thin loading, multi-dataset context |
| `io` | Importers, exporters, label I/O, settings I/O, sync sources, PDF/NPZ import |
| `detectors` | Detectors, embedders, clippers, eval, processors, training |
| `downloads` | Demo dataset downloads (AG News, BBC, GTZAN, IMDB, image sources, UCSF, video, generic extract) |
| `integration` | End-to-end workflows, thread safety, async jobs |
| `cli` | CLI autodetect, load sort window, progress bars |
| `converters` | Media converters (document, video, image) |
| `projection` | VTSBrowse UMAP projection + hex-tile pyramid (library tier) |
| `frontend` | Frontend-only gate: Angular `build:prod` + `npm audit` + the headless Vitest unit suite. No Python tests; `./run-tests.sh frontend` skips pytest. Also runs as part of the full `./run-tests.sh`. |
| `gpu` | CUDA-only tests (excluded by default) |

**Recommended workflow**: Run `./run-tests.sh <group>` for the area you changed, then `./run-tests.sh` for the full suite.

`tests/` is the app-tier suite (uses `client`, `vtsearch.routes`, `vtsearch.settings`, `vtsearch.auth`, etc.). `tests_lib/` mirrors the same layout but every file must be import-clean of Flask, `vtsearch.routes`, `vtsearch.settings`, `vtsearch.auth`, `vtsearch.shim`, `vtsearch.autorun_processors`, and `vtsearch.settings_io` — verified by `./run-tests.sh vtscore-clean`. Add a new test to `tests_lib/` if it doesn't touch any app-tier module; otherwise add it to `tests/`.

## Test Markers

The default filter lives in `pyproject.toml`'s `addopts`: `-m 'not gpu and not slow' --timeout=300 --timeout-method=thread`.

- **Default** (`./run-tests.sh` with no group, or a bare `pytest`): fast CPU tests only (~35s). Excludes `gpu` and `slow`.
- **`slow`**: 3 tests, in **two** trees — one CLI subprocess test that spawns `python app.py --autodetect` (`tests/cli/test_cli_main_subprocess.py`, ~16s) and two real-`toponymy` fit tests (`tests_lib/projection/test_toponymy_smoke.py`, module-level `pytestmark`, ~1 min each; `importorskip`ped when toponymy isn't installed). Run with `python -m pytest tests/ tests_lib/ -m slow` — passing only `tests/` silently misses two thirds of them.
- **`gpu`**: CUDA-only tests (`tests_lib/gpu/test_gpu.py`). Run with `-m gpu`.
- **All tests**: `-m ''`.
- **Per-test timeout**: 300s, thread-based (signal-based interruption doesn't work on xdist workers). One hung test fails by name instead of stalling the run; the wall-clock cap in `run-tests.sh` is the backstop for a worker that dies outright and can no longer fire its own timeout.

**Gotcha: naming a group re-opens `slow` and `gpu`.** `./run-tests.sh <group>` passes `-m "<group>"` on the command line, and a command-line `-m` *replaces* the one in `addopts` rather than combining with it. So `./run-tests.sh cli` does run the slow subprocess test, and `./run-tests.sh projection` does run the toponymy smoke tests — a group run is slower than the same tests in a default run. That is also why `./run-tests.sh gpu` works at all. To get the default exclusions back inside a group, spell the filter out after `--` (a later `-m` wins): `./run-tests.sh cli -- -m 'cli and not slow'`.

## Test Workflow (IMPORTANT)

Testing can crash the session. To avoid losing work, follow this workflow:

1. **Commit and push before running tests.** Before running `pytest` or any test command, commit all current changes and push to your working branch. Use a message like `"WIP: pre-test checkpoint"` if the work isn't finalized yet.
2. **Start the run in the foreground with the maximum timeout, and never *launch* it with `run_in_background`.** The test command has a slow startup phase: `ensure-test-deps.sh` installs dependencies (~1-2 min on first run), then `conftest.py` imports `app.py` and generates test media/embeddings before any tests execute. There may be no output for several minutes; this is normal, and is not a sign that output capture is broken.

   **Pass `600000` ms (10 minutes) — the Bash tool's maximum.** A measured warm full `./run-tests.sh` is ~3.5 minutes on a 4-vCPU box (the heavy gates — pyright, pip-audit, the Vitest suite — run concurrently with pytest, so the wall clock is close to pytest's own); a cold container adds ~3 minutes of dep install up front. A cold run can still brush the cap — that is fine: when the tool's cap is hit the harness moves the run to the background rather than killing it; wait for the completion notification and read the output file it names. What matters is that you *started* it in the foreground so the harness tracks it.

   Two consequences worth knowing before you run it:
   - **Do not pipe the run through `tail`/`grep`.** If the harness backgrounds a pipeline, nothing flushes until the whole pipeline ends, so the output file sits empty and you can't watch progress. Run the script bare and read the tail of the output file afterwards.
   - **A run that outlives the tool's cap is not a timeout.** The script has its own 30-minute wall-clock cap (`VTSEARCH_TEST_TIMEOUT`) and prints a distinctive `TESTS TIMED OUT` banner when *it* fires. Absent that banner, the run is still healthy. To stay well inside 10 minutes, run one group at a time — a group run skips the heavy whole-repo gates (pyright, pip-audit, and the frontend gates unless the group is `core`/`frontend`) and typically finishes in well under a minute warm.
3. **If tests fail and fixes are needed**, make the fixes, then commit and push again before re-running tests.
4. **Repeat** until tests pass. Every cycle of fixes should be committed and pushed before the next test run.

This ensures work is recoverable if the session crashes during a test run.

## Reading Test Results (IMPORTANT)

A `./run-tests.sh` run prints its verdict as its very last output, in a `====`-bordered block:
- `RUN PASSED (all gates green; pytest summary above)` → all good
- `RUN FAILED: pytest, pyright` → those gates failed; each failing gate's `TESTS BLOCKED` banner and log tail were printed above

Because the heavy gates run concurrently with pytest and report after it, pytest's own summary block (`ALL 1600 TESTS PASSED (3 skipped, total: 1603)` / `TESTS FAILED: 2 failed, ...`) sits *above* the lane report in a full run — it is still the place to read pytest's counts, and it is the last output of a bare `pytest` invocation.

**ONLY look at these summary blocks** (bordered by `====` lines) to determine pass/fail. Many test names contain the word "error" (e.g., `test_memory_errors.py`, `TestErrorResponseFormat`). These test **error-handling behavior**; they are not failures.

**Do NOT scan test names or output for the word "error" to detect failures.** A line like:
```
tests/test_memory_errors.py::TestPickleMemoryError::test_importer_background_oom_reports_error PASSED
```
means the test **passed**; the word "error" is part of the test name, not an indication of failure.

## Test Isolation (IMPORTANT)

All mutable global state is reset automatically before each test via two autouse fixtures in `conftest.py`:

1. **`reset_state`** — Clears all dataset contexts and creates a fresh `_test_default` context with the pre-generated test medias replayed into it. Also clears:
   - `autorun_extractors`, `autorun_localizers` (global state)
   - Progress cache and progress trackers
   - Login provider and dataset/model registries

2. **`isolated_settings`** — Redirects `SETTINGS_PATH` to a per-test temp file so settings writes never touch `data/settings.json`. Yields the temp path for tests that need to inspect the file.

**When writing new tests:**
- Do NOT add per-file or per-class autouse fixtures to clear autorun state, reset settings, or reset votes — `conftest.py` handles all of this automatically.
- Do NOT add inline `.pop()` or `.clear()` cleanup at the end of tests — the conftest fixtures run before each test regardless of whether the previous test passed or failed.
- If a test needs to temporarily empty `medias`, use the save/restore pattern with try/finally (since `medias` is intentionally NOT reset between tests to avoid expensive re-generation):
  ```python
  saved = dict(medias)
  medias.clear()
  try:
      # ... test logic ...
  finally:
      medias.update(saved)
  ```
- If a test needs to read the settings file path (e.g. to verify persistence), use `isolated_settings` as a parameter: `def test_foo(self, isolated_settings): ...`

`tests_lib/conftest.py` provides app-free, settings-free shared fixtures: `reset_contexts` (autouse, resets dataset/detector contexts, progress trackers, async jobs, label-sync, registries), `_allow_test_tmp_paths` (autouse, widens path validation for tmp dirs), `_stub_embedding_models` (session, stubs every embedder). It also installs a library-only `CoreConfig.from_settings()` builder so library code that calls it works without the app shim. `tests/helpers.py` and `tests_lib/helpers.py` (and likewise `tests/fixtures/medias.py` / `tests_lib/fixtures/medias.py`) are intentional duplicates so each tier is self-contained. **Import them tier-qualified** — `from tests.helpers import ...` inside `tests/`, `from tests_lib.helpers import ...` inside `tests_lib/` — never as a bare `from helpers import ...`. There is deliberately no `pythonpath` entry in `pyproject.toml`: putting both directories on `sys.path` made the bare name resolve to `tests/helpers.py` for *both* trees (pytest inserts `pythonpath` entries in reverse order), which made `tests_lib/helpers.py` dead code and let the copies drift. `tests/core/test_test_tier_helpers.py` gates both halves of this: the duplicated files must stay byte-identical, and no bare `helpers` import may reappear.

## Avoiding Flaky Tests (IMPORTANT)

When writing new tests, avoid these three common sources of flakiness.

### 1. Always seed random number generators

Never call `np.random.randn()`, `np.random.rand()`, `torch.randn()`, or similar without a fixed seed. Random embeddings feed into neural net training and sorting, where different values cause non-deterministic convergence — making assertions pass or fail depending on the random draw.

**Do this:**
```python
rng = np.random.default_rng(42)
fake_embeddings = rng.standard_normal((n, dim)).astype(np.float32)
```

**Not this:**
```python
fake_embeddings = np.random.randn(n, dim).astype(np.float32)  # FLAKY; unseeded
```

### 2. Never use `time.sleep()` for thread synchronization

`time.sleep(0.2)` to "wait for a thread to start" is unreliable on loaded machines. Use `threading.Event` for deterministic synchronization, and set generous polling timeouts.

**Do this:**
```python
started = threading.Event()
def target():
    started.set()
    # ... work ...
thread = threading.Thread(target=target)
thread.start()
started.wait(timeout=5)
```

**Not this:**
```python
thread.start()
time.sleep(0.2)  # FLAKY; may not be enough on a loaded machine
```

### 3. Never use bounded loops to simulate "cancellable" or "interruptible" work

A `for i in range(100): sleep(0.05)` loop finishes in 5 seconds — but on a loaded machine the code that's supposed to interrupt it (e.g. setting a cancel flag) can take longer than 5 seconds to run. If the loop completes before the interrupt arrives, the test follows the wrong code path and fails.

**Do this:**
```python
def slow_load():
    started.set()
    while True:                            # exits ONLY via CancelledError
        dataset_progress.check_cancelled()
        time.sleep(0.05)
```

**Not this:**
```python
def slow_load():
    started.set()
    for i in range(100):                   # FLAKY; can finish before cancel arrives
        dataset_progress.check_cancelled()
        time.sleep(0.05)
```

## Environment Notes (Claude Code on the web)

- **Chromium *is* available — check before assuming it isn't.** The cloud container ships a Playwright chromium under `PLAYWRIGHT_BROWSERS_PATH` (`/opt/pw-browsers`). This entry used to say the opposite, and that stale claim was load-bearing: it is why GUI behaviour got *reasoned about* from the spec instead of *watched* (see issue #2898, where two rounds of frontend fixes were shipped without anyone ever opening a tab). So when a question is "what does the browser actually do here?", go and look.
  - The container's chromium revision does **not** necessarily match the one the `playwright` npm pin wants, so a bare `chromium.launch()` can fail with `Executable doesn't exist` even though a perfectly good browser is present. Do **not** run `npx playwright install` (the environment sets `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1`). Use `scripts/screenshots/launch.mjs`'s `launchChromium()`, which tries the pinned build and falls back to whatever is in the browsers directory; `CHROMIUM_PATH` overrides both.
  - This does not change the **test suite**, which stays deliberately browser-free: the frontend unit suite runs on **Vitest + jsdom** (the Angular 21 `@angular/build:unit-test` builder) and Karma is gone, while the Python backend tests (`./run-tests.sh`) never needed a browser. Keep it that way — a browser-dependent gate would be slow and machine-sensitive. The browser is for *investigation* and for the screenshot harness, not for `run-tests.sh`.
  - A jsdom stub is not a browser. `window.open` is the cautionary example: the Vitest specs returned a truthy fake handle, so they cheerfully passed against code that could never work in Chrome. When a behaviour depends on real browser semantics (popups, user activation, navigation, focus), verify it in chromium *as well as* in the unit suite.

## More docs

- [`docs/SETUP.md`](docs/SETUP.md) — prerequisites, virtualenv, dependency install, frontend build, Docker, SLURM, running the tests.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — key concepts (media items, votes, media types, processors, origins), directory map, dependency graph, plugin systems, state management (multi-dataset / multi-detector contexts, proxies, `X-Dataset-Id` / `X-Detector-Id` headers), auth, origin tracking.
- [`docs/FRONTEND.md`](docs/FRONTEND.md) — Angular SPA architecture: feature-area boundaries, the service layer, the **zoneless change-detection rules**, active dataset/detector propagation, the generated OpenAPI client, component/modal conventions. Read this before changing frontend state or reactivity.
- [`docs/API.md`](docs/API.md) and `docs/api/*.md` — REST API reference.
- [`docs/CLI.md`](docs/CLI.md) — CLI flags and autodetect workflow.
- [`docs/ML.md`](docs/ML.md) — training/scoring details.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — production/offline deployment, env vars, data directory, troubleshooting.
- [`docs/EXTENDING.md`](docs/EXTENDING.md) + [`docs/EXTENDING-plugins.md`](docs/EXTENDING-plugins.md) + [`docs/EXTENDING-media.md`](docs/EXTENDING-media.md) + [`docs/EXTENDING-processors.md`](docs/EXTENDING-processors.md) — how to add plugins.
- [`vtscore/docs/README.md`](vtscore/docs/README.md) — the library tier's own doc set (quickstart, concepts, per-package reference, tutorials, FAQ).
- [`docs/plans/`](docs/plans/) — future-work design docs (open/proposed; shipped work is pruned out, not archived); check here before adding a "Phase N" feature.
- [`docs/RELEASE.md`](docs/RELEASE.md) — the `dev` → `main` release runbook (the procedure the Dev2Main Routine follows: vulture audit, release summary, punch-card refresh, release PR, issue close-out, plan-pointer prune).
- [`docs/branch-protection.md`](docs/branch-protection.md) — who can land on `main` vs `dev`, and what the Free-plan private repo can and cannot enforce.
- [`docs/style-guide.md`](docs/style-guide.md) — frontend SCSS conventions (the styling half of [`docs/FRONTEND.md`](docs/FRONTEND.md)).
- [`CHANGELOG.md`](CHANGELOG.md) — curated record of notable user-facing app changes ([`vtscore/CHANGELOG.md`](vtscore/CHANGELOG.md) is the library's).
