# Issue tracker conventions

This page explains the reasons behind the rules in
[CLAUDE.md's "Issue lifecycle"](../CLAUDE.md#issue-lifecycle-critical). The
rules themselves, and the table of what each stage owes, stay in CLAUDE.md
because every session needs them in context. This page covers what each signal
means, the views it keeps honest, how the rules are enforced, and the incidents
behind them. How the release closes issues is in
[`docs/RELEASE.md`](RELEASE.md) steps 6 and 6b.

## The four signals and the views they serve

An issue carries two labels chosen when it is filed and two statuses that track
the work:

| Signal | Answers | Set by |
|---|---|---|
| `claude` | Who wrote this? | the filing session, at creation |
| `experiment` | Does closing it need machine time? | the filing session, at creation |
| `solved` | Is any problem-solving left? | the fix session, when it opens the PR |
| assignee `samggreenberg` | Is somebody on it right now? | the session that starts work on it |

Together they make the tracker answer the questions people actually ask:

```
is:issue is:open -label:solved                # what a human should work on next
is:issue is:open -label:solved no:assignee    # ...and nobody is on it right now: free to pick up
is:issue is:open label:solved                 # solved; waiting only on merges
is:issue is:open -label:claude                # filed by a human
label:experiment                              # needs machine time booked
-label:experiment                             # can be picked up right now
```

Each view is only as good as the signal behind it. The rules exist to keep the
signals true, and most of them are about doing the write at the right moment.

## `claude`: authorship is otherwise invisible

Claude sessions file issues through the repository owner's GitHub account, so
no `author:` query separates a Claude-written issue from a human-written one.
The label is the only signal, and so the burden is on Claude. Claude labels its
own issues. Humans file without ceremony and are never asked to remember a
`human` tag.

That only works if Claude is exhaustive. A Claude-filed issue that slips
through unlabelled silently pollutes the human-filed view. The label says
nothing about the topic. Nearly every issue here concerns work next to Claude,
and that is never why the label goes on.

## `experiment`: a scheduling gate

`label:experiment` is the queue of work that needs machine time booked, and
`-label:experiment` is what can be picked up right now. A wrong label in either
direction costs a session. GRID-only work in the pick-up-now queue goes to a
container that cannot do it (#3694 is that incident), and laptop work in the
experiment queue waits for a booking it never needed. See the
`grid-experiments` skill for how runs are launched.

The `<!-- experiment: <reason> -->` marker exists so the gate survives the
label being dropped. It has a fixed form: `<!--`, the word `experiment`, a
colon, then the reason. It renders as nothing on GitHub. It is only a hint. It
is never required, a missing marker never removes the label, and its text is
never parsed: whether the marker is present is the whole signal.

Its twin, `<!-- not-an-experiment: <reason> -->`, opts an issue out of the
hook's "this reads like a run" heuristic. That reason **is** read. For its
first weeks the opt-out matched on its prefix alone. Four markers in three days
then had to be corrected by hand, each one disproved by its own text (#3708).
So a reason that describes a run is now refused.

## `solved`: no problem-solving left

`solved` does not wait for the merge. "PR open" and "merged into `dev`" both
have no thinking left in them, so neither should be offered to someone asking
what to work on. In any case, no session is around to observe a merge, so a
label tied to the merge would in practice never be applied.

The label is transient, not a historical fact. It comes off in the write that
closes the issue, and also if the fix falls through. A closed issue that still
carried it would assert something false, and if that issue were ever reopened
it would wrongly read as solved.

## The assignee: is somebody on it right now?

Assignment answers a different question from the labels. Its job is the
`no:assignee` view, which is what stops two sessions, or a session and a human,
from starting the same issue an hour apart. That only works if the assignee
goes on **early** and comes off **promptly**. A stale assignee is worse than
none, because it makes free work look taken.

- **Assign before reading.** Investigating an issue is working it, and
  evaluation is the longest window in which someone else could pick up the same
  number. Assign whoever filed the issue: the assignee names who is *doing* the
  work, and every session here works through the owner's account.
- **Unassign in the write that adds `solved`.** Once the issue is solved nobody
  is working it, and only merges remain. This half went unfollowed while it was
  written as prose only, so since #4144 the hook enforces it.
- **Closing clears it too.** A closed issue has nobody working it.

## Linking: why the PR keyword is load-bearing

The Dev2Main sweep ([`docs/RELEASE.md`](RELEASE.md) step 6) decides what to
close by reading **PR bodies**, not issue comments. A `Refs #N` on a PR that
actually finishes the issue is not a harmless understatement. The sweep reads
it as "still partial" and skips the issue, and because no later release looks
at an already-merged PR again, the issue stays open forever while its fix is
live in `main`. #2930, #2940 and #2951 each went this way. That is why the
default is `Closes`, why `Refs` needs work you can name, and why the issue
comment has to agree with the keyword.

The issue comment names the PR number (`Addressed in #M`) rather than a commit
SHA because the backstop script matches `#M`. A SHA-only pointer is a real
claim the script cannot resolve on its own, so it lands in `NEEDS REVIEW` and
costs a human a lookup.

**Why a fix session leaves the issue open.** GitHub auto-closes a
keyword-linked issue only when the PR merges into the **default branch**
(`main`), and fix PRs target `dev`. That is intended: an issue stays open until
its fix reaches `main` and actually ships, and the Dev2Main sweep closes it then
with `state_reason: completed`. Meanwhile `solved` keeps it out of the queue a
human picks from.

## Duplicates: only a pointer resolves an issue

An issue is only ever resolved by something that names its number. When a fix
PR names one of two duplicates, the other has *nothing* pointing at it, and all
three of step 6's buckets (closing keywords, non-closing references, and the
orphan backstop over issue comments) are blind to it by construction. #2911, a
duplicate of #3025 whose fix PR named only #3025, is the incident. That is why
a duplicate is linked the moment it is found, before any fix exists.

A duplicate found after its fix has already reached `main` is the one issue a
fix session closes itself. The sweep's window is `origin/main..origin/dev`, and
a fix already on `main` has fallen out of that window, so nothing would ever
pick the issue up again.

## The recommended model: routing

Naming a model in every issue lets a task go to the cheapest model that will do
it well. A Haiku-tier mechanical edit should not burn Opus, and a
regression-prone refactor should not go to a model that will botch it. The
family name, not the version, is what stays meaningful after the next release.

## Enforcement: the hook and the backstop

Prose in CLAUDE.md only binds a session that checked it out and still has it in
view, so the rules most often missed are enforced in code. There are two layers:

- **`.claude/hooks/require-issue-labels.py`** is a `PreToolUse` gate on both
  `issue_write` and `gh issue …`. It blocks:
  - a create missing `claude`, or missing `experiment` when the issue reads
    like a run;
  - an update whose `labels` drop `experiment`;
  - a write that adds `solved` without clearing the assignee;
  - a `completed` close that keeps `solved` or omits `labels`.

  Its docstring records why each guard exists. The guards that have to look
  the issue up do it through `gh`, and they let the call through whenever they
  cannot get an answer (no `gh`, unauthenticated, offline). So those guards are
  awake on the laptop, and in a web container they catch nothing.
- **`scripts/reconcile-solved-labels.py`** is the backstop for whatever got
  past the hook. It is a pure function of PR and issue JSON that you gather and
  pipe in: with `gh` on the laptop, or with the `github` MCP tools in a web
  container, where `gh` is absent and the ambient `GH_TOKEN` gets a 403. It
  never guesses at a comment posted *after* a fix pointer, which could be a
  thank-you or a report that the fix does not work, so it sends that issue to
  `NEEDS REVIEW`. It only ever *removes* an assignee, because it cannot tell
  "nobody is working this" from "a session started five minutes ago". So
  assigning has no safety net. The recipe and the bucket semantics are in
  [`docs/RELEASE.md`](RELEASE.md) step 6b.
