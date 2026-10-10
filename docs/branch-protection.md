# Controlling who can change `main` (and `dev`)

Goal: **only @samggreenberg should land changes on `main`.** `dev` is the shared
integration branch and stays open to collaborators via PRs.

## Current state (verified 2026-09-08)

This repository is **public**, so branch protection and rulesets are available at
no cost, and both long-lived branches carry protection:

| | |
|---|---|
| visibility | **public** (`"private": false`) |
| default branch | `main` |
| `main` | **protected** |
| `dev` | **protected** |
| collaborators | 4 — @samggreenberg (admin), @xofm31, @MatthewELucio, @CarHorseBatteryStaple (write) |

That is the *existence* of protection, which is what the branch API reports. It
does not say what each rule enforces — see
[Reading the rules that are actually set](#reading-the-rules-that-are-actually-set)
below, and read them before relying on any specific guarantee.

### Collaborators cannot be made read-only

**Every collaborator on a personal-account repo has write access.** The
Read / Triage / Write / Maintain roles exist only inside organizations, so the
three collaborators above cannot be made read-only while the repo lives under a
personal account. Restricting *who may push to `main`* is therefore done by the
branch rule's push restriction, not by giving anyone a weaker role.

## `dev` is governed by a ruleset as well as branch protection

`dev` carries two independent layers, and a merge must satisfy both:

- **Classic branch protection** (Settings → Branches), described under
  [The intended rules](#the-intended-rules) below.
- **A repository ruleset** targeting `dev` (Settings → Rules → Rulesets). A
  refusal from this layer reads *"Repository rule violations found"* rather than
  the branch-protection wording, which is how to tell which layer said no.

### `suite-grid` is informational, not required

`suite-grid` is a GitHub commit status posted by
[`scripts/slurm/suite.sbatch`](../scripts/slurm/suite.sbatch) at the end of every
GRID run of the full suite, success or failure, on the exact SHA it tested. It is
the only machine-readable record that a commit passed the suite (the repo's two
GitHub Actions workflows only publish and gate nothing), so it is worth posting
and worth reading on a PR.

It is **not** a required check, and must not become one: a cloud session has no
`ssh` to reach the GRID, no working `gh` to post a status, and the GitHub MCP
tools can merge but cannot post one, so requiring it locks every Claude Code on
the web session out of merging (#4133 tried it; #4149 reverted it). It would also
make every merge of `dev` into a PR branch wait on a fresh ~13-minute GRID run,
since the status belongs to one SHA.

**The merge gate on every surface is a full, green `./run-tests.sh`.** A
`suite-grid` status is supporting evidence when a GRID session has one; its
absence blocks nothing. Do not re-add it to the required checks without a way for
cloud sessions to produce it.

## Why `dev` survives a Dev2Main release

Worth stating explicitly, because the repo now has **"Automatically delete head
branches"** enabled and the [Dev2Main](RELEASE.md) release PR is `dev` → `main` —
which makes `dev` the *head* branch of a PR that gets merged every release.

`dev` is not deleted, for two independent reasons:

1. **It is protected.** GitHub's automatic head-branch deletion skips protected
   branches. A protection carrying `allow_deletions: false` blocks the deletion
   outright rather than merely declining to initiate it.
2. **It is the base of other open PRs.** GitHub does not auto-delete a branch
   that another open PR is targeting, and on any ordinary day `dev` is the base
   of several.

If it were ever deleted anyway, nothing is lost: after the release merge `dev`'s
tip is an ancestor of `main`, and the merged PR page offers **Restore branch** to
recreate it at the same commit. The cost would be disruption — the
`.claude/hooks/session-start.sh` hook and every open PR's base break until it is
restored — not lost work.

## Soft controls, which still do useful work

Protection rules are the hard gate; these remain worth keeping because they shape
behaviour before anyone reaches the gate.

### CODEOWNERS auto-requests review

[`.github/CODEOWNERS`](../.github/CODEOWNERS) is `* @samggreenberg`, so GitHub
automatically requests that review on every PR. Combined with a `main` rule that
requires Code Owner review, it makes @samggreenberg the only valid approver for
anything landing on `main`.

### Team convention

`CLAUDE.md` encodes the working agreement, and it is what keeps `main` quiet in
practice — nobody working off `dev` has a routine reason to touch `main`:

- All work branches off `dev`; all PRs target `dev`, **never** `main`.
- `main` is updated **only by promoting `dev` → `main`**. The Dev2Main Routine
  does it, acting through @samggreenberg's account: it opens the release PR and,
  as the runbook's last step, merges it ([`RELEASE.md`](RELEASE.md) step 8).
- Collaborators do not push directly to `main`.

### Notifications

So an unwanted push to `main` is visible rather than silent: **Watch → All
Activity** (or **Custom → Pushes**), or subscribe to the `main` commit feed at
`https://github.com/samggreenberg/vtsearch/commits/main.atom`.

## Reading the rules that are actually set

The branch listing reports only whether a branch is protected. To see what the
protection contains:

```bash
gh api repos/samggreenberg/vtsearch/branches/main/protection
gh api repos/samggreenberg/vtsearch/branches/dev/protection
```

The UI equivalent is **Settings → Rules → Rulesets**, or the classic
**Settings → Branches** editor.

## The intended rules

Recorded here as the reference for what protection *should* say. Check the live
rules against this rather than assuming they match.

Lock `main` — PR plus Code Owner review required, only @samggreenberg may push:

```bash
gh api -X PUT repos/samggreenberg/vtsearch/branches/main/protection \
  --input - <<'JSON'
{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "require_code_owner_reviews": true,
    "dismiss_stale_reviews": true
  },
  "restrictions": { "users": ["samggreenberg"], "teams": [], "apps": [] },
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

With CODEOWNERS as above, `require_code_owner_reviews` means only
@samggreenberg's review satisfies the gate, and `restrictions.users` limits
pushes. Set `enforce_admins: true` to bind yourself to the same rules.

**That review gate and the self-merging release interact.** The Dev2Main Routine
merges the release PR through @samggreenberg's account (RELEASE.md step 8). With
`enforce_admins: false` the owner's account may merge without a separate
approval. GitHub will not let an author approve their own PR, so turning on
`enforce_admins` makes every release wait for a human again. The routine then
leaves the PR open and says so rather than working around the refusal.

Keep `dev` lighter — a PR guardrail with no mandatory approver, so routine work
and Claude PRs keep flowing:

```bash
gh api -X PUT repos/samggreenberg/vtsearch/branches/dev/protection \
  --input - <<'JSON'
{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

`allow_deletions: false` on `dev` is the setting that makes the release flow
above safe. It is worth confirming it is actually set.
