# Locking down `main`

Goal: **only @samggreenberg can approve and land changes to `main`.** No other
collaborator can push directly to `main` or self-merge a PR into it.

As of this writing the repo had **no protection on `main`** and 7 collaborators
with `write` access (`xofm31`, `sbwilli3`, `matsunagateitoku`,
`trevoradriaanse`, `qr1338`, `GCHQDev42081`, `drew-synergist-computing`), all of
whom could change `main`. The steps below close that gap.

This pairs with the repo-root [`.github/CODEOWNERS`](../.github/CODEOWNERS)
(`* @samggreenberg`), which makes Sam the sole code owner.

## Option A — Branch ruleset (recommended, modern UI)

GitHub UI: **Settings → Rules → Rulesets → New branch ruleset**

1. **Name**: `protect-main`
2. **Enforcement status**: `Active`
3. **Bypass list**: empty (or add only `samggreenberg` if you want an admin escape hatch).
4. **Target branches**: add target → `Include by pattern` → `main`.
5. Enable these rules:
   - ☑ **Restrict creations / deletions** (no deleting `main`).
   - ☑ **Restrict updates** (blocks direct pushes; changes must go through a PR).
   - ☑ **Require a pull request before merging**
     - Required approvals: **1**
     - ☑ **Require review from Code Owners** ← this is what makes *you* the
       required approver, via `.github/CODEOWNERS`.
     - ☑ **Dismiss stale approvals when new commits are pushed**
   - ☑ **Block force pushes**

Because CODEOWNERS lists only `@samggreenberg`, the "Code Owners" requirement
can only be satisfied by your review. Other write collaborators can open PRs and
even click "approve," but their approval does not satisfy the Code Owner rule,
so the PR stays unmergeable until you approve.

## Option B — Classic branch protection (`gh` CLI)

If you prefer classic protection over a ruleset, run this from a machine
authenticated as `samggreenberg` (`gh auth status` to confirm):

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
  "restrictions": {
    "users": ["samggreenberg"],
    "teams": [],
    "apps": []
  },
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_linear_history": false
}
JSON
```

Key fields:
- `"restrictions".users = ["samggreenberg"]` — only you may push to `main`.
- `"require_code_owner_reviews": true` — with CODEOWNERS, only your review counts.
- `"enforce_admins": false` — you keep an admin override; set `true` to bind
  yourself to the same rules.

### Verify

```bash
gh api repos/samggreenberg/vtsearch/branches/main/protection
# or, for a ruleset:
gh api repos/samggreenberg/vtsearch/rulesets
```

## Protecting `dev` (lighter than `main`)

`dev` is the integration branch where day-to-day work merges before it's
promoted to `main`. It should **not** carry the same owner-only gate as `main`,
or you become the required reviewer for every change (including every Claude
PR). The goal here is just a guardrail: **no direct or force pushes — everything
goes through a PR — but no mandatory approver**, so routine work keeps flowing.

GitHub UI: **Settings → Rules → Rulesets → New branch ruleset**

1. **Name**: `protect-dev`
2. **Enforcement status**: `Active`
3. **Bypass list**: empty (add `samggreenberg` only if you want an escape hatch).
4. **Target branches**: `Include by pattern` → `dev`.
5. Enable:
   - ☑ **Restrict deletions** (no deleting `dev`).
   - ☑ **Require a pull request before merging**
     - Required approvals: **0** (a PR is required, but no review is mandated).
     - ☐ Do **not** enable "Require review from Code Owners" — that would pull
       you into every `dev` PR via `.github/CODEOWNERS`.
   - ☑ **Block force pushes**

Equivalent classic protection via `gh` (run as `samggreenberg`):

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

`"required_approving_review_count": 0` with a non-null
`required_pull_request_reviews` block means "a PR is required, but it can merge
without a formal approval." `"restrictions": null` keeps push access as-is (any
write collaborator can open/merge PRs); `allow_force_pushes: false` is the
guardrail.

If you later want reviews on `dev` too, bump the count to `1` (any collaborator
can satisfy it) — but leave `require_code_owner_reviews` off unless you
specifically want to be the gate.

## Note on collaborator write access

Branch protection governs `main` specifically; the 7 collaborators keep `write`
on the rest of the repo (needed for `dev` and feature branches). If you also
want to reduce their general access, change their role in
**Settings → Collaborators and teams** — but that's separate from protecting
`main`, which the steps above fully cover.
