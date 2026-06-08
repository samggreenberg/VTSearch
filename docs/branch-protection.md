# Controlling who can change `main` (and `dev`)

Goal: **only @samggreenberg should land changes on `main`.** `dev` is the shared
integration branch and stays open to collaborators via PRs.

## Reality check: this repo is private on the Free plan

Two GitHub limitations apply here, and they shape everything below:

1. **No branch protection or rulesets.** On the Free plan, protected branches and
   repository rulesets are only available on **public** repositories. A private
   repo on Free **cannot** enforce "require a PR," "require my review," or
   "restrict who pushes to `main`." Those settings are simply unavailable.
2. **No granular collaborator roles.** On a **personal-account** repo (this one),
   every collaborator you add has **write/push** access. The Read / Triage /
   Write / Maintain roles only exist inside **organizations**. So you can't make
   any of the 7 collaborators (`xofm31`, `sbwilli3`, `matsunagateitoku`,
   `trevoradriaanse`, `qr1338`, `GCHQDev42081`, `drew-synergist-computing`)
   read-only either.

Net effect: **right now, nothing GitHub-side can hard-*prevent* a collaborator
from pushing to `main`.** Enforcement requires changing the plan or visibility
(see the last section). Until then, the controls below are *soft* — they signal,
notify, and rely on team convention, but do not block.

## Soft controls in effect today

### 1. CODEOWNERS auto-requests your review

`.github/CODEOWNERS` is `* @samggreenberg`. Even without branch protection,
GitHub uses it to **automatically request your review** on every PR. This does
not *block* a merge on the Free/private plan, but it makes your sign-off the
visible, expected step on anything heading toward `main`.

### 2. Team convention (the actual gate, for now)

The enforced rules don't exist, so the working agreement is the gate. The
project already encodes it in `CLAUDE.md`:

- All work branches off `dev`; all PRs target `dev`, **never** `main`.
- `main` is updated **only by @samggreenberg**, by promoting `dev` → `main`.
- Collaborators do not push directly to `main`.

Keep `main` as the stable/release branch and do all day-to-day merging on `dev`.
Because everyone works off `dev`, no one has a routine reason to touch `main`.

### 3. Notifications so you'd *see* an unwanted push

Since you can't block pushes, make sure you'd notice one:

- **Watch** the repo (top-right **Watch → All Activity**, or **Custom → Pushes**)
  so direct pushes to `main` generate a notification.
- Optionally subscribe to the `main` branch's commit feed (Atom):
  `https://github.com/samggreenberg/vtsearch/commits/main.atom`.

This turns "someone changed `main`" from invisible into an alert you can act on
(revert + a conversation), which is the best available recourse without
enforcement.

## When you want real enforcement

The soft controls above can't *prevent* a push. To actually restrict `main` so
only you can land changes, pick one of these — then the enforced steps further
below become available:

| Path | Cost | Result |
|------|------|--------|
| **GitHub Pro** | ~$4/mo | Keep repo private; classic branch protection works on private repos. Cheapest real fix. |
| **Make repo public** | Free | Branch protection + rulesets become available for free. (Code becomes public.) |
| **Move into an Organization** | Free org / paid Team | Granular member roles (read-only members, contribute via fork+PR). Private-branch *protection* still needs the org **Team** plan. |

### Enforced setup (once on Pro, or public)

Classic branch protection via `gh` (run as `samggreenberg`):

```bash
# Lock main: PR + your Code Owner review required, only you may push.
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

With `.github/CODEOWNERS` = `* @samggreenberg`, `require_code_owner_reviews`
means only *your* review satisfies the gate; `restrictions.users` limits pushes
to you. Set `enforce_admins: true` to bind yourself to the same rules.

For `dev`, keep it lighter — a PR guardrail with no mandatory approver so routine
work (and Claude PRs) keeps flowing:

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

Verify with `gh api repos/samggreenberg/vtsearch/branches/main/protection`.
The equivalent UI lives under **Settings → Rules → Rulesets** (or the classic
**Settings → Branches** editor) once your plan/visibility allows it.
