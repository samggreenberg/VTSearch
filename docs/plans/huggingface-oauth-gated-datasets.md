# HuggingFace OAuth for gated demo datasets

**Status:** Phase 1 shipped.

## Problem

Several demo datasets (and some model weights, e.g. DINOv3) are *gated* on the
HuggingFace Hub: downloading them requires an authenticated request from an
account that has accepted the dataset's terms. VTSearch's raw download path
(`vtscore/datasets/downloader/core.py`) used unauthenticated `requests`, so a
gated dataset returned `401`/`403` and the failure surfaced as a raw
`requests.HTTPError` string. That message was long enough to grow the dashboard
loading-task row tall enough to **push the Dismiss/Cancel button out of view**,
and there was no in-app way to authenticate.

## What shipped (Phase 1)

### Authentication: "Sign in with HuggingFace" (OAuth 2.0 + PKCE)

- **Token store** — `vtscore/security/hf_auth.py` holds a single HuggingFace
  credential **in memory only** (process-scoped, never written to disk; it's a
  short-lived secret the OAuth flow re-mints). Exposes `set_credential`,
  `clear_credential`, `get_token`, `is_authenticated`, `get_status`, and
  `auth_header_for_url(url)` which returns a Bearer header **only** for
  `huggingface.co` / `hf.co` hosts (so the token never leaks to presigned CDN /
  Xet redirect targets). Also defines `GatedResourceError`.
- **OAuth routes** — `vtsearch/routes/auth_huggingface.py` (plain Flask
  blueprint, kept off the OpenAPI surface like `events_bp`):
  - `GET /api/auth/huggingface/status` → `{configured, authenticated, username, scopes}`
  - `GET /api/auth/huggingface/login` → `{configured, authorize_url}` (PKCE
    verifier + CSRF state stashed in the Flask session)
  - `GET /api/auth/huggingface/callback` → exchanges the code, fetches the
    username, stores the token, redirects to `/?hf_auth=success|error`
  - `POST /api/auth/huggingface/logout` → clears the token
- **Operator setup** (one-time): register an OAuth app at
  `huggingface.co/settings/applications/new` with redirect URI
  `<base-url>/api/auth/huggingface/callback` and the `read-repos` scope, then set
  `HF_OAUTH_CLIENT_ID` (+ `HF_OAUTH_CLIENT_SECRET` for a confidential client).
  Optional: `HF_OAUTH_REDIRECT_URI` (proxy override), `HF_OAUTH_SCOPES`.
  When unconfigured, the UI shows setup guidance instead of a dead button, and
  the `HF_TOKEN` env var still works as a fallback.

### Wiring the token into downloads + embedders

- `core._open_validated_stream` now attaches the HF Bearer header per redirect
  hop (host-checked each hop).
- A `401`/`403` raises a short, actionable `GatedResourceError` (no retry)
  instead of a raw HTTPError; wording adapts to whether we're signed in.
- `vtscore/media/embedder.hf_token()` prefers the OAuth token, then `HF_TOKEN`,
  so signing in also unlocks gated model weights.

### Frontend

- `HuggingFaceAuthService` (signals) drives status/login/logout.
- A **HuggingFace** tab in Settings: sign-in/out, status, and setup help.
- The dashboard's failed-load row offers **"Sign in with HuggingFace"** when the
  error is a gated HF error, and its layout was fixed so a long error message
  can never obscure the action buttons (flex row, `min-width:0` scrollable
  message, `flex-shrink:0` actions).
- The root component toasts the OAuth round-trip result and strips the
  `hf_auth` query params.

## Open follow-ups

- **Auto-retry after sign-in.** The gated-error row offers sign-in but not a
  one-click retry of the *same* demo load (the loading task doesn't carry the
  original `load-demo` params). Today the user re-picks the dataset. Threading
  the request params through the loading task would enable a "Retry" button.
- **Token persistence / refresh.** The credential is in-memory only, so a server
  restart requires signing in again, and we don't use the OAuth refresh token to
  silently renew an expired access token. Both are deliberate Phase-1 trade-offs.
- **Per-dataset gated badges.** The demo picker could mark gated datasets up
  front (and prompt sign-in before the user starts a doomed download) rather than
  only reacting on failure.
