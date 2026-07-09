# HuggingFace OAuth for gated demo datasets

**Status:** Phase 1 shipped — "Sign in with HuggingFace" (OAuth 2.0 + PKCE)
unlocks gated demo datasets and gated model weights. Open follow-ups first;
shipped detail below.

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

## Problem

Several demo datasets (and some model weights, e.g. DINOv3) are *gated* on the
HuggingFace Hub. The raw download path (`vtscore/datasets/downloader/core.py`)
used unauthenticated `requests`, so a gated resource returned `401`/`403` as a
raw `requests.HTTPError` string long enough to push the dashboard row's
Dismiss/Cancel button out of view — and there was no in-app way to authenticate.

## What shipped

- **Token store** — `vtscore/security/hf_auth.py` holds a single HF credential
  **in memory only** (process-scoped, never on disk). `set_credential`,
  `clear_credential`, `get_token`, `is_authenticated`, `get_status`, and
  `auth_header_for_url(url)` (Bearer only for `huggingface.co`/`hf.co`, so the
  token never leaks to presigned CDN/Xet redirects). Defines `GatedResourceError`.
- **OAuth routes** — `vtsearch/routes/auth_huggingface.py` (plain Flask blueprint,
  off the OpenAPI surface): `GET .../status`, `GET .../login` (PKCE verifier +
  CSRF state in the Flask session), `GET .../callback` (exchanges code, stores
  token, redirects to `/?hf_auth=success|error`), `POST .../logout`.
- **Operator setup** (one-time): register an OAuth app with redirect URI
  `<base-url>/api/auth/huggingface/callback` and `read-repos` scope, set
  `HF_OAUTH_CLIENT_ID` (+ `HF_OAUTH_CLIENT_SECRET` for a confidential client;
  optional `HF_OAUTH_REDIRECT_URI`, `HF_OAUTH_SCOPES`). Unconfigured → UI shows
  setup guidance; `HF_TOKEN` env var still works as a fallback.
- **Downloads + embedders** — `core._open_validated_stream` attaches the HF
  Bearer header per redirect hop (host-checked each hop); `401`/`403` raises a
  short, actionable `GatedResourceError` (no retry); `embedder.hf_token()` prefers
  the OAuth token then `HF_TOKEN`, so signing in also unlocks gated weights.
- **Frontend** — `HuggingFaceAuthService` (signals) drives status/login/logout; a
  **HuggingFace** tab in Settings; the failed-load row offers **"Sign in with
  HuggingFace"** on a gated error with a layout fix so a long message can't
  obscure the buttons; the root component toasts the OAuth result and strips the
  `hf_auth` query params.
