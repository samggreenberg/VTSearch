# HuggingFace OAuth for gated demo datasets

**Status:** Remaining work is the open follow-ups below (auto-retry after
sign-in, token persistence/refresh, per-dataset gated badges).

**Background:** Phase 1 shipped a "Sign in with HuggingFace" flow (OAuth 2.0 +
PKCE) that unlocks gated demo datasets and gated model weights, with an
in-memory-only token store.

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
