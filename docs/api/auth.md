# Authentication & Static UI

[← Back to API index](../API.md)

VTSearch has **two independent auth systems**, easy to confuse:

1. **VTSearch user auth** (`/api/auth/*`) — who the *VTSearch user* is. Governs
   per-user data directories and the login gate. Configured by the server's
   **login provider**.
2. **HuggingFace OAuth** (`/api/auth/huggingface/*`) — hands the *server* a
   HuggingFace Hub token so it can download gated demo datasets and gated model
   weights. It does **not** change who the VTSearch user is.

---

## VTSearch user auth

### Login providers

The active provider is chosen once at server start via the `--login` CLI flag
(see [CLI.md](../CLI.md)); with no flag the app runs single-user. The provider
determines what `/api/auth/*` does and whether the SPA shows a login screen.

| Provider | `--login` | `provider` name | `login_required` | Behaviour |
|----------|-----------|-----------------|------------------|-----------|
| `DefaultLoginProvider` | *(none, default)* | `"default"` | `false` | Single-user. Every request is authenticated as `"default"`; `data/` used directly (no per-user subdir). Login/logout return 400. |
| `TrivialLoginProvider` | `--login trivial` | `"trivial"` | `true` | Cookie-based, **no password** (dev/testing multi-user). Username sent to `POST /api/auth/login`, stored in a signed session cookie; per-user data dir `data/<username>/`. |
| `ApiKeyLoginProvider` | `--login api_key` | `"api_key"` | `false` | Bearer-token for headless clients. Reads `Authorization: Bearer <key>`, hashes it (SHA-256), looks it up in `data/api_keys.json`. No login UI; no `/api/auth/login` support (send the header directly). |

The SPA switches on `login_required` from `GET /api/auth/status`: when `true`
it shows a login screen at startup, otherwise it goes straight to the app.

### Server-side enforcement

Whether the server actually *rejects* unauthenticated requests is governed by
the provider's `enforce_auth()`, checked by a `before_request` hook: when it
is true and `is_authenticated(request)` is false, any `/api/*` request is
refused with **401** `{"message": "Authentication required", "error_code":
"auth_required", ...}` before reaching a route handler. Exactly three paths are
exempt — `/api/auth/status`, `/api/auth/login`, `/api/auth/logout` — so a
client can always discover the auth mode and log in; everything else,
including `/api/auth/huggingface/*`, sits behind the gate. Non-API paths
(the SPA shell, favicons) are never gated.

Per provider:

- **`default`** — `enforce_auth()` is `True` but every request is
  authenticated, so single-user deployments never see a 401. No change from
  historical behaviour.
- **`trivial`** — `enforce_auth()` is **`False`**, deliberately: the provider
  is passwordless, so a server-side gate would add no security (any caller
  could simply log in as any name first). Requests without a session cookie
  are served as `"anonymous"`; the login screen is an identity switcher, not
  an access control.
- **`api_key`** — enforced. Requests without a valid Bearer token get 401
  with `WWW-Authenticate: Bearer`. This is the provider to use when access
  must actually be gated.
- **Custom providers** — enforced by default (`enforce_auth()` inherits
  `True`); a provider whose `is_authenticated()` raises fails **closed**
  (401). Override `enforce_auth()` to `False` only if anonymous access is a
  legitimate mode for your deployment.

### Usernames are path components

Any provider that returns a per-user data dir puts the username on the
filesystem twice: `data/<username>/` is where per-user settings are written,
**and** it is the confinement root passed to `validate_server_filepath()` for
server-file importers and exporters. That check calls `base_dir.resolve()`,
which *collapses* `..` rather than rejecting it — so an unvalidated username
would silently widen the sandbox rather than trip it.

Every username is therefore screened by `vtsearch.auth.is_safe_username()`
(`[A-Za-z0-9._-]+`, and not a bare `.` / `..` segment) at the point it enters
the app: `ApiKeyLoginProvider` screens `api_keys.json` at load time, and
`TrivialLoginProvider` re-screens the session cookie on every read — the
cookie is only integrity-protected by `app.secret_key`, so a client that
knows the key can put an arbitrary string in it regardless of what
`POST /api/auth/login` accepted. An unsafe name resolves to `"anonymous"`
with `authenticated: false`.

This is *not* an impersonation defence. In `trivial` mode anyone may claim
any username by design; the check only keeps a username from becoming a
path escape.

### Auth status

```
GET /api/auth/status
```

→
```json
{
  "provider": "default",
  "user": "default",
  "authenticated": true,
  "login_required": false
}
```

Returns the active login provider name, current user, whether the request is
authenticated, and whether the frontend should show a login screen. With
`DefaultLoginProvider`, every request is authenticated as `"default"`.

### Login

```
POST /api/auth/login
```

**Body:** `{"username": "..."}`

Only the `trivial` provider supports login.

→ **`trivial`**: sets the session username and returns the auth status dict
(same shape as `GET /api/auth/status`, now `authenticated: true`).
→ **`default` / `api_key`**: **400** `{"message": "Login/logout not supported by
the active provider", ...}`.

### Logout

```
POST /api/auth/logout
```

Only the `trivial` provider supports logout.

→ **`trivial`**: clears the session username and returns the auth status dict.
→ **`default` / `api_key`**: **400** `{"message": "Login/logout not supported by
the active provider", ...}`.

---

## HuggingFace OAuth ("Sign in with HuggingFace")

Authenticates VTSearch's **outbound** requests to the HuggingFace Hub so gated
demo datasets and gated model weights download successfully. The obtained token
lives only in a process-scoped in-memory store (`vtscore.security.hf_auth`);
nothing is written to disk. This is **server-side auth, not VTSearch user
auth** — signing in here gives the server a Hub token, it doesn't log a user in.

The flow is **available only when configured** — the server operator registers
a HuggingFace OAuth app and sets `HF_OAUTH_CLIENT_ID` (plus optional
`HF_OAUTH_CLIENT_SECRET`, `HF_OAUTH_REDIRECT_URI`, `HF_OAUTH_SCOPES`). When
unconfigured, the endpoints return `configured: false` so the UI can show setup
guidance instead of a dead button.

These are plain Flask routes (browser redirects), so they are **not** on the
OpenAPI surface.

### HuggingFace status

```
GET /api/auth/huggingface/status
```

→
```json
{"configured": true, "authenticated": false, "username": "", "scopes": ""}
```

`configured` reflects whether an OAuth client id is set; the rest reflect the
current sign-in state (`authenticated`, `username`, `scopes`).

### HuggingFace login

```
GET /api/auth/huggingface/login
```

Begins the PKCE OAuth handshake.

→ Configured: `{"configured": true, "authorize_url": "https://huggingface.co/oauth/authorize?..."}`
— the frontend navigates the browser to `authorize_url`.
→ Unconfigured: `{"configured": false}`.

### HuggingFace callback

```
GET /api/auth/huggingface/callback
```

OAuth redirect target (not called directly). Exchanges the authorization code
for a token, stores it in memory, then **302-redirects** back to `/?hf_auth=success`
(or `/?hf_auth=error&hf_auth_reason=...` on failure). Not a JSON endpoint.

### HuggingFace logout

```
POST /api/auth/huggingface/logout
```

Forgets the stored Hub credential (sign out).

→ `{"ok": true}`

---

## Static / UI

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve the single-page application (`index.html`) |
| GET | `/favicon.ico` | Site favicon (204 if missing) |
| GET | `/favicon-{variant}.ico` | Favicon variant: `smile`, `frown`, or `surprised` (404 for unknown variant, 204 if file missing) |
| GET | `/logo.svg` | Site logo (204 if missing) |
