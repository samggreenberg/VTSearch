# Authentication & Static UI

[← Back to API index](../API.md)

---

## Authentication

### Auth status

```
GET /api/auth/status
```

→ ```json
{
  "provider": "default",
  "user": "default",
  "authenticated": true,
  "login_required": false
}
```

Returns the active login provider name, current user, whether the request
is authenticated, and whether the frontend should show a login screen.
With `DefaultLoginProvider`, every request is authenticated as `"default"`.

### Login

```
POST /api/auth/login
```

**Body:** `{"username": "...", "password": "..."}`

→ Provider-specific response. With `DefaultLoginProvider`, returns 404.
With `TrivialLoginProvider`, authenticates and returns `{"ok": true, "user": "..."}`.

### Logout

```
POST /api/auth/logout
```

→ Provider-specific response. With `DefaultLoginProvider`, returns 404.

---

## Static / UI

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve the single-page application (`index.html`) |
| GET | `/favicon.ico` | Site favicon (204 if missing) |
| GET | `/favicon-{variant}.ico` | Favicon variant: `smile`, `frown`, or `surprised` (404 for unknown variant, 204 if file missing) |
| GET | `/logo.svg` | Site logo (204 if missing) |
