# Auto-Find settings tab + results exporter wiring

**Status:** Core feature is in place; remaining work is the open follow-ups below (UI caller for `/api/auto-detect`, streaming-export settings fallback, and minor exporter polish).

## Open follow-ups

- The `/api/auto-detect` route currently has no first-party UI caller (the
  feature is CLI-driven); the server-side auto-export is wired and tested so it
  works the moment a UI flow calls the route.
- [ ] #2384 — Wire the settings-based exporter fallback into the streaming CLI export path
- Scheduling stays external (cron / systemd timers around
  `python app.py --autodetect --user <name> --api-key <key>`); no built-in
  scheduler.
- [ ] #2385 — Support template variables in the email exporter subject

### Known limitation: cross-user stale references
Deleting a detector scrubs only the *deleting* user's Auto-Find list; there is no way to enumerate other users' settings files (user data dirs are login-provider-defined), so another user's list may keep a stale name. That staleness is reported, not hidden: their next run lists it under `missing_detectors`.
