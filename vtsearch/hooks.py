"""Flask request-lifecycle hooks for the VTSearch app.

Holds the ``before_request`` / ``after_request`` / ``teardown_request``
handlers that used to live inline in ``app.py``: per-request id + user +
dataset/detector context resolution, server-side auth enforcement, the
in-handler marker, and the API-response ``Cache-Control`` /
``X-Request-Id`` shaping.

``app.py`` calls :func:`register_hooks` to wire them onto its module-level
``app``. Registration is done explicitly (rather than via ``@app.before_request``
decorators at import time) so the ordering the handlers run in is preserved and
obvious: Flask runs ``before_request`` handlers in registration order and
``after_request`` / ``teardown_request`` handlers in reverse registration order.
"""

import logging
import os
import time

from flask import g, request

from vtsearch.logging_config import new_request_id

#: Attribute stamped on a view function by :func:`state_sync_exempt`.
_STATE_SYNC_EXEMPT_ATTR = "_vts_state_sync_exempt"


def state_sync_exempt(view):
    """Mark ``view`` as not needing the lock-taking per-request state sync.

    A view whose handler never reads the dataset/detector proxies does not
    need the rehydration :func:`_set_request_context` performs, and keeping
    it off ``_state_lock`` stops a high-frequency poll (e.g. the jobs/active
    spinner feed) from piling up behind a long lock-holder and exhausting the
    worker's threads (2026-06-19: one stuck job parked 23 threads on the
    futex and froze the whole UI).

    **What "off the lock" covers, and what it rests on.** The marker itself
    only skips the rehydrate. The rest of :func:`_set_request_context` still
    runs for an exempt route -- the SPA sends ``X-Dataset-Id`` /
    ``X-Detector-Id`` on every request, so every request resolves them
    through ``get_context`` / ``get_detector_context``. Those two take
    ``_context_registry_lock`` (a dict lookup, never held across a call-out)
    rather than ``_state_lock``, which is what makes the exemption worth
    anything: before #3869 they took ``_state_lock``, so an exempt route
    blocked for exactly as long as the blocker ran -- a ``jobs/active`` poll
    sat 2425 ms behind a vote in the #3853 capture, and the resulting
    diagnosis blamed the GIL. If either resolver ever starts taking
    ``_state_lock`` again, this decorator stops buying anything.

    The marker lives on the **view function** rather than in a URL-prefix
    list next to the hook, so renaming a route cannot silently drop its
    exemption — which would reintroduce exactly the hang the exemption
    exists to prevent, with no test or type error to catch it.

    Apply it directly beneath the ``@bp.route`` decorator so it stamps the
    fully-wrapped view that gets registered::

        @bp.route("/api/jobs/active")
        @state_sync_exempt
        @bp.response(200, Schema)
        def active_jobs(): ...

    (Both orderings work in practice — ``functools.wraps`` copies
    ``__dict__`` — but stamping the outermost wrapper does not depend on
    every intervening decorator being well-behaved.)
    """
    setattr(view, _STATE_SYNC_EXEMPT_ATTR, True)
    return view


def _is_state_sync_exempt() -> bool:
    """Whether the matched view opted out of the lock-taking state sync."""
    from flask import current_app, request

    endpoint = request.endpoint
    if not endpoint:
        # No URL rule matched (404s, static misses): nothing to exempt.
        return False
    return getattr(current_app.view_functions.get(endpoint), _STATE_SYNC_EXEMPT_ATTR, False)


def _set_request_id():
    """Assign a unique request id to every request.

    The id is exposed on ``g.request_id`` so log records produced during
    the request automatically carry it (via the structured-logging
    ContextFilter), and echoed back in the ``X-Request-Id`` response
    header so clients can quote it in bug reports. If the caller supplied
    their own ``X-Request-Id`` header we trust it (truncated to 64 chars
    to bound log line size), which lets gateways/load balancers propagate
    end-to-end trace ids.
    """
    from flask import request

    inbound = request.headers.get("X-Request-Id")
    g.request_id = inbound[:64] if inbound else new_request_id()


def _set_user_context():
    """Populate ``g.user`` from the active LoginProvider on every request.

    A misbehaving provider must not be able to take down the server by
    raising from ``get_user``; swallow exceptions and fall back to
    ``"default"``.
    """
    from flask import request

    from vtsearch.auth import get_login_provider

    try:
        provider = get_login_provider()
        g.user = provider.get_user(request)
    except Exception:
        logging.getLogger(__name__).exception("Login provider get_user failed")
        g.user = "default"


# API paths reachable without credentials when the active provider enforces
# auth. Exactly what the SPA needs to render the login screen and log in:
# `/api/auth/status` tells it whether a login is required at all, and
# login/logout are how credentials are established/cleared in the first
# place. Deliberately exact-match (not a prefix): `/api/auth/huggingface/*`
# configures a server-wide HuggingFace token and must stay behind the gate.
_AUTH_EXEMPT_PATHS = frozenset(
    {
        "/api/auth/status",
        "/api/auth/login",
        "/api/auth/logout",
    }
)


def _enforce_auth():
    """Reject unauthenticated ``/api/*`` requests for enforcing providers.

    ``_set_user_context`` above only *identifies* the caller; this hook is
    the *authentication* step (see issue #2946): when the active provider's
    ``enforce_auth()`` is true and ``is_authenticated(request)`` is false,
    the request is aborted with a JSON 401 before any route handler (or the
    lock-taking state sync below) runs. Non-API paths (the SPA shell,
    favicons) and the ``_AUTH_EXEMPT_PATHS`` allowlist pass through.

    ``DefaultLoginProvider`` authenticates every request, so single-user
    deployments never hit the 401 branch. ``TrivialLoginProvider`` opts out
    via ``enforce_auth() -> False``. A provider that raises from either
    check fails **closed** (401): a broken enforcing provider must not
    degrade to serving everyone as anonymous — which was the bug this hook
    exists to fix. The built-in providers' checks cannot raise, so the
    fail-closed path is unreachable for stock deployments.
    """
    from flask import request

    from vtsearch.auth import get_login_provider

    path = request.path
    if not path.startswith("/api/") or path in _AUTH_EXEMPT_PATHS:
        return None
    provider = get_login_provider()
    try:
        if not provider.enforce_auth() or provider.is_authenticated(request):
            return None
    except Exception:
        logging.getLogger(__name__).exception("Login provider auth check failed; rejecting request (fail closed)")
    from vtsearch.errors import error_response

    body, status = error_response("Authentication required", 401, error_code="auth_required")
    body.status_code = status
    try:
        challenge = provider.www_authenticate()
    except Exception:
        challenge = None
    if challenge:
        body.headers["WWW-Authenticate"] = challenge
    return body


def _set_request_context():
    """Resolve per-request dataset/detector context from HTTP headers.

    If the frontend sends ``X-Dataset-Id`` or ``X-Detector-Id``, the
    corresponding context is stashed on ``g`` so that proxy objects
    (``medias``, ``good_votes``, etc.) resolve to it for the duration of
    this request, without mutating global "active" state.

    When a header is absent the proxies fall back to the thread-local /
    empty context. When a header is **present but names an unloaded id**,
    the unloaded id is stashed on ``g`` so the resolver raises
    ``DatasetNotLoadedError`` / ``DetectorNotLoadedError`` on proxy
    access (mapped to 409). Routes that never touch the proxies still
    respond normally (see logical-bug-audit H16).

    Any failure here must not 500 every subsequent request; fall back to
    the default (empty) context.
    """
    from flask import request

    from vtscore.state.core import (
        DatasetNotLoadedError,
        DetectorNotLoadedError,
        get_context,
        get_detector_context,
    )

    # Pin a marker so the request-missing-context predicate can distinguish
    # "actively inside a route handler" from "Flask test client is still
    # preserving the popped request context for inspection". Cleared in
    # teardown_request below.
    g._vts_in_request_handler = True

    try:
        # Headers (Angular HttpClient interceptor) take priority, with query
        # params as fallback for browser-native requests (<img src>,
        # <audio src>, <video src>) that bypass Angular's interceptor.
        ds_id = request.headers.get("X-Dataset-Id") or request.args.get("dataset_id")
        if ds_id:
            ctx = get_context(ds_id)
            if ctx is not None:
                g._dataset_context = ctx
            else:
                # Header refers to a dataset that isn't loaded.  Stash the id
                # so the resolver raises DatasetNotLoadedError at proxy
                # access; silent fallback to the empty context hid stale
                # results from the client (logical-bug-audit H16).  Routes
                # that never touch the dataset proxies (registry listings,
                # auth, file browser, etc.) still respond normally.
                g._unloaded_dataset_id = ds_id

        detector_id = request.headers.get("X-Detector-Id") or request.args.get("detector_id")
        if detector_id:
            det_ctx = get_detector_context(detector_id)
            if det_ctx is not None:
                g._detector_context = det_ctx
            else:
                g._unloaded_detector_id = detector_id
    except Exception:
        logging.getLogger(__name__).exception("Request context resolution failed")

    # If the active (dataset, detector) pair has changed since the detector's
    # cid-keyed vote dicts were last derived, rehydrate them from the on-disk
    # labelset against the active dataset's medias.  Media ids are dataset-
    # specific, so without this the left-pane shows stale cids from the
    # previous dataset as if they were votes in the current one.
    #
    # Also drop the detector's cached MLP / per-label embedding cache when
    # the active dataset's embedder differs from the one the MLP was
    # trained on; scoring with a cross-space MLP either crashes (different
    # dim) or silently produces garbage labels (same dim).  See
    # logical-bug-audit H5.
    # Skip the lock-taking state-sync for endpoints whose handlers never
    # read the proxies (e.g. the jobs/active spinner poll), so a frequent
    # poll cannot queue on `_state_lock` behind a long-running job. The
    # opt-out is a `@state_sync_exempt` marker on the view itself. The header
    # resolution above is already lock-free with respect to `_state_lock`
    # (see that decorator's docstring), so skipping this block is the last
    # thing an exempt route needs to stay off the lock entirely.
    if not _is_state_sync_exempt():
        try:
            from vtscore.detectors.dataset_sync import (
                ensure_detector_model_matches_active_embedder,
                ensure_votes_match_active_dataset,
            )

            ensure_votes_match_active_dataset()
            ensure_detector_model_matches_active_embedder()
        except (DatasetNotLoadedError, DetectorNotLoadedError):
            # The route handler will hit the same error when it touches
            # the proxies; the global error handler turns it into a 409.
            pass
        except Exception:
            logging.getLogger(__name__).exception("Vote rehydrate failed")


def _clear_in_request_handler_marker(exc):  # noqa: ARG001
    """Clear the in-handler marker so the request-missing predicate
    doesn't fire while Flask's test client is preserving a popped
    request context after the response."""
    from flask import g, has_request_context

    if has_request_context():
        g._vts_in_request_handler = False


def _no_cache_api(response):
    """Prevent browsers from caching mutable API responses.

    Without this, concurrent or rapid-fire fetches to the same endpoint
    (e.g. ``/api/datasets/registry``) can receive stale cached data,
    causing the frontend to miss newly loaded datasets.

    Endpoints serving genuinely immutable data (e.g. frozen projection
    tiles) opt out by setting their own ``Cache-Control`` in the view; we
    defer to that rather than clobbering it with ``no-store``.
    """
    from flask import request

    if request.path.startswith("/api/") and "Cache-Control" not in response.headers:
        response.headers["Cache-Control"] = "no-store"
    return response


def _echo_request_id(response):
    """Surface the per-request id on every response so clients can quote it."""
    rid = getattr(g, "request_id", None)
    if rid:
        response.headers["X-Request-Id"] = rid
    return response


#: Env var naming the slow-request threshold, in milliseconds.
_SLOW_REQUEST_MS_ENV = "VTSEARCH_SLOW_REQUEST_MS"

#: Default threshold. A request slower than this is logged at WARNING.
_DEFAULT_SLOW_REQUEST_MS = 1000.0


def slow_request_threshold_ms() -> float:
    """Milliseconds above which a request counts as slow.

    Read per call rather than captured at import, so a restart with a
    different ``VTSEARCH_SLOW_REQUEST_MS`` takes effect and a test can lower
    the bar without reloading the module. An unparseable value falls back to
    the default rather than raising: bad configuration must not be able to
    fault every request.
    """
    raw = os.environ.get(_SLOW_REQUEST_MS_ENV)
    if raw is None:
        return _DEFAULT_SLOW_REQUEST_MS
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_SLOW_REQUEST_MS


def _set_request_start() -> None:
    """Stamp the counters :func:`_log_slow_request` reports a difference of.

    Wall time alone cannot say what a slow request was doing, which is what
    stalled #3853 for a week: a 4918ms vote POST and a 4918ms wait on a lock
    look identical to a ``perf_counter`` pair.  CPU time separates "did the
    work" from "blocked or was descheduled", and the GC total separates "a
    collection froze every thread" from both.
    """
    from vtscore.concurrency.stalls import gc_pause_ms_total, thread_cpu_ms

    g.request_start = time.perf_counter()
    g.request_cpu_start = thread_cpu_ms()
    g.request_gc_start = gc_pause_ms_total()


def _log_slow_request(response):
    """Log any request slower than the threshold, at WARNING.

    **WARNING, not INFO, on purpose.** ``VTSEARCH_LOG_LEVEL`` defaults to
    ``WARNING``, so anything quieter would leave a stall with no trace on a
    default deployment -- which is why the 6-20s post-vote stalls in #3853
    could never be diagnosed after the fact.

    The per-request id is included because the browser reads that same id off
    the ``X-Request-Id`` response header (:func:`_echo_request_id`), so a
    stall seen in DevTools can be matched to the server line explaining it.

    Below the threshold, and only when the logger is enabled for INFO, the
    same figures are written as a ``request trace`` line so a diagnostic run
    has the whole chain to add up rather than just its outliers.  The two
    prefixes are distinct (not ``request:`` and ``slow request:``) so a log
    reader can match either one without matching the other.
    """
    from vtscore.concurrency.stalls import gc_pause_ms_total, thread_cpu_ms

    start = getattr(g, "request_start", None)
    if start is None:
        # No start stamp: the request failed before ``_set_request_start``
        # ran, or a test built a bare request context. Nothing to report.
        return response
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    cpu_ms = max(0.0, thread_cpu_ms() - getattr(g, "request_cpu_start", 0.0))
    gc_ms = max(0.0, gc_pause_ms_total() - getattr(g, "request_gc_start", 0.0))
    logger = logging.getLogger(__name__)
    if elapsed_ms >= slow_request_threshold_ms():
        logger.warning(
            "slow request: %s %s -> %s in %.0fms cpu=%.0fms gc=%.0fms (request_id=%s)",
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
            cpu_ms,
            gc_ms,
            getattr(g, "request_id", None),
        )
    elif logger.isEnabledFor(logging.INFO):
        # Every request, at INFO, so a *felt* pause that no single request is
        # long enough to explain can still be accounted for. #3853's residue
        # is exactly that shape: a vote's chain of ~10 requests, a retrain and
        # a collection each costing less than any threshold, summing to the
        # half-second the reviewer feels.  A per-request bar cannot see a sum
        # by construction, so the analysis has to be done over the whole
        # trace; this is the line ``analyze_app_log.py`` adds up.
        logger.info(
            "request trace: %s %s -> %s in %.0fms cpu=%.0fms gc=%.0fms (request_id=%s)",
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
            cpu_ms,
            gc_ms,
            getattr(g, "request_id", None),
        )
    return response


def register_hooks(app) -> None:
    """Register all request-lifecycle hooks on ``app``.

    Ordering is preserved from the original inline registration in
    ``app.py``: ``before_request`` handlers fire in registration order,
    ``after_request`` / ``teardown_request`` handlers in reverse.
    """
    app.before_request(_set_request_start)
    app.before_request(_set_request_id)
    app.before_request(_set_user_context)
    app.before_request(_enforce_auth)
    app.before_request(_set_request_context)
    app.teardown_request(_clear_in_request_handler_marker)
    app.after_request(_log_slow_request)
    app.after_request(_no_cache_api)
    app.after_request(_echo_request_id)
