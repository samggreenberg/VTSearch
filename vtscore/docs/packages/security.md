# `vtscore.security`

Defensive helpers used at every external-input boundary: path validation
to prevent directory traversal, URL validation to block SSRF, hardened
archive extraction, and an allowlist-based pickle unpickler so loading a
`.pkl` dataset can't lead to arbitrary code execution. Every module here
is app-uncoupled - they take their inputs explicitly and operate on bytes
and strings, not request objects or settings. `login.py` carries the
identity abstraction the path checks depend on.

Related docs: [`state.md`](state.md) for the contexts that hold the
deserialised dataset artefacts; [`concurrency.md`](concurrency.md) for
the load pipeline that calls these helpers during dataset import.

**Import from the defining module.** `vtscore/security/` has no
`__init__.py` - it is a PEP 420 implicit namespace package, so
`from vtscore.security import validate_url` raises `ImportError`. Import
from `vtscore.security.path_validation`, `.url_validation`, or `.pickle`
as the snippets below do.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/security/path_validation.py` | Server file-path validation and per-user confinement |
| `vtscore/security/url_validation.py` | SSRF guard, validated fetch/stream helpers, browser-URL checks |
| `vtscore/security/origin_validation.py` | Confinement for origin dicts arriving from outside the server |
| `vtscore/security/pickle.py` | Allowlist unpickler for dataset deserialisation |
| `vtscore/security/archive.py` | Hardened single-member tar extraction (`safe_tar_extract`) |
| `vtscore/security/hf_auth.py` | In-memory HuggingFace credential store + `GatedResourceError` |
| `vtscore/security/login.py` | `LoginProvider`: how an identity is established, and which subtree it is confined to |

- [Login providers](#login-providers)
- [Path validation](#path-validation)
  - [Media-carried file references](#media_file_read_roots-and-resolve_media_file_pathfilepath_str)
  - [Example media](#example_media_dir)
  - [Origin confinement](#origin-confinement)
- [URL validation](#url-validation)
  - [`validate_url` (SSRF guard)](#validate_url-ssrf-guard)
  - [Fetching: `open_validated_stream` / `fetch_validated_url`](#fetching-open_validated_stream--fetch_validated_url)
  - [Browser-URL validation](#browser-url-validation)
- [Archive extraction](#archive-extraction)
- [HuggingFace credentials](#huggingface-credentials)
- [Pickle safety](#pickle-safety)
- [Why no `find_class` shim is needed](#why-no-find_class-shim-is-needed)

---

## Login providers

`vtscore/security/login.py` answers the two questions the path checks
below are built on: **how** an identity is established, and **where** that
identity's data lives. `vtscore.state.current_user` resolves *which*
username the work belongs to; this module maps that username to a
directory, and that directory is the confinement root.

| Name | Description |
|------|-------------|
| `LoginProvider` | ABC. Subclasses implement `get_user(request)` and `is_authenticated(request)`; override `get_user_data_dir(username, base_data_dir)` to opt into per-user confinement, and `login_required()` / `status_dict(request)` for the app's auth UI |
| `DefaultLoginProvider` | Single-user, no auth. Every caller is `"default"`; the data dir is `DATA_DIR` itself, so nothing is confined |
| `set_login_provider(p)` / `get_login_provider()` | The process-wide active provider. `DefaultLoginProvider()` until something replaces it |
| `get_user_data_dir(username=None)` | `provider.get_user_data_dir(username or get_current_user(), DATA_DIR)` |
| `is_safe_username(name)` | `True` when *name* is safe as a path component: matches `[A-Za-z0-9._-]+` and is not an all-dots traversal segment |

The `request` argument is typed `Any` and never introspected by the
library - it is handed straight back to the provider that asked for it.
That is what lets the abstraction be Flask-free while the app's own
providers (`TrivialLoginProvider`, `ApiKeyLoginProvider` in
`vtsearch/auth/`, which read `flask.session` and the `Authorization`
header) build on it. `vtsearch.auth` re-exports every name in the table,
so there is exactly one active provider per process however you reach it.

Embedding `vtscore` without the app and want per-user confinement? Register
a provider; every path check in the library starts enforcing it:

```python
from pathlib import Path
from vtscore.security.login import LoginProvider, set_login_provider

class MyProvider(LoginProvider):
    name = "mine"

    def get_user(self, request):
        return "alice"

    def is_authenticated(self, request):
        return True

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:
        return base_data_dir / username

set_login_provider(MyProvider())
```

Validate any username you did not construct yourself with
`is_safe_username()` before returning it from `get_user()`: the name
becomes a path component, and `..` segments in a confinement root are
*collapsed* by `Path.resolve()` rather than rejected - which silently
widens the sandbox for every server-file importer and exporter.

## Path validation

`vtscore/security/path_validation.py` exposes the helpers used by every
server-file importer / exporter, every server-path field on a plugin
form, and every serve-time read of a path that arrived *on a media*.

| Function | Description |
|----------|-------------|
| `get_file_access_base_dir() -> Path \| None` | Resolve the per-user base directory; `None` (unrestricted) in single-user mode |
| `example_media_dir() -> Path` | The directory detector example media is cached in, per user |
| `validate_server_filepath(filepath_str, base_dir=None) -> Path` | Resolve and (when `base_dir` is given) assert containment; raises on escape |
| `media_file_read_roots() -> list[Path] \| None` | Roots a media-carried file reference may be read from; `None` (unrestricted) in single-user mode |
| `resolve_media_file_path(filepath_str) -> Path \| None` | Confine a media-carried file reference; `None` (rather than a raise) when it escapes |
| `sanitize_template_value(value) -> str` | Replace path separators / `..` tokens with `_` |
| `rglob_follow_symlinks(root, pattern) -> list[Path]` | `Path.rglob`-equivalent that descends into symlinked directories |
| `glob_top_level(root, pattern) -> list[Path]` | Non-recursive variant; direct children of `root` only |

### `get_file_access_base_dir()`

In single-user / no-auth mode (`DefaultLoginProvider`) it returns `None`,
which tells `validate_server_filepath` to apply **no** confinement: the
lone trusted user may read from and write to any server-readable path.
There is no per-user boundary to protect, so the app does not impose one.
In multi-user mode it returns the current user's data directory so each
user is confined to their own `data/<username>/` subtree. Which of the two
applies is decided entirely by the registered
[login provider](#login-providers) - there is no separate switch, and
`example_media_dir()` below reuses this split rather than re-deriving it.

### `validate_server_filepath(filepath_str, base_dir=None)`

```python
from vtscore.security.path_validation import validate_server_filepath, get_file_access_base_dir

resolved = validate_server_filepath(user_supplied_path, get_file_access_base_dir())
# resolved is the canonical Path; reading from it is safe
```

Contract: when `base_dir` is `None` (the single-user / no-auth case) the
path is **unrestricted** - relative paths resolve against the process CWD,
absolute paths are used as-is, and the canonical `Path.resolve()` is
returned with no containment check. When `base_dir` is a path (multi-user)
relative paths resolve against it, absolute are used as-is, `Path.resolve()`
is called (follows symlinks, normalises `..`), and
`resolved.relative_to(base_resolved)` is checked - if it raises
`ValueError`, the path escaped and we re-raise. Symlinks are followed
during resolution, so a symlinked file inside `base_dir` whose target is
outside is rejected.

### `media_file_read_roots()` and `resolve_media_file_path(filepath_str)`

`validate_server_filepath` guards paths the *user* just typed. A media
carries paths of its own - `media_path`, a lazy clip's source path, an
archive-member archive path - and those are equally untrusted, because
they are copied verbatim out of a loaded dataset pickle and whatever they
point at is served straight back to the requester. They are the
filesystem twin of the `media_url` SSRF hole, so they get a matching
guard, and every serve-time read goes through it:

```python
from vtscore.security.path_validation import resolve_media_file_path

path = resolve_media_file_path(media["media_path"])
if path is not None and path.exists():
    return path.read_bytes()      # None -> serve nothing, no request-time raise
```

`media_file_read_roots()` decides what "inside" means. In single-user
mode it returns `None` and `resolve_media_file_path` is a pass-through,
matching `get_file_access_base_dir`. In multi-user mode the allowed roots
are the user's data dir **plus** `DATA_DIR`: demo datasets extract into
the shared dir as *siblings* of the per-user dirs
(`data/ESC-50-master/audio`), so confining to `data/<username>/` alone
would make every thin demo dataset unservable. The boundary this draws is
"inside the app's data tree" - enough to stop `/etc/shadow` and every
other server file, but it does leave a crafted reference able to name
another user's subtree, since user dirs are siblings under `DATA_DIR` and
there is no way to enumerate them.

Unlike `validate_server_filepath` it returns `None` instead of raising:
these are serve-time reads inside a resolution chain, and falling through
to the next step is the right behaviour for a reference that shouldn't be
honoured.

### `example_media_dir()`

The single definition of where a detector's media exemplars are cached:
`get_file_access_base_dir()` when it is set, else `DATA_DIR`, plus
`example_media/`. Every writer (the upload / from-media-id /
datasource-import routes, the browse-source copy) and every reader (media
seeding, label building, the `example_media` sentinel resolver) goes
through it.

It exists because splitting those two halves is invisible in single-user
mode and silently lossy in multi-user mode: uploads once landed in
`data/<username>/example_media/` while the readers looked in
`data/example_media/`, so an uploaded exemplar never became a vote and
its label could never be thumbnailed, embedded, or exported (issue
#3102). Resolve the directory here, never by re-spelling
`DATA_DIR / "example_media"`.

### `sanitize_template_value(value)`

Server-side sync sources accept admin-defined path templates like
`data/labels/{detector_name}.json` and substitute user-controlled values
at runtime. Without sanitisation, a `detector_name` of
`../../etc/passwd` would let the substitution escape. The helper
rewrites `/`, `\`, and `\0` to `_`, and collapses empty / `.` / `..` to
`_`:

```python
from vtscore.security.path_validation import sanitize_template_value

sanitize_template_value("safe_name")         # "safe_name"
sanitize_template_value("../../etc/passwd")  # "______etc_passwd"
sanitize_template_value("")                  # "_"
sanitize_template_value("..")                # "_"
```

Defence-in-depth - the template engine should still call
`validate_server_filepath` on the final substituted path.

### `rglob_follow_symlinks` and `glob_top_level`

`Path.rglob()` does **not** descend into symlinked directories - media
files inside symlinked sub-folders are silently skipped during dataset
import. `rglob_follow_symlinks(root, pattern)` uses
`os.walk(followlinks=True)` to traverse symlinked directory trees;
`glob_top_level(root, pattern)` is the non-recursive variant (direct
children only, file symlinks followed). Neither applies containment
itself - if you need it after a glob, pass each result through
`validate_server_filepath`.

### Origin confinement

`vtscore/security/origin_validation.py` guards the flows that accept a
**whole origin dict** from outside the server - a request body
(`POST /api/example-sort-origin`), or a detector JSON's saved media
examples. An origin is normally stamped by the server at import time and
trusted afterwards; one that arrives from outside has not been, and
resolving it re-runs filesystem or network access from user-supplied
params.

```python
def confine_origin_params(origin: Any) -> Any: ...
```

Two design choices are worth knowing before you call it:

- **Every string param is checked, not just path-shaped ones.** Params
  are untyped, and the tokens that matter (`..`, `.`, `~`) carry no path
  separator to key off. Checking a non-path is harmless - a plain
  relative name resolves inside the user's own directory and passes - so
  the check deliberately errs towards validating too much.
- **It returns a confined *copy*, not a boolean.** The validator anchors
  a relative path at the user's data dir while the consuming source
  would anchor it at the process CWD, so a bare pass/fail would approve
  one path and open another. Only the params the source factories
  actually resolve as filesystem paths (`_PATH_PARAM_KEYS`) are
  rewritten - turning an opaque key into an absolute path would corrupt
  it. **Consume the returned origin, not the input.**

URL-valued params are deliberately *not* path-checked here. They are
re-validated with `validate_url` at fetch time by the URL-backed sources
(and the downloader re-checks every redirect hop), and running them
through the path validator would spuriously reject them in multi-user
mode.

The same validate-then-use-a-different-path trap exists one level down,
which is why `confine_server_filepath(filepath_str, base_dir)` exists
alongside `validate_server_filepath`: it hands back the canonical path
the check actually approved. Store and forward *that* string. With
`base_dir=None` (single-user, no auth) both anchors are already the
process CWD, so the input comes back verbatim and stored origins stay
relative and portable across checkouts.

---

## URL validation

`vtscore/security/url_validation.py` exposes two guards. They look
similar and defend against different things, so pick by **who makes the
request**: `validate_url` for URLs *the server* fetches, and
`validate_browser_url` for URLs *the user's browser* opens.

Alongside them sit the two fetch primitives every server-side fetch is
meant to go through, `open_validated_stream` and `fetch_validated_url`.
A single up-front `validate_url` is not enough on its own - a public URL
can `302` to an internal host - so the redirect chain has to be walked by
hand with each hop re-checked.

### `validate_url` (SSRF guard)

For SSRF protection on outbound HTTP requests:

```python
def validate_url(url: str) -> str:
    """Raises ValueError if the URL uses a non-HTTP(S) scheme, has no
    hostname, or resolves to a private/internal IP address."""
```

Checks: (1) scheme must be `http` or `https`; (2) hostname must be
present; (3) `socket.getaddrinfo` resolves the hostname to all its IPs
(A and AAAA), and **every** resolved address must be publicly routable.
An address is rejected when any of these `ipaddress`-module predicates
hold:

| Predicate | Examples |
|-----------|----------|
| `is_private` | `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `fd00::/8` |
| `is_loopback` | `127.0.0.0/8`, `::1` |
| `is_reserved` | `240.0.0.0/4` and similar |
| `is_link_local` | `169.254.0.0/16` (incl. cloud metadata `169.254.169.254`), `fe80::/10` |
| `is_multicast` | `224.0.0.0/4`, `ff00::/8` |
| `is_unspecified` | `0.0.0.0`, `::` |

This blocks SSRF attacks targeting AWS metadata, internal admin
endpoints, etc. An unparseable IP is also treated as unsafe - defensive
`True` rather than silently allowing the address through.

```python
from vtscore.security.url_validation import validate_url

validate_url("https://example.com/data.json")            # → original URL
validate_url("http://169.254.169.254/latest/meta-data/") # raises ValueError
validate_url("file:///etc/passwd")                       # raises ValueError
validate_url("https://10.0.0.1/")                        # raises ValueError
```

### Fetching: `open_validated_stream` / `fetch_validated_url`

```python
def open_validated_stream(session, url, *, headers_for_url=None,
                          timeout=(10, 60)) -> requests.Response:
    """GET an already-validated url with allow_redirects=False, re-running
    validate_url on every hop.  Returns the final response; caller closes."""

def fetch_validated_url(url: str, *, timeout=(10, 30)) -> bytes:
    """validate_url + open_validated_stream + raise_for_status, returning
    the whole body."""
```

`open_validated_stream` deliberately does **not** validate its first URL:
the caller owns that check, and this covers only the hops the caller
cannot see. That split is what lets the resuming downloader validate once
and then retry a partial transfer without paying a fresh DNS resolve per
attempt (and without a transient resolver failure turning a retryable
connection error into a hard `ValueError`). `headers_for_url` is
recomputed per hop, so a credential scoped to one host - the HuggingFace
bearer token, say - is never replayed to a redirect target on another.

`fetch_validated_url` is the whole-body convenience wrapper for fetch
sites that want bytes rather than a stream to spool to disk. It is what
`vtscore.media.base._fetch_media_url` calls, so a media's `media_url` -
which can arrive verbatim from a loaded pickle and whose bytes are served
straight back to the requester - cannot name `file:///etc/passwd` or an
internal service. (It previously used `urllib.request.urlopen`, whose
default opener services `file://` and `ftp://` and obliged.)

Users of these primitives: the dataset downloader
(`vtscore/datasets/downloader/core.py`) and the `media_url` fallback in
`vtscore/media/base.py`. A new outbound fetch belongs here too rather
than reaching for `requests.get` / `urlopen` directly.

### Browser-URL validation

```python
def validate_browser_url(url: str) -> str:
    """Raises ValueError if the URL is empty, contains whitespace or
    control characters, uses a non-HTTP(S) scheme, or has no hostname."""
```

Used for the `open_url` an exporter can return so the frontend opens a
third-party page in a new tab (see
[`exporters`](exporters.md#open_url-handing-the-user-off-to-another-site)).

This is deliberately **not** the SSRF guard. The fetch is made by the
user's browser, not by us, so no server-side request exists to forge:
resolving the hostname and refusing private IPs would block a perfectly
reasonable `http://localhost:9000/viewer` companion app while buying no
protection at all. What it enforces instead is the part that *is*
dangerous once a string reaches `window.open` - the scheme. Only `http`
and `https` pass; `javascript:`, `data:`, `file:` are rejected. Embedded
whitespace and control characters are rejected too, since they let a URL
render as one target while resolving to another.

It makes no network call, so it is also free to run on every export.

```python
from vtscore.security.url_validation import validate_browser_url

validate_browser_url("https://example.com/r?ids=a,b")  # -> original URL
validate_browser_url("http://localhost:9000/viewer")   # -> allowed (browser-side)
validate_browser_url("javascript:alert(1)")            # raises ValueError
validate_browser_url("https://exa mple.com/")          # raises ValueError
```

The frontend re-checks the scheme before calling `window.open`, and
opens with `noopener` so the destination can't navigate the VTSearch tab
(reverse tabnabbing).

**Caveat:** DNS rebinding attacks are not addressed. A hostname that
resolves to a public IP at validation time and a private IP at fetch
time will pass `validate_url`. For rebinding protection, resolve to an
IP yourself and pass that IP (plus a `Host:` header) to the HTTP client.

---

## Archive extraction

`vtscore/security/archive.py` is the single audited path for pulling a
member out of a tar, so no call site re-implements traversal protection
by hand.

```python
def safe_tar_extract(tar: TarFile, member: TarInfo, dest: str | Path) -> None: ...
```

On interpreters that ship PEP 706 extraction filters
(`tarfile.data_filter`) every member goes through the strict `data`
filter, which strips a leading `/` so absolute names land *inside* the
destination, refuses `..` traversal, and refuses symlink / hardlink
members pointing outside. On older interpreters `_reject_unsafe_member`
reproduces the same three guarantees by hand before anything is written.
Either way the caller gets identical behaviour, so never call
`tar.extract` / `tar.extractall` directly.

---

## HuggingFace credentials

`vtscore/security/hf_auth.py` holds the active HuggingFace OAuth
credential **in memory only** - it is never written to settings or to
disk, and a restart signs the user out.

| Name | Description |
|------|-------------|
| `HFCredential` | Frozen dataclass: `access_token`, `username`, `expires_at`, `scopes`, plus `is_expired()` |
| `set_credential(token, *, username="", expires_at=None, scopes="")` / `clear_credential()` | Store / drop the active credential (lock-guarded) |
| `get_token()` / `is_authenticated()` / `get_status()` | Read the token, a boolean, or a JSON-serialisable snapshot for the UI |
| `auth_header_for_url(url)` | `{"Authorization": "Bearer …"}` **iff** *url* targets the Hub, else `{}` |
| `GatedResourceError` | Raised when a download fails because the resource is gated; carries `url` and `status` |

`auth_header_for_url` is the one to use on every request. It returns an
empty dict for any non-Hub host, so callers can merge it in
unconditionally without leaking the token to a third-party CDN or a
redirect target - which matters because Hub downloads redirect to signed
Xet URLs that carry their own authorization and neither need nor should
see the bearer token.

`GatedResourceError` exists as a distinct type because the frontend keys
off it to offer a "Sign in with HuggingFace" affordance; raise it rather
than a generic error when a 401/403 means "gated", not "broken".

---

## Pickle safety

`vtscore/security/pickle.py` provides a restricted unpickler used for
every `.pkl` dataset load. Loading a maliciously-crafted pickle
otherwise allows arbitrary code execution (`pickle.loads` will
instantiate `os.system` etc. without complaint), which is unacceptable
for any input arriving over HTTP or from an untrusted filesystem.

### The allowlist contract

`_PICKLE_SAFE_CLASSES` (`pickle.py`) enumerates every
`(module, name)` pair the unpickler will instantiate. Plain Python
primitives (`int`, `float`, `str`, `None`, `bool`, `dict`, `list`,
`tuple`) are handled by pickle's opcodes directly and never trigger
`find_class`.

| Module / class | Why |
|----------------|-----|
| `builtins.{set, frozenset, bytes, bytearray, complex}` | Container subclasses used inside VTSearch pickles |
| `collections.OrderedDict` | Used when round-tripping ordered dicts |
| `numpy.ndarray`, `numpy.dtype` | Embedding arrays |
| `numpy.core.multiarray._reconstruct`, `scalar` | numpy's `__reduce__` helpers (legacy module path) |
| `numpy._core.multiarray._reconstruct`, `scalar` | Same helpers under numpy's post-1.25 module rename |
| `numpy.core.numeric._frombuffer`, `numpy._core.numeric._frombuffer` | Used by numpy's pickle protocol for some dtypes |

Anything else - `os.system`, `subprocess.Popen`, `builtins.eval`,
`vtsearch.routes.something` - is refused with
`pickle.UnpicklingError`.

### `RestrictedUnpickler` and `safe_pickle_load`

```python
from vtscore.security.pickle import safe_pickle_load

with open("dataset.pkl", "rb") as f:
    medias, embeddings = safe_pickle_load(f)
# Same as pickle.load(f) but rejects non-allowlisted classes.
```

`safe_pickle_load(f, **kwargs)` is a drop-in replacement for
`pickle.load(f)`. It uses `RestrictedUnpickler`, which overrides
`find_class` to enforce the allowlist:

```python
# vtscore/security/pickle.py
def find_class(self, module: str, name: str) -> Any:
    if (module, name) in _PICKLE_SAFE_CLASSES:
        return super().find_class(module, name)
    raise pickle.UnpicklingError(
        f"Forbidden pickle class: {module}.{name}. "
        f"Only plain Python types and numpy arrays are allowed."
    )
```

### `peek_pickle_dataset_summary(f)`

A specialised unpickler for the dataset-upload preview flow. The same
allowlist applies, but four opcodes are stubbed out so the loader can
extract just the structural shape without materialising embeddings or
media bytes: `BINFLOAT` (reads 8 bytes, appends `None`),
`BINBYTES` / `BINBYTES8` / `SHORT_BINBYTES` (reads and discards the
payload, appends `b""`), `APPEND` / `APPENDS` (drops the value(s) that
would have been appended).

The outer dict structure is materialised (so you can index into
`data["medias"][0]["media_type"]`), but embedding lists are empty and inline
media-byte blobs are `b""`. For a multi-GB dataset upload this turns a
30-second `pickle.load` into a sub-second peek.

```python
from vtscore.security.pickle import peek_pickle_dataset_summary

with open(uploaded_pkl, "rb") as f:
    summary = peek_pickle_dataset_summary(f)
n_medias = len(summary["medias"])
media_type = summary["medias"][0]["media_type"] if n_medias else None
```

The peek result is **only** safe for inspecting structure. Code that
touches a value beyond the structural shape will see stub values and
produce nonsense.

---

## Why no `find_class` shim is needed

A common defensive pattern is to install a `find_class` shim that
translates legacy module paths (e.g.
`vtsearch.models.foo` → `vtscore.detectors.foo`) at load time so old
pickles keep working after a refactor. **`vtscore` deliberately does
not do this.**

The allowlist already encodes the contract that **no `vtsearch.*` or
`vtscore.*` class reference can appear in a sanctioned pickle.**
VTSearch pickles are by construction limited to plain Python containers
and numpy arrays (see the "No Persisted Vectors or MLPs" rule in
`CLAUDE.md`: embeddings are persisted as numpy arrays inside the media
dicts, never as `vtsearch.*` objects). If a pickle contains a reference
to any `vtsearch.*` or `vtscore.*` class, that pickle was not produced
by a sanctioned code path - the right behaviour is to refuse it, not
to silently rewrite the class name.

This invariant means renaming, moving, or deleting a `vtscore.*` class
**cannot break** on-disk pickles, because no on-disk pickle ever held
one. An attacker who crafts a pickle referencing `vtsearch.cli.system`
gets the same `UnpicklingError` as one trying to inject `os.system` -
no fast path lets internal classes through.

If you find yourself wanting to add a `vtsearch.*` or `vtscore.*` class
to the allowlist, stop: that would weaken the invariant. The right fix
is to make the pickle's producer emit a plain dict (or numpy array)
instead of an instance of that class.
