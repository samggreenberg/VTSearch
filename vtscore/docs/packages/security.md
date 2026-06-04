# `vtscore.security`

Defensive helpers used at every external-input boundary: path validation
to prevent directory traversal, URL validation to block SSRF, and an
allowlist-based pickle unpickler so loading a `.pkl` dataset can't lead
to arbitrary code execution. The three modules
(`path_validation.py`, `url_validation.py`, `pickle.py`) have no app
coupling - they take their inputs explicitly and operate on bytes and
strings, not request objects or settings.

Related docs: [`state.md`](state.md) for the contexts that hold the
deserialised dataset artefacts; [`concurrency.md`](concurrency.md) for
the load pipeline that calls these helpers during dataset import.

## Contents

- [Path validation](#path-validation)
- [URL validation](#url-validation)
- [Pickle safety](#pickle-safety)
- [Why no `find_class` shim is needed](#why-no-find_class-shim-is-needed)

---

## Path validation

`vtscore/security/path_validation.py` exposes four helpers used by every
server-file importer / exporter and every server-path field on a plugin
form.

| Function | Description |
|----------|-------------|
| `get_file_access_base_dir() -> Path \| None` | Resolve the per-user base directory; `None` (unrestricted) in single-user mode |
| `validate_server_filepath(filepath_str, base_dir=None) -> Path` | Resolve and (when `base_dir` is given) assert containment; raises on escape |
| `sanitize_template_value(value) -> str` | Replace path separators / `..` tokens with `_` |
| `rglob_follow_symlinks(root, pattern) -> list[Path]` | `Path.rglob`-equivalent that descends into symlinked directories |
| `glob_top_level(root, pattern) -> list[Path]` | Non-recursive variant; direct children of `root` only |

### `get_file_access_base_dir()`

In single-user / no-auth mode (`DefaultLoginProvider`) it returns `None`,
which tells `validate_server_filepath` to apply **no** confinement: the
lone trusted user may read from and write to any server-readable path.
There is no per-user boundary to protect, so the app does not impose one.
In multi-user mode it returns the current user's data directory so each
user is confined to their own `data/<username>/` subtree. This is the only
function in the package that imports `vtsearch.auth`.

### `validate_server_filepath(filepath_str, base_dir=None)`

```python
from vtscore.security import validate_server_filepath, get_file_access_base_dir

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

### `sanitize_template_value(value)`

Server-side sync sources accept admin-defined path templates like
`data/labels/{detector_name}.json` and substitute user-controlled values
at runtime. Without sanitisation, a `detector_name` of
`../../etc/passwd` would let the substitution escape. The helper
rewrites `/`, `\`, and `\0` to `_`, and collapses empty / `.` / `..` to
`_`:

```python
from vtscore.security import sanitize_template_value

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

---

## URL validation

`vtscore/security/url_validation.py` exposes one function for SSRF
protection on outbound HTTP requests:

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
from vtscore.security import validate_url

validate_url("https://example.com/data.json")            # → original URL
validate_url("http://169.254.169.254/latest/meta-data/") # raises ValueError
validate_url("file:///etc/passwd")                       # raises ValueError
validate_url("https://10.0.0.1/")                        # raises ValueError
```

**Caveat:** DNS rebinding attacks are not addressed. A hostname that
resolves to a public IP at validation time and a private IP at fetch
time will pass `validate_url`. For rebinding protection, resolve to an
IP yourself and pass that IP (plus a `Host:` header) to the HTTP client.

---

## Pickle safety

`vtscore/security/pickle.py` provides a restricted unpickler used for
every `.pkl` dataset load. Loading a maliciously-crafted pickle
otherwise allows arbitrary code execution (`pickle.loads` will
instantiate `os.system` etc. without complaint), which is unacceptable
for any input arriving over HTTP or from an untrusted filesystem.

### The allowlist contract

`_PICKLE_SAFE_CLASSES` (`pickle.py:17`) enumerates every
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
from vtscore.security import safe_pickle_load

with open("dataset.pkl", "rb") as f:
    medias, embeddings = safe_pickle_load(f)
# Same as pickle.load(f) but rejects non-allowlisted classes.
```

`safe_pickle_load(f, **kwargs)` is a drop-in replacement for
`pickle.load(f)`. It uses `RestrictedUnpickler`, which overrides
`find_class` to enforce the allowlist:

```python
# vtscore/security/pickle.py:48
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
from vtscore.security import peek_pickle_dataset_summary

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
