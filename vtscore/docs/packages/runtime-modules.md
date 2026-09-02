# Top-level runtime modules

Three small modules sit directly under `vtscore/` rather than in a
package of their own. None is big enough to need its own guide, and
together they cover "things the process needs that aren't about media":
shared file I/O for plugins, GPU backend selection, and a
single-instance lock.

| Module | Concern |
|--------|---------|
| [`vtscore/io.py`](#vtscoreio) | Read a JSON file, write a file atomically, take a cross-process file lock |
| [`vtscore/gpu_backends.py`](#vtscoregpu_backends) | Route UMAP and k-means to cuML on a usable GPU, degrade to CPU otherwise |
| [`vtscore/single_instance.py`](#vtscoresingle_instance) | Refuse a second server process on the same port |

(`vtscore/config/` also sits at the top level but is large enough to
have its own page: [`config.md`](config.md). The CLI modules likewise -
see [`cli.md`](cli.md).)

---

## `vtscore.io`

Shared file I/O for plugins that read or write server-side files. Two
patterns used to live inline across importers, exporters and sources:
the `exists()` / `is_file()` / `read_bytes()` / `json.loads()` dance
with slightly different error text every time, and the tmp-file +
`fsync` + `os.replace` ritual re-implemented per exporter. Forget one
piece of the second and a crash mid-write leaves a half-written file.

```python
def read_server_json(path, *, missing_ok: bool = False) -> Any: ...
def atomic_write_text(path, text: str) -> None: ...
def atomic_write_bytes(path, data: bytes) -> None: ...
def atomic_write_json(path, obj, *, indent: int = 2) -> None: ...
def sanitize_csv_cell(value: str) -> str: ...
def desanitize_csv_cell(value: str) -> str: ...

@contextmanager
def file_lock(path) -> Iterator[None]: ...
```

The helpers are deliberately small. Their job is to standardise the
`ValueError` text the framework surfaces to users, and to make "a future
file-writing plugin that forgets `fsync`" impossible without explicitly
working around the helper.

**CSV cells.** `sanitize_csv_cell` prefixes a value starting with `=`,
`+`, `-`, `@`, tab or carriage return with a single quote, so a label
someone typed cannot execute as a formula when the export is opened in a
spreadsheet. `desanitize_csv_cell` reverses it on re-import, stripping a leading
apostrophe only when the next character is one the sanitiser would have
escaped. One ambiguity is inherent to the scheme and unresolvable by the
reader: a value that genuinely begins `'-foo` is indistinguishable from
the sanitised form of `-foo`, and loses its apostrophe.

**`file_lock` locks a sibling, not the file.** The lock is taken on
`<path>.lock` rather than on the data file, because an atomic write
replaces the data file's inode via `os.replace` - any fd held against
the old inode would be locking a file nobody can reach any more. The
sibling's inode is stable.

It takes an in-process `threading.Lock` first and *then* the POSIX
`flock`, so threads within one process serialise even where `flock` is
unavailable. `flock` releases when the process exits, so a crash never
leaves a stale lock. On Windows (no `fcntl`) only the in-process lock
applies and cross-process protection degrades silently; VTSearch is
deployed on Linux, so this affects only the rare Windows-dev case.

---

## `vtscore.gpu_backends`

cuML (part of NVIDIA's RAPIDS suite) is an optional, GPU-only
dependency. When a usable CUDA device resolves *and* cuML imports, the
two heavyweight CPU clustering steps move to the GPU:

| Consumer | CPU | GPU |
|----------|-----|-----|
| [Projection](projection.md) UMAP | `umap-learn` | `cuml.manifold.UMAP` |
| [Coverage atlas](state.md#coverage-atlas) hierarchical k-means | `sklearn.cluster.KMeans` | `cuml.cluster.KMeans` |

```python
def cuml_enabled() -> bool: ...
def umap_fit_transform(mat, *, n_components, n_neighbors, min_dist, metric, random_state) -> np.ndarray: ...
def kmeans_fit_predict(vecs, *, n_clusters, random_state, n_init) -> tuple[np.ndarray, float | None]: ...
```

cuML's UMAP and KMeans are API-compatible with their CPU counterparts,
so call sites only swap which estimator they construct, not how they use
it. Both force `output_type="numpy"`, so downstream code sees plain
numpy arrays regardless of backend.

### Fit-time failures degrade, not just construction failures

This is the part worth knowing. cuML compiles its cuVS/raft kernels with
nvrtc **lazily, on the first `fit`** - so a broken GPU toolchain (a
CUDA-12 nvrtc handed CUDA-13 fp8 headers it can't parse) raises *during
the fit*, long after the estimator constructed cleanly. A guard around
construction alone would not catch it.

So both entry points wrap the whole construct-and-fit in one `try`. Any
cuML hiccup, whenever it happens: logs one warning, flips a
process-global kill switch, and re-runs on the CPU library. After that
`cuml_enabled()` returns `False` for the rest of the process, so the run
skips cuML entirely rather than re-paying a multi-second compile failure
on every call - which matters because the coverage atlas alone fits
k-means dozens of times. The switch resets only on a fresh interpreter.

The warning is deliberately at `warning` level and fires once: a GPU box
silently running the slow path should be visible in the logs.

### Output is not byte-identical to the CPU path

cuML is a separate CUDA-native implementation, so identical parameters
do not give identical coordinates or labels. That is safe for both
consumers because each computes its result exactly once and then freezes
it - the projection is frozen per dataset, the atlas is cached in the
dataset pickle - so the non-reproducibility never surfaces. The
*structure* (neighbourhoods, cluster topology) is preserved, which is
the whole point of these algorithms.

### Install and opt-out

`scripts/install.sh` installs cuML by default on GPU hosts, but as a
separate **best-effort** step (`vts_install_cuml`): it is a
multi-gigabyte stack on a CUDA-major-pinned separate index
(`pypi.nvidia.com`), so a slow or unreachable index, or a torch resolver
conflict, must not abort an otherwise good GPU install. For the same
reason it is kept out of the main `requirements/gpu.txt` pass;
`docker/Dockerfile.gpu` installs it in its own dedicated fail-loud
layer.

Two distinct escape hatches:

| Variable | When | Effect |
|----------|------|--------|
| `VTSEARCH_SKIP_CUML=1` | install time | Skip the host-script install step |
| `VTSEARCH_DISABLE_CUML=1` | runtime | Force the CPU libraries even though cuML is installed and the GPU is usable |

Whenever cuML is absent - skipped, failed to install, unsupported
platform - everything falls back automatically. `cuml_enabled()` also
routes its device check through `vtscore.config.resolve_device`, so it
honours `VTSEARCH_DEVICE` and the CUDA smoke test: a GPU the installed
wheels can't actually drive resolves to `"cpu"` and disables the cuML
path.

---

## `vtscore.single_instance`

A process-level lock so `python app.py` refuses to start twice on the
same port. Running the server twice in one allocation reloads the model
stack (~17 GB) and OOM-kills the SLURM job.

```python
def lock_path_for(port: int) -> str: ...
def acquire(port: int) -> IO[str]: ...

class AlreadyRunningError(RuntimeError):
    port: int
    holder_pid: str
```

`acquire` takes an exclusive `flock` and writes the holding PID into the
lockfile, so the error message can name the process that already has it.
**Keep the returned handle open for the process lifetime** - `flock`
releases when the handle closes or the process exits, which is exactly
why `flock` was chosen: a crash never leaves a stale lock behind, so
there is no "delete the pidfile and try again" recovery step.

The lock lives in `VTSEARCH_RUNDIR` if set, else the system temp dir, as
`vtsearch-<port>.lock`.

POSIX only (`fcntl`), consistent with the rest of the server, which
already relies on `/proc` and POSIX semantics.
