# 2026-10-15 — numpy 2.4 renamed its dispatch targets, so the numpy pin did nothing on a new numpy (#4930)

**What happened.** #4110 pinned numpy's kernels with `NPY_DISABLE_CPU_FEATURES="AVX512F AVX512_SKX"`, the
names numpy used for its AVX-512 dispatch targets up to 2.3. numpy 2.4 regrouped its x86 targets into
microarchitecture levels (`X86_V3`, `X86_V4`) plus `AVX512_ICL` and `AVX512_SPR`. numpy skips a name its
build does not dispatch and raises an `ImportWarning`, which Python hides by default. Neither old name
survived, so on numpy 2.4 or later the pin read as set in the environment and disabled nothing: an AVX-512
node kept its AVX-512 `exp`, `log`, `argsort` and `partition`, which #4110 had found differing from an AVX2
node's.

`pyproject.toml` leaves numpy unpinned, and a fresh install resolves it to 2.5. The GRID's two venvs run
numpy 2.2.6, where the old pin works. #4922's own test, which imports numpy under `-W error`, caught it on a
cloud container's fresh install, before any GRID venv moved.

**Cost.** None yet. The next venv rebuild would have brought back the Intel-vs-AMD drift #4110 removed, and every
log would still have shown the pin as set.

**The rule.** An env pin that names a library's internal identifiers is tied to that library's version.
Choose it from the version, check its *effect* rather than its *value*, and check it where a warning cannot be
hidden. Here:

- `_expcommon.numpy_kernel_pin()` reads numpy's version from its package metadata, without importing numpy,
  and returns `"AVX512F AVX512_SKX"` before 2.4 and `"X86_V4 AVX512_ICL AVX512_SPR"` from 2.4.
  `pin_cpu_kernels()` and `gridenv.sh` both use it.
- A cell's `kernels:` log line names numpy's version and the targets its ufuncs run (`numpy_targets=AVX2,SSE41`
  or `X86_V3` when pinned). Unpinned on an AVX-512 node, `AVX512_SKX` or `X86_V4` shows up there.
- `__cpu_features__` is no proof. It keeps reporting the CPU's own flags, so before 2.4 `AVX512_ICL` reads as
  on even though no kernel uses it. `numpy.lib.introspect.opt_func_info()` shows what runs.

End to end on numpy 2.5.4, an Intel and an AMD run of the same cell are bit-identical in 72 of 72 cells with the
version-chosen pin, and in 0 of 72 with the old one (`docs/experiments/2026-10-15-numpy-pin-4930/`).

**Status.** *Prevented* for the numpy versions probed (2.2.6 to 2.5.4).
`tests_lib/meta/test_cpu_kernel_pins.py` imports the installed numpy under its pin with `-W error` and fails
if any optimized ufunc runs above AVX2, so a future rename fails the suite, as this one did. That check can
only catch an *incomplete* pin on an AVX-512 machine. *Advice only* for a stale value: an exported
`NPY_DISABLE_CPU_FEATURES` still wins, so a shell that sourced `gridenv.sh` before a numpy upgrade carries the
old value into the jobs it submits. Read the `kernels:` line.
