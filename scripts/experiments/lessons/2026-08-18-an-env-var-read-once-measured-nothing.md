# 2026-08-18 — an env var read once at init measured nothing (#3160)

**Cost:** ~10 min and one wasted job pair.

**What broke.** To price `ATEN_CPU_CAPABILITY=avx2`, a script alternated the
setting *inside one process* — set the var, time the resize, unset it, time it
again, five reps each — and reported the pin costing **−0.9%**, with a tidy
±0.02 s SE. The number was meaningless: torch reads that variable **once, at
import**, so every rep had run the same AVX-512 kernels. A clean, plausible,
entirely fictional null.

It self-caught only because the script printed two things beside each rep that
it did not strictly need: `torch.backends.cpu.get_cpu_capability()` (which kept
saying `AVX512` under a request for `avx2`) and a checksum of the output pixels
(identical under both settings, when the whole point was that they differ).
Re-run as alternating *processes*, the real figure is 2.18 ± 0.02 s → 1.62 ±
0.02 s, a 26% change in the opposite direction from the one measured.

The general shape is not about this variable. **A request is not a setting.**
Anything advisory — an env var, a `use_fast=`/`backend=` kwarg, a `--gres` type,
a precision request — can be ignored in silence, and an arm that records what it
*asked for* will report a contrast it never ran. The neighbouring #3146 study hit
the identical shape the same week from the other side: a processor backend that
had silently changed under it.

**Prevented?** *Partly, by construction.* The pile's provenance sidecar now
records `cpu_capability` — the value torch **resolved** — alongside
`aten_cpu_capability_requested`, so a pin that did not take is visible in the
artifact rather than assumed away. The probes do the same. There is no preflight
check for this in general: what "resolved" means differs per knob, so the
transferable rule is the habit — **read the setting back from the library, and
put the readback next to the measurement.**
