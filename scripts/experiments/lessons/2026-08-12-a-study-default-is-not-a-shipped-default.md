# 2026-08-12 — a study default is not a shipped default (#3129)

**What happened.** The overview benchmark exists to measure *what a current user
gets*, so every behavioural knob was deliberately left unset. `CALIB_PATCH_STYLES`
was one of them — and its default is `max_patch,max_patch_pca_hac`, because the
**calibration study** wanted that contrast. `max_patch_pca_hac` lost the
Max-Patch study at the operating point (PR #2749) and production no longer
carries the tree it delegates to. So "leave the defaults alone" silently added a
retired arm to a baseline, and doubled the cost of every patch cell: all three
sizing cells finished `max_patch` and were still grinding through the HAC style
when they were cancelled.

Caught by the owner reading a status line, not by any check. Cost: one cancelled
array ~3 minutes in, plus ~50 minutes of sizing.

**Prevented?** *Advice only.* The general form — "is this default a product
default or this study's default?" — is not mechanically checkable without a
per-knob provenance table that does not exist. What *is* now pinned: the
benchmark launcher sets `CALIB_PATCH_STYLES=max_patch` explicitly and passes it
in `ENVX` rather than relying on `--export=ALL`, so the value cannot drift with
the submitting shell.

**Rule of thumb:** when a study's premise is "defaults", enumerate the defaults
and say where each one comes from. A knob whose default was chosen by another
study is a knob you are setting, not a knob you are leaving alone.
