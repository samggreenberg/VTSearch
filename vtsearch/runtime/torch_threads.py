"""How many native-math threads the *server* should use.

Kept in its own module, importing nothing but ``os``, because ``app.py`` has to
decide this **before** anything imports torch/numpy/scipy -- OpenMP and MKL read
their env vars during that import and never look again.

The library default is 1 thread (``vtscore.config.TORCH_THREADS``), which is
right for batch work and tests: every extra thread allocates its own scratch
buffers, and a fat node's core count has OOM'd jobs here before. The
*interactive server* is the other case -- a person is waiting on every vote --
so it defaults to the allocation instead.

The failure this exists to prevent (2026-09-20): a server restarted by hand,
without the ``VTSEARCH_TORCH_THREADS`` the SLURM launcher sets, ran torch
single-threaded on an 8-CPU allocation for over an hour. Nothing in the log said
so, and the only reason anyone noticed was an unrelated investigation.

Honesty about the size of it: the slow labelling session that prompted this was
measured to be image *transfer*, not threads -- the sheets were too heavy -- so
the cost of running single-threaded here is **not** quantified. What is certain
is that the launcher sets it deliberately, with the comment "the default is 1
thread, which is painfully slow for SigLIP etc.", and that losing it silently
is worse than losing it loudly.
"""

from __future__ import annotations

import os

#: Env var an operator can set to override the default either way.
ENV_VAR = "VTSEARCH_TORCH_THREADS"


def allocated_cpus() -> int:
    """CPUs this process may actually run on.

    ``os.sched_getaffinity`` is the cgroup/affinity mask the scheduler imposed,
    which is the allocation. ``os.cpu_count()`` is the *machine*, and on these
    nodes reports 40 where the job holds 8 -- sizing from it oversubscribes.
    Mirrors ``vtscore.config.runtime.allocated_cpus``, duplicated here only
    because importing ``vtscore`` this early would import torch with it.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            return max(1, len(getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def resolve(environ: "os._Environ[str] | dict[str, str] | None" = None) -> int:
    """Threads for the server: the env var if set and usable, else the allocation.

    A blank or unparseable value falls back rather than raising: a server that
    will not start because a stray export said ``VTSEARCH_TORCH_THREADS=``
    helps nobody, and the fallback is the value that was wanted anyway.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_VAR) or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return allocated_cpus()
