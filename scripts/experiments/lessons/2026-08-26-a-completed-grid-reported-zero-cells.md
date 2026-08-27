# 2026-08-26 — a completed grid reported "0 / 6480" cells (#3156)

**Study:** #3156 `vg_scale` overview. **Cost:** none this time, ~20 minutes of
"did the run fail?" — but only because the *analyzer* counted independently and
said 6480. Read alone, the status file said the run produced nothing.

`scale-3156-final` finished all 6480 cells cleanly. Its post-analysis wrote:

```
cells: 0 / 6480
zero-byte: 0
```

The count came from:

```sh
echo "cells: $(ls $d/task_*.csv 2>/dev/null | grep -vc __) / 6480"
```

This grid writes **four** files per cell — the main frame plus `__sweep`,
`__cutdiag`, `__cutincl` — so `$d/task_*.csv` expanded to 25,920 paths, exceeded
`ARG_MAX`, and `ls` died with *"Argument list too long"*. `2>/dev/null` swallowed
the message and `grep -vc` counted zero matching lines. **A shell failure
rendered as a plausible number**, in the field whose whole job is to say whether
the run happened.

Note which safety measure made it worse: `2>/dev/null` was there to suppress the
"no such file" noise of an empty directory, and it suppressed the one message
that explained the zero.

**The general shape:** a count derived from a *pipeline* reports the pipeline's
shape, not the filesystem's. `ls | grep -c` cannot distinguish "no files" from
"ls never ran". This is the same family as
[echo $? after a pipeline reads the pipeline](2026-08-12-echo-exit-after-a-pipeline-reads-the-pipeline.md).

**Prevented.** Both shell call sites now use `find`, which streams and never
builds an argv:

```sh
find "$d" -maxdepth 1 -name 'task_*.csv' ! -name '*__*' | wc -l
```

Fixed in `launch_horizon.sh`'s `status` and in the #3156 post-analysis script.
The python analyzers were never affected — `glob.glob` has no argv — which is
exactly why the discrepancy was survivable.

**Still only advice:** any *new* ad-hoc status script can reintroduce it. If you
are counting cells in shell, use `find`; the moment a grid grows a side frame,
the glob is four times bigger than you think.
