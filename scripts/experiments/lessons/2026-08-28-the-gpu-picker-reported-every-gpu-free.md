# 2026-08-28 — the GPU picker reported every GPU on the cluster as free (#3299)

**What broke.** `scripts/slurm/pick_gpu.py` measured usage from the `GresUsed=`
field of `scontrol show node --oneliner`. This cluster (Slurm 23.11.6) writes
that field on **zero** nodes — `scontrol show node --oneliner | grep -c GresUsed=`
returns `0`, and the multi-line form does not carry it either. The reader
flattened the missing field to an empty string, so `used = {}` and
`free = total` on every node, on every type, always.

The visible symptom was a picker that looked like it was working:

```
pick_gpu:   a100      23 free /  23 total
pick_gpu:   l40s      14 free /  14 total
pick_gpu:   v100     110 free / 110 total
pick_gpu: a100 -- 23/23 free -- fastest type with 3 free to start on now
```

Every count is the node total. Read off `AllocTRES=` instead, the same moment
was `a100 0/23`, `l40s 2/14`, `v100 109/110`. The three `vg_box_* × clip`
build jobs went to A100 and came back `Reason=Priority`,
`StartTime=2026-08-29T12:33` — **a 24-hour wait, with 109 V100s idle**.

**Cost.** ~15 minutes, all of it noticing. Cancelled and resubmitted with
`VTS_GPU=v100`; all three jobs started within 20 seconds and the six cells
built in ~11 minutes. Had nobody looked at `squeue`, it would have been a day.
Structurally the cost is larger and older: because `free == total` everywhere,
rule 2 of the selection order ("fastest type with `--need` free") always
matched the **first** candidate, so since 2026-08-17 the picker has returned
`a100` unconditionally. It was a hardcoded `a100` wearing a query.

**The general form.** The
[predecessor lesson](2026-08-17-a-hardcoded-gpu-type-is-a-pin-that.md) replaced
a hardcoded `v100` with "ask the scheduler", and the replacement did not ask —
it read a field that does not exist here and could not tell that apart from a
field that says zero. **A missing measurement must not be spellable as a
measurement of zero.** Both are falsy, and the failure is silent in the
direction that looks like good news: everything free, first choice wins, log
line reads exactly like success.

It survived a full test suite because **every fixture wrote a `GresUsed`
field**. Twelve parser tests, all green, all describing a cluster the code
never meets. A fixture is a claim about production, and this one was never
checked against a real record.

**Prevented.** `_node_gpus_used` now prefers `GresUsed` where a cluster writes
it, falls back to `AllocTRES`'s `gres/gpu:<type>=<n>`, and returns `None` when
**neither** field is present — a node whose usage cannot be read now contributes
nothing at all, exactly like a drained one, so an unreadable cluster degrades to
the fallback type instead of confidently claiming everything is free. The new
tests in `tests_lib/core/test_pick_gpu.py` are built from records copied
verbatim off `rack5n06` and `rack10n01` rather than composed, so the fixture
cannot drift away from the machine again.

**Still advice.** `VTS_GPU=v100` in `~/.bashrc` short-circuits the picker
entirely (`pick_gpu: v100 (VTS_GPU is set; not querying the scheduler)`) and
has silently pinned launches before — see
[`grid-gpu-node-gotchas`](../GRID-PLAYBOOK.md). Today it would have been the
*right* answer by accident, which is not a reason to keep it.
