# 2026-08-25 — preflight failed for a reason that had nothing to do with the run (#3115)

**Study:** #3115 / #3116 fold combine-rule run. **Cost:** minutes, and no wrong
science — but it is the *inverse* of the incident that matters, so it is worth
the file.

Running `preflight.sh` from a non-interactive `ssh grid` — which is how a
launcher's `arms` mode reaches it — produced this:

```
FAIL  could not check the region-voting premise: Traceback (most recent call last):
  ... TypeError: unsupported operand type(s) for |: 'type' and 'types.GenericAlias'
FAIL  could not check patch styles: Traceback (most recent call last):
  ... TypeError: unsupported operand type(s) for |: 'types.GenericAlias' and 'NoneType'
FAIL  could not compare this run's knobs against production: Traceback ...
```

Three failures, ~30 lines of stack, one cause: **no venv**, so `python` was the
system interpreter, too old to evaluate `X | None` at import time. Sourcing
`gridenv.sh` first turned all three green, region-voting premise included
(`patch_grid=4193/4193`).

**Two separate things were wrong, and only one of them is mine.**

The launcher called preflight from a shell that had never activated the venv.
That is now fixed in `launch_folds_3115.sh`, which sources `gridenv.sh` before
the gate and dies with a named error if it cannot.

But preflight *itself* handled it badly. Check 5 already resolves `import
vtscore` and already says the useful line — *"is the venv active (source
gridenv.sh)?"* — and then checks 6, 7 and 12 each re-derive the identical cause
and print it as a raw traceback. The one actionable line scrolls off the top;
what a reader sees is three unrelated-looking type errors deep inside `vtscore`,
which invites debugging the tree instead of the shell.

**The general form.** A gate that reports *n* failures for one cause is worse
than a gate that reports one, because the reader's first move is to ask what the
failures have in common — and here they look like they have nothing in common.
When one check establishes that a whole class of later checks cannot run, the
later ones should say **"not checked, because of the above"**, not fail
independently. "Not checked" is also the honest word: those premises are still
unverified, and a `SKIP` that read as `ok` would be the far more dangerous bug —
which is exactly what #2905 hit, where preflight reported `ok` without having
looked.

**Status: prevented.** `preflight.sh` now sets `PY_USABLE=0` when check 5 cannot
resolve the import, and checks 6, 7 and 12 fail with a one-line
*"NOT checked: python cannot import the tree (see above)"* instead of a
traceback. They still **fail** — nothing is downgraded, and a run cannot launch
on unverified premises.


---

## Addendum, same day — and the sibling lesson did not replicate

While sizing this study's cells I applied #2883's thread-pinning fix
(`OMP_NUM_THREADS=1` and siblings) on the reasoning that the anchored arm fits 52
`GaussianMixture` mixtures per step, so the per-fit BLAS pool should dominate.
Then I measured it, paired, on the same cell:

| cell 1 — `visual_genome_m x siglip`, whole_image, 150 steps | elapsed | MaxRSS |
|---|---:|---:|
| unpinned | 20m33s | 594 MB |
| pinned | 20m31s | 511 MB |

**Two seconds apart.** #2883's 36x was a *login-node* effect — a shared box,
already oversubscribed — and a cpu-partition task with `--cpus-per-task=1` does
not reproduce it. The pins stayed, but on the other half of that lesson (120
concurrent cells each spawning a node-sized pool is antisocial regardless), and
the launcher's comment now carries both numbers instead of the prediction.

Worth recording because of *how* nearly it went the other way: mid-run I reported
that pinning "looks like it helps", comparing a job **8:43 into its run** against
a **completed 20:33** one. That is not a comparison at all, and it would have
written a fictional speedup into a launcher comment where the next reader would
have believed it. A partial timing is not a timing — the same shape as the
lesson it came from, one level up.
