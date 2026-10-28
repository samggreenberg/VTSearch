# 2026-10-14: 24 GB, the torch-2.6 venv backups from #4838 (owner-approved)

#4838 moved both GRID venvs from torch 2.6.0+cu124 to 2.14.1+cu126 by a rename swap, so the old venvs were left
beside them. Both are now deleted:

- `/expscratch/sgreenberg/vtsearch-venv.t26` (11 GB), the app venv until 2026-10-13 22:52.
- `/exp/sgreenberg/projects/VTSearch/.venv.t26` (13 GB), the experiment venv until 20:20. Deleting it
  took `/exp/sgreenberg` from 82% to 58% of its 50 GB.

Why it was safe:
- #4838 is `solved` and PR #4847 is merged.
- After the swap, the new venvs passed the GRID suite twice and the post-swap checks. A 48-cell SotA A/B gave identical
  picks and F-beta under both torches.
- No tool names a `.t26` path; `VTS_VENV` and gridenv.sh's default name the live venvs.
- No running job used them. The running #4384 tasks were checked through `/proc/<pid>/environ`: they had the live venv.

How to rebuild one (about 20 min): make a venv with the `python/3.12.3` module and
`pip install --no-deps -r` the saved freeze list. Torch comes from the cu124 index and cuML from pypi.nvidia.com.
The lists, taken just before the swap copies, are `/expscratch/sgreenberg/venv-4838/torch26-freeze-{app,exp}.txt`.
The editable `vtsearch` line points at the deploy clone (app) or the `/exp` checkout (exp).
