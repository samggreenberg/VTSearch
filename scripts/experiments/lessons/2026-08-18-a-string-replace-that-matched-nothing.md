# 2026-08-18 — a string-replace edit that matched nothing, an hour later (#3160)

**Cost:** ~15 min, one crashed backfill job (its real work had already
completed), one confusing `IndexError`.

**What broke.** A column was added to `build_pile.py`'s provenance table — both
the row tuple and the format string — in one commit. `ruff format` then
reformatted the single-line tuple across seven lines. A later edit, applied with
`str.replace()` on the *old* single-line text, silently matched nothing: the
format string kept its seven slots and the row kept six fields. Nothing failed
at edit time, nothing failed at commit time, and both pre-commit hooks passed —
the code is syntactically fine. It surfaced an hour later as

```
IndexError: Replacement index 6 out of range for positional args tuple
```

inside a SLURM job, *after* it had written all 21 sidecars but before it printed
the table that was the point of running it.

**The mechanism worth remembering:** a formatter is a second author. Any edit
that matches on source text can be invalidated by a reformat between commits,
and `str.replace()` reports that by doing nothing at all. The same script in the
same session *did* use `assert old in s` for two other edits, which is precisely
why those did not rot.

**Prevented?** *Advice only, but cheap advice.* When editing source by text
substitution, assert the anchor matched (`assert old in s`) — a failed assert is
a five-second fix, a silent no-op is an hour. Where a row and its format string
must agree, prefer a structure that cannot disagree (a list of `(header, value)`
pairs) over two parallel literals.
