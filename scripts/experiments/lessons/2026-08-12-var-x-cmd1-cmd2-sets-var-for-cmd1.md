# 2026-08-12 — #3121 `VAR=x cmd1 && cmd2` sets VAR for cmd1 only
**Cost:** ~15 min, but the exposure was much larger than the cost.

**What broke.** The pile scripts located their own checkout via `VTS_REPO`. In
`VTS_REPO=... python build.py --verify && python build.py --bands`, the shell
applies the assignment to the **first** command only, so the second ran without
it, skipped its `sys.path` insert, and resolved `import vtscore` through the
venv's editable install — pointing at the *main* checkout, **592 commits stale**
and missing embedders the pile uses. It surfaced as a confusing `ImportError`.

That was the lucky outcome. A stale-but-compatible tree would not have raised at
all; it would have embedded cells with different code, silently.

**Now prevented by** deriving the checkout from `__file__` instead of an env var,
and asserting at startup that `vtscore` actually resolved inside this checkout
(`assert_vtscore_is_this_checkout`). **Generalise it:** a script that needs a
particular tree should *verify* it got that tree, not *request* it.
