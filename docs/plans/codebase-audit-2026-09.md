# Codebase audit — September 2026 (structure & organization)

**Background.** A tech-debt audit (god modules, duplication, dead code, layering)
was run at `0de6fb63`. Every concrete finding became a GitHub issue (#3375–#3453)
and has since closed. Before removing any `vtscore` surface, follow CLAUDE.md's
"Dead code in `vtscore/` is a claim you cannot verify by grepping" rule — an
in-repo grep does not prove a public name is unused.

What remains is one tracked question and one design fork.

- [ ] #3452 — Find out who uses the autorun extractor/localizer surface before touching it (Sonnet). Kept as-is pending an answer from the external developers.

<!-- item-sep -->

- **What is `CoreConfig` for?** — `vtscore/config/core_config.py`

  Some 20 call sites call `CoreConfig.from_settings()` ad hoc, each invoking ~18 settings getters through the app shim, so the frozen-value-object abstraction buys nothing while costing a full settings snapshot per lookup. The design comment at the top of `vtscore/config/core_config.py` still says "Until those land this class is unused at runtime" — stale for a while now.

  *The fork:* either restore the original design (build one snapshot per operation and pass it down, which is a real plumbing change) or accept that the getters won and replace `CoreConfig` with direct calls. Both are defensible; picking one is a design call, not a cleanup. The stale comment should go either way.
