---
name: style-check
description: Run a systematic frontend style audit. Use when the user asks for a "style check", "style sweep", or wants to find SCSS violations of the rules in docs/style-guide.md across the codebase. Static-only - does not render the app.
---

# Frontend style check

Static audit of `frontend/src/**/*.scss` against the rules in
`docs/style-guide.md`. Catches the categories of mistakes that produce
the kind of "spacing looks off / fonts inconsistent" bugs that motivated
this skill in the first place.

## How to run it

```
python3 .claude/scripts/style-check.py
```

The script prints findings grouped by style-guide section, with
`file:line: snippet` for each hit, plus a one-line "why" per rule.
It exits 0 even when findings exist - this is a **report tool, not a
CI gate**. Many hits are legitimate (see "Curating findings" below);
your job is to review them and decide which to fix.

## What it checks

| Rule | What | How |
|---|---|---|
| Broken `var()` | `var(--x)` with no fallback whose `--x` is defined nowhere (SCSS tree or template/TS style binding) — an invalid/empty render the Angular build won't catch | Paren-aware `var()` parser vs. the set of all defined custom properties |
| Deleted alias | Use of a design-token alias removed in the consolidation pass (`--border-color`, `--bg-secondary`, `--bg-primary`, `--accent-color`, `--color-accent`, `--error`), even behind a `var()` fallback | Alias→canonical map |
| §4.1-2 | Hardcoded `px`/`rem` in `padding`/`margin`/`gap`/`font-size` | Regex on the property list |
| §4.3   | Hex color literals in component SCSS | `#xxxxxx` outside `_variables.scss` |
| §4.6   | `font-weight: 700` / `bold` | Regex |
| §4.7   | Heading tags (`h1`–`h6`) restyled in component SCSS | Top-level `hN {` rule blocks |
| §4.10  | `transition: all <custom-duration>` | Regex, ignores tokenised durations |
| §1.8   | Raw `z-index` integers that bypass the `--z-*` scale | Regex, skips lines already resolving through a `var(--…)` |
| §1.10  | Raw `opacity: 0.7` where `var(--opacity-dim)` is the token | Regex on the canonical dim value only |
| Bespoke accent tint | Hand-rolled `color-mix(in srgb, var(--accent) N%, transparent)` instead of `--accent-highlight-bg` | Regex on the bare `var(--accent)` form, so the fallback-carrying decorative tint isn't flagged |
| §4.13  | `font: inherit` inside component SCSS (specificity trap) | Regex |
| §4.14  | `flex-direction: column` blocks without an explicit `gap` | Brace-depth aware SCSS parser |
| §4.15  | Shared utility classes redeclared locally | Top-level `.{class} {` outside shared files |

## Curating findings

The scanner is intentionally inclusive - it will flag legitimate
patterns alongside real bugs. Apply judgement before fixing:

- **§4.13 `font: inherit`** is sometimes intentional - e.g. a button
  that is itself nested inside an explicitly-styled small-font header
  (`folder-browser.component.scss:93` inherits its parent's
  `--font-2xs` uppercase header style on purpose). Check the parent
  chain before flagging.
- **§4.14 `flex-direction: column` without `gap`** is fine when each
  child element carries its own `margin` or `padding` - the rule
  allows either pattern. Top-level layout panels (`.left-panel`,
  `.center-panel`, `.app-shell`, etc.) usually fall in this bucket.
  Flag a hit only when stacked siblings look like form rows / labelled
  sections (`.form-group`, `<label>`, `.section-title`, `<p>`).
- **§4.15 redeclared utility class** is almost always real drift -
  the redeclared `.info-text`/`.error-text` blocks across modals are
  near-identical copies. Worth consolidating, but a bulk rewrite
  touches many components; consider doing it as its own PR.
- **§1.10 raw `opacity: 0.7`** has legitimate hits: animation keyframes
  that fade `0`↔`1`, hover-brighten rest states, and the two-tier
  done/future progression dims all carry their own values. Flag it when
  the element is decorative or secondary *at rest*.
- **§1.8 raw `z-index`** and the **bespoke accent tint** are, like the
  two token-resolution checks, close to objective - both have a single
  canonical token (`--z-*`, `--accent-highlight-bg`) and a hit means the
  component reinvented it. Adding a new layer means adding a `--z-*`
  token to `_variables.scss`, not a bare integer.
- **§4.1-2 raw `px`/`rem`** is sometimes a deliberate off-scale value
  with an inline comment justifying it (e.g.
  `padding: var(--space-xs) 0.625rem; // 10px - off-scale horizontal
  kept exact`). Comments documenting the exception are acceptable;
  unexplained raw values are not.

When in doubt, read the file context (a few lines above and below the
hit) before deciding it's a violation.

## How to report

When invoked by the user, run the script, then:

1. **Summarise total findings** (e.g. "83 hits across 10 rules").
2. **Highlight the clear bugs first** - the two token-resolution checks
   (**Broken `var()`** and **Deleted alias**) are objective: they have no
   legitimate hits, so treat every one as a bug to fix. After those,
   anything in §4.13 or §4.15 is usually genuine; §4.6 (`font-weight:
   700`) and §4.10 (`transition: all`) are almost always real.
3. **Group §4.14 hits by likely category** (real form/section
   containers vs. top-level layout panels) before listing.
4. **Offer to fix categories one at a time** rather than dumping a
   single mega-diff. The user can pick which categories matter for
   this sweep.

Keep the response scannable - file paths, line numbers, a one-line
"why". Don't paste the script's raw output verbatim if it's hundreds
of lines; curate first.

## Extending it

When a new style-guide rule lands (or a new specificity trap is
discovered the way `font: inherit` was), add a check function in
`.claude/scripts/style-check.py` and a row to the table above. Keep
each check small and focused - a check that flags multiple
unrelated patterns becomes hard to curate.

The scanner is static-only. A visual sweep (render key modals,
screenshot, look for layout drift) is a useful follow-up but lives
outside this skill.
