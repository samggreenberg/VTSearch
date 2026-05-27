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
| §4.1-2 | Hardcoded `px`/`rem` in `padding`/`margin`/`gap`/`font-size` | Regex on the property list |
| §4.3   | Hex color literals in component SCSS | `#xxxxxx` outside `_variables.scss` |
| §4.6   | `font-weight: 700` / `bold` | Regex |
| §4.7   | Heading tags (`h1`–`h6`) restyled in component SCSS | Top-level `hN {` rule blocks |
| §4.10  | `transition: all <custom-duration>` | Regex, ignores tokenised durations |
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
- **§4.1-2 raw `px`/`rem`** is sometimes a deliberate off-scale value
  with an inline comment justifying it (e.g.
  `padding: var(--space-xs) 0.625rem; // 10px - off-scale horizontal
  kept exact`). Comments documenting the exception are acceptable;
  unexplained raw values are not.

When in doubt, read the file context (a few lines above and below the
hit) before deciding it's a violation.

## How to report

When invoked by the user, run the script, then:

1. **Summarise total findings** (e.g. "83 hits across 8 rules").
2. **Highlight the clear bugs first** - anything in §4.13 or §4.15 is
   usually genuine; §4.6 (`font-weight: 700`) and §4.10 (`transition:
   all`) are almost always real.
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
