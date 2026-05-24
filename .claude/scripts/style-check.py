#!/usr/bin/env python3
"""Static SCSS audit for the VTSearch frontend.

Scans `frontend/src/**/*.scss` for violations of the rules in
`docs/style-guide.md`.  Designed to be invoked by the `/style-check`
skill: prints findings grouped by rule, with file:line and a one-line
explanation per hit.  Exits 0 even when violations are found — this is
a report tool, not a CI gate.

Run from the repo root or via the skill.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCSS_DIR = REPO_ROOT / "frontend" / "src"

# Files where raw values are legitimate (token source-of-truth).
EXCLUDE_FILES = {
    SCSS_DIR / "scss" / "_variables.scss",
}

# Files where shared utility classes are the source-of-truth (allowed
# to define `.info-text`, `.form-label`, etc.).
SHARED_UTILITY_FILES = {
    SCSS_DIR / "scss" / "_components.scss",
    SCSS_DIR / "scss" / "_picker-shared.scss",
    SCSS_DIR / "scss" / "_data-table.scss",
    SCSS_DIR / "scss" / "_layout.scss",
}


@dataclass
class Finding:
    rule: str
    description: str
    hits: list[tuple[Path, int, str]] = field(default_factory=list)


def scss_files() -> list[Path]:
    files = sorted(SCSS_DIR.rglob("*.scss"))
    return [f for f in files if f not in EXCLUDE_FILES]


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# §4.1-2 Hardcoded px/rem in spacing or font-size properties
# ---------------------------------------------------------------------------
SPACING_PROP = re.compile(
    r"\b(padding|padding-(?:top|right|bottom|left)|margin|margin-(?:top|right|bottom|left)|"
    r"gap|row-gap|column-gap|font-size)\s*:\s*([^;}\n]+)"
)
RAW_LENGTH = re.compile(r"(?<![\w\.\-])(\d+(?:\.\d+)?)(px|rem)\b")
# Values that are conventionally OK even though they're raw:
#   0 of any unit, 1px borders, hairline offsets in skeletons.
RAW_OK_VALUES = {"0px", "0rem", "1px"}


def check_raw_lengths(files: list[Path]) -> Finding:
    f = Finding(
        rule="§4.1-2 Hardcoded px/rem in spacing/font-size",
        description=(
            "Use --space-* / --font-* tokens. Raw values in padding | margin | "
            "gap | font-size bypass the token system."
        ),
    )
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            for m in SPACING_PROP.finditer(line):
                value = m.group(2)
                # Skip if the value is fully a token: `var(--space-md)` etc.
                if "var(--" in value and not RAW_LENGTH.search(
                    re.sub(r"var\([^)]+\)", "", value)
                ):
                    continue
                for length_match in RAW_LENGTH.finditer(value):
                    raw = length_match.group(0)
                    if raw in RAW_OK_VALUES:
                        continue
                    # Allow 1px borders inside `border:` declarations — we
                    # already filter by SPACING_PROP, so they wouldn't reach
                    # here, but skip any leftover.
                    f.hits.append((path, lineno, line.rstrip()))
                    break
    return f


# ---------------------------------------------------------------------------
# §4.3 Hex colors in component SCSS
# ---------------------------------------------------------------------------
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def check_hex_colors(files: list[Path]) -> Finding:
    f = Finding(
        rule="§4.3 Hex colors in component SCSS",
        description=(
            "Theme variables only. Component-local hex literals don't respond "
            "to theme changes — add a token to _variables.scss instead."
        ),
    )
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            # Skip URL-like sequences.
            sanitized = re.sub(r"url\([^)]+\)", "", line)
            for m in HEX_COLOR.finditer(sanitized):
                f.hits.append((path, lineno, line.rstrip()))
                break
    return f


# ---------------------------------------------------------------------------
# §4.6 font-weight: 700 / bold
# ---------------------------------------------------------------------------
BOLD_RE = re.compile(r"font-weight\s*:\s*(700|bold)\b")


def check_bold_weight(files: list[Path]) -> Finding:
    f = Finding(
        rule="§4.6 font-weight: 700 / bold",
        description="Use --weight-semibold (600). <strong> is fine; raw 700 is not.",
    )
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if BOLD_RE.search(line) and not line.strip().startswith("//"):
                f.hits.append((path, lineno, line.rstrip()))
    return f


# ---------------------------------------------------------------------------
# §4.7 Heading tags restyled in component SCSS
# ---------------------------------------------------------------------------
HEADING_RULE_RE = re.compile(r"^\s*h[1-6]\s*\{")


def check_heading_restyling(files: list[Path]) -> Finding:
    f = Finding(
        rule="§4.7 Heading tags visually restyled in component SCSS",
        description=(
            "Use the right tag; let the global rule style it. Margin-only "
            "scoping (`h3 { margin: 0 0 var(--space-md); }`) is fine — it "
            "positions the heading without changing its identity. This flags "
            "blocks that override `font-size` or `font-weight`, which is the "
            "actual 'make a smaller tag look like a bigger one' anti-pattern."
        ),
    )
    for path in files:
        if path in SHARED_UTILITY_FILES:
            continue
        text = path.read_text()
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            if HEADING_RULE_RE.match(lines[i]):
                # Walk the block until the opening `{` is matched, then
                # inspect the body for visual restyling.
                rule_line = i + 1
                start = i
                stripped = re.sub(r"//.*$", "", lines[i])
                opens = stripped.count("{")
                closes = stripped.count("}")
                depth = opens - closes
                j = i + 1
                while j < len(lines) and depth > 0:
                    stripped = re.sub(r"//.*$", "", lines[j])
                    depth += stripped.count("{") - stripped.count("}")
                    j += 1
                # `body` includes the opening line so single-line rules
                # (`h1 { margin: 0; }`) are inspected correctly.
                body = "\n".join(lines[start:j])
                if re.search(r"\bfont-size\b", body) or re.search(
                    r"\bfont-weight\b", body
                ):
                    f.hits.append((path, rule_line, lines[i].rstrip()))
                i = j
            else:
                i += 1
    return f


# ---------------------------------------------------------------------------
# §4.10 transition: all <custom-duration>
# ---------------------------------------------------------------------------
TRANSITION_ALL_RE = re.compile(r"transition\s*:\s*all\s+0?\.\d+s")


def check_transition_all(files: list[Path]) -> Finding:
    f = Finding(
        rule="§4.10 transition: all <custom-duration>",
        description=(
            "Use var(--transition-base) (or a specific property). `transition: "
            "all 0.2s` couples every animatable property to a one-off duration."
        ),
    )
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if TRANSITION_ALL_RE.search(line) and "var(--transition" not in line:
                f.hits.append((path, lineno, line.rstrip()))
    return f


# ---------------------------------------------------------------------------
# §4.13 `font: inherit` on form-input/form-select aliases
# ---------------------------------------------------------------------------
def check_font_inherit_trap(files: list[Path]) -> Finding:
    f = Finding(
        rule="§4.13 `font: inherit` may silently override .form-select",
        description=(
            "Angular view encapsulation gives component-scoped rules higher "
            "specificity than the global .form-input/.form-select. `font: "
            "inherit` in a component class then wins, dropping --font-md and "
            "rendering the page-root 1rem. Use explicit `font-family: inherit; "
            "font-size: var(--font-md);` or set the size that actually applies."
        ),
    )
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\bfont\s*:\s*inherit\b", line) and not line.strip().startswith("//"):
                f.hits.append((path, lineno, line.rstrip()))
    return f


# ---------------------------------------------------------------------------
# §4.14 flex-direction: column rule blocks without an explicit `gap`
# ---------------------------------------------------------------------------
def check_flex_column_no_gap(files: list[Path]) -> Finding:
    f = Finding(
        rule="§4.14 `flex-direction: column` block missing explicit `gap`",
        description=(
            "Stacked-column containers must own their inter-row spacing. "
            "Either set `gap: var(--space-*)` on the parent or rely on a "
            "child class (form-group, etc.) that has its own margins."
        ),
    )
    for path in files:
        text = path.read_text()
        # Walk through SCSS, tracking brace depth and per-block flags.
        # When a block closes, if it set flex-direction: column without a
        # `gap` declaration anywhere in its OWN body, flag the line where
        # the column directive appeared.
        depth = 0
        stack: list[dict] = [{"has_col": False, "col_line": 0, "has_gap": False}]
        line_no = 0
        for line in text.splitlines():
            line_no += 1
            stripped = re.sub(r"//.*$", "", line)
            for ch in stripped:
                if ch == "{":
                    depth += 1
                    stack.append({"has_col": False, "col_line": 0, "has_gap": False})
                elif ch == "}":
                    if stack:
                        frame = stack.pop()
                        if frame["has_col"] and not frame["has_gap"]:
                            # Only flag rules that look like display:flex containers.
                            f.hits.append(
                                (path, frame["col_line"], _read_line(text, frame["col_line"]))
                            )
                    depth = max(0, depth - 1)
            if "flex-direction" in stripped and "column" in stripped and stack:
                stack[-1]["has_col"] = True
                stack[-1]["col_line"] = line_no
            if re.search(r"\bgap\s*:", stripped) and stack:
                stack[-1]["has_gap"] = True
    return f


def _read_line(text: str, line_no: int) -> str:
    lines = text.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].rstrip()
    return ""


# ---------------------------------------------------------------------------
# §4.15 Shared utility classes redeclared locally
# ---------------------------------------------------------------------------
SHARED_CLASSES = [
    "info-text",
    "error-text",
    "success-text",
    "status-text",
    "form-label",
    "form-input",
    "form-select",
    "form-hint",
    "form-group",
    "btn",
    "modal-content",
    "modal-header",
    "modal-body",
    "modal-footer",
    "modal-close",
    "modal-backdrop",
    "back-btn",
]


def check_redeclared_utility(files: list[Path]) -> Finding:
    f = Finding(
        rule="§4.15 Shared utility classes redeclared in component SCSS",
        description=(
            "The classes in _components.scss are the source of truth. "
            "Redeclaring `.info-text { ... }` etc. locally drifts styling "
            "across modals and defeats the shared baseline. Extend with a "
            "child selector (`.my-panel .info-text { ... }`) if a scoped "
            "tweak is genuinely needed."
        ),
    )
    # End of the class name must be a non-class-character — not `-` or
    # `\w` — so `.btn-good` and `.form-group--section` (which are new
    # classes, not redeclarations of `.btn` or `.form-group`) don't match.
    pattern = re.compile(
        r"^\.(" + "|".join(re.escape(c) for c in SHARED_CLASSES) + r")(?![-\w])[^{]*\{"
    )
    for path in files:
        if path in SHARED_UTILITY_FILES:
            continue
        text = path.read_text()
        depth = 0
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = re.sub(r"//.*$", "", line)
            # Only flag declarations at top level (depth 0). Nested
            # `.parent { .form-input { ... } }` is a scoped override,
            # not a redeclaration.
            if depth == 0 and pattern.match(line):
                f.hits.append((path, lineno, line.rstrip()))
            depth += stripped.count("{") - stripped.count("}")
            depth = max(depth, 0)
    return f


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(findings: list[Finding]) -> int:
    total = 0
    bold = "\033[1m"
    reset = "\033[0m"
    dim = "\033[2m"
    yellow = "\033[33m"
    for finding in findings:
        count = len(finding.hits)
        total += count
        header = f"{bold}── {finding.rule} ──{reset}  {yellow}{count} hit{'s' if count != 1 else ''}{reset}"
        print()
        print(header)
        print(f"{dim}{finding.description}{reset}")
        if not finding.hits:
            print("  (clean)")
            continue
        for path, lineno, snippet in finding.hits:
            print(f"  {rel(path)}:{lineno}: {snippet.strip()}")
    print()
    print(f"{bold}Total findings: {total}{reset}")
    print(
        "Findings are informational — review and curate before fixing; "
        "some hits (e.g. a column-header button intentionally inheriting "
        "its parent's small uppercase font) are legitimate."
    )
    return 0


def main() -> int:
    files = scss_files()
    findings = [
        check_raw_lengths(files),
        check_hex_colors(files),
        check_bold_weight(files),
        check_heading_restyling(files),
        check_transition_all(files),
        check_font_inherit_trap(files),
        check_flex_column_no_gap(files),
        check_redeclared_utility(files),
    ]
    return print_report(findings)


if __name__ == "__main__":
    sys.exit(main())
