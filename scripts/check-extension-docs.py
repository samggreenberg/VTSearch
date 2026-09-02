"""Check that the extension guides describe members that actually exist.

The two extension doc sets — ``docs/EXTENDING-*.md`` (app tier) and
``vtscore/docs/extending/`` (library tier) — both spell out the contract for
the same plugin ABCs, in their own words. Independent prose over one shared
class is exactly the thing that rots: each set can drift from the code, and
from the other, without anything failing. Issue #3442 found three live
instances at once, each half of a contradicting pair:

* ``embedders.md`` told authors to override ``embed_text``; the real hook is
  ``_embed_text_impl`` (the public method L2-normalizes, so an override there
  ships un-normalized query vectors).
* ``EXTENDING-media.md`` documented ``_patch_forward(image)``; the real hook
  is ``_patch_forward_impl(media)``, so a plugin following it defined a method
  nothing ever calls.
* ``vtscore/docs/extending/README.md`` still named the ``LabelsetExporter``
  base class after the rename to ``ResultsExporter``, contradicting its own
  per-family page.

All three are *member-name* errors, and a member name is mechanically
checkable. This script does that, in the spirit of
``scripts/check-vtscore-docs.py``: two cheap invariants, stdlib only, no
imports of the package under test (so it runs before the venv is warm and
can't be broken by a heavy optional dependency).

1. **Every member named in a contract table exists on the class.** Members are
   resolved through the in-repo base chain, so inherited names count.
2. **Every contract section is registered in :data:`SECTIONS`.** Without this,
   a new "abstract interface reference" heading would ship unchecked — the
   coverage hole that makes invariant 1 look healthier than it is.

What this deliberately does *not* check: that the prose is accurate, that
signatures match, or that every member is documented. Those need a reader.
This checks the mechanical half — the half that rots silently.

Run from the repo root:
    python scripts/check-extension-docs.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Packages scanned for class definitions. A documented member counts as real
#: if it is defined anywhere in this tree, on the class or on one of its
#: in-repo bases.
PACKAGES = ("vtscore", "vtsearch")

#: Which classes each contract section describes, as
#: ``(doc path, exact heading text, classes)``. A section is the run of lines
#: from its heading to the next heading of the same or shallower level; every
#: markdown table inside it is checked. Several classes may be listed when one
#: section tabulates an ABC and its subtypes together — a documented member
#: need only exist on one of them.
SECTIONS: list[tuple[str, str, tuple[str, ...]]] = [
    # --- app tier -------------------------------------------------------
    ("docs/EXTENDING-media.md", "MediaType abstract interface reference", ("MediaType",)),
    ("docs/EXTENDING-media.md", "MediaEmbedder abstract interface reference", ("MediaEmbedder",)),
    ("docs/EXTENDING-media.md", "Embedder capability flags", ("MediaEmbedder",)),
    ("docs/EXTENDING-media.md", "MediaClipper abstract interface reference", ("MediaClipper",)),
    ("docs/EXTENDING-media.md", "MediaConverter abstract interface reference", ("MediaConverter",)),
    ("docs/EXTENDING-media.md", "MediaSource abstract interface reference", ("MediaSource",)),
    (
        "docs/EXTENDING-processors.md",
        "Processor abstract interface reference",
        ("Processor", "Detector", "Localizer", "Extractor"),
    ),
    # Not an ABC contract: this section spells out how ``dirty_keys`` on
    # ``UserSyncState`` arbitrates between a local write and an upstream pull.
    # It is registered because invariant 2 (rightly) flags any "The ... contract"
    # heading, and the names it does cite are members worth checking.
    ("docs/EXTENDING-plugins.md", "The dirty-key contract", ("UserSyncState",)),
    # --- library tier ---------------------------------------------------
    ("vtscore/docs/extending/clippers.md", "The contract", ("MediaClipper",)),
    ("vtscore/docs/extending/converters.md", "The contract", ("MediaConverter",)),
    ("vtscore/docs/extending/dataset-importers.md", "The minimum contract", ("DatasetImporter",)),
    ("vtscore/docs/extending/embedders.md", "The contract", ("MediaEmbedder",)),
    ("vtscore/docs/extending/embedders.md", "Capability flags", ("MediaEmbedder",)),
    ("vtscore/docs/extending/label-importers.md", "The contract", ("LabelImporter",)),
    ("vtscore/docs/extending/labelset-sources.md", "The contract", ("LabelsetSource",)),
    ("vtscore/docs/extending/media-types.md", "The `MediaType` contract", ("MediaType",)),
    ("vtscore/docs/extending/results-exporters.md", "The contract", ("ResultsExporter",)),
]

#: Headings that look like a contract section and must therefore appear in
#: :data:`SECTIONS`. Checked across every file either doc set owns.
_CONTRACT_HEADING = re.compile(r"(abstract interface reference|^the .*contract$|^capability flags$)", re.I)

#: Doc trees whose contract headings invariant 2 polices.
DOC_GLOBS = ("docs/EXTENDING-*.md", "vtscore/docs/extending/*.md")

#: Names that a contract section may mention without them being members of
#: that section's classes: parameter-dict keys, payload fields, and members
#: the prose explicitly marks as living off the ABC. Keyed by section so one
#: family's vocabulary can never excuse a typo in another's.
IGNORE: dict[tuple[str, str], frozenset[str]] = {
    # The "Not on the ABC, but required by the app" table: `from_config` is an
    # app-side factory convention implemented by concrete processors, and the
    # section says so in as many words.
    ("docs/EXTENDING-processors.md", "Processor abstract interface reference"): frozenset({"from_config"}),
    # Parameter-dict keys tabulated alongside the `parameters` member.
    ("docs/EXTENDING-media.md", "MediaClipper abstract interface reference"): frozenset(
        {"key", "label", "type", "default", "min", "max", "step"}
    ),
    ("vtscore/docs/extending/clippers.md", "The contract"): frozenset(
        {"key", "label", "type", "default", "min", "max", "step"}
    ),
    ("vtscore/docs/extending/converters.md", "The contract"): frozenset({"key", "label", "type", "default", "options"}),
}

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
#: Every backticked code span on a line.
_SPAN = re.compile(r"`([^`]+)`")
#: The leading identifier of a member reference: ``foo``, ``foo()``,
#: ``foo(bar)``, ``foo: int``, ``foo (property)``.
_IDENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")
#: A span that reads as a member reference wherever it sits in a table row,
#: not just in the first cell: a call (``_patch_forward(image) -> Out``) or a
#: private name (``_embed_text_impl``). Public bare words are excluded here
#: because ``str`` and ``dict`` look exactly like ``name`` — those are caught
#: in the first cell, where a bare word can only be a member.
_INLINE_MEMBER = re.compile(r"^(_[a-z][A-Za-z0-9_]*|[a-z][A-Za-z0-9_]*\()")


def _index_module_level() -> set[str]:
    """Return every module-level function and constant name in the tree.

    A contract section may legitimately point at a free function
    (``set_progress_callback()``, ``embedders_for_type()``) rather than a
    method. Those are real, greppable symbols, so they satisfy the invariant —
    what the gate is proving is that a documented name still exists somewhere,
    not that it is a method.
    """
    names: set[str] = set()
    for pkg in PACKAGES:
        for path in sorted((ROOT / pkg).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for stmt in tree.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(stmt.name)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    names.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    names.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    return names


def _index_classes() -> dict[str, dict[str, set[str]]]:
    """Return ``{class name: {"members": {...}, "bases": {...}}}`` for the tree.

    Classes are keyed by bare name and merged across files. Two same-named
    classes therefore pool their members, which can only *hide* a bad doc
    reference, never invent one — the conservative direction for a gate.
    """
    classes: dict[str, dict[str, set[str]]] = {}
    for pkg in PACKAGES:
        for path in sorted((ROOT / pkg).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    entry = classes.setdefault(node.name, {"members": set(), "bases": set()})
                    entry["members"] |= _members_of(node)
                    entry["bases"] |= _base_names(node)
    return classes


def _base_names(node: ast.ClassDef) -> set[str]:
    """Return the bare names of *node*'s bases, unwrapping generic subscripts."""
    names: set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Subscript):  # SyncSource[list[dict], LabelSet]
            base = base.value
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _members_of(node: ast.ClassDef) -> set[str]:
    """Return every name *node* defines: methods, class attrs, and ``self.x``."""
    members: set[str] = set()
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.add(stmt.name)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            members.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            members.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
    # Attributes bound in __init__ (or any method) are members too.
    for sub in ast.walk(node):
        targets: list[ast.expr] = []
        if isinstance(sub, ast.Assign):
            targets = list(sub.targets)
        elif isinstance(sub, ast.AnnAssign):
            targets = [sub.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                members.add(target.attr)
    return members


def _resolve(name: str, classes: dict[str, dict[str, set[str]]], seen: set[str] | None = None) -> set[str]:
    """Return every member of *name*, following its in-repo base chain."""
    seen = seen if seen is not None else set()
    if name in seen or name not in classes:
        return set()
    seen.add(name)
    members = set(classes[name]["members"])
    for base in sorted(classes[name]["bases"]):
        members |= _resolve(base, classes, seen)
    return members


def _section_lines(lines: list[str], heading: str) -> tuple[int, list[str]] | None:
    """Return the 1-based start line and body of the section titled *heading*."""
    for i, line in enumerate(lines):
        match = _HEADING.match(line)
        if not match or match.group(2) != heading:
            continue
        depth = len(match.group(1))
        body: list[str] = []
        for follow in lines[i + 1 :]:
            nxt = _HEADING.match(follow)
            if nxt and len(nxt.group(1)) <= depth:
                break
            body.append(follow)
        return i + 1, body
    return None


def _documented_members(body: list[str]) -> list[tuple[str, str]]:
    """Return ``(identifier, raw span)`` for every member reference in *body*.

    Only table rows are scanned, which keeps the sweep bounded to the places a
    contract is actually tabulated. Within a row, the first cell's span counts
    whatever it looks like; later cells count only when the span is a call or a
    private name, so a type in a signature column isn't mistaken for a member.
    """
    found: list[tuple[str, str]] = []
    for line in body:
        if not line.lstrip().startswith("|"):
            continue
        for index, raw in enumerate(_SPAN.findall(line)):
            first_cell = index == 0 and line.lstrip().startswith("| `")
            if not first_cell and not _INLINE_MEMBER.match(raw):
                continue
            ident = _IDENT.match(raw)
            if ident:
                found.append((ident.group(1), raw))
    return found


def _check_sections(classes: dict[str, dict[str, set[str]]], module_level: set[str]) -> list[str]:
    """Invariants 1 and 3, which both walk the same sections.

    1. Every member named in a contract table exists on one of the section's
       classes (or is itself a class — docs legitimately tabulate types).
    3. A documented wrapper is documented alongside its hook: if a section
       names a public ``X`` for which the class defines ``_X_impl``, it must
       name ``_X_impl`` too, or it is presenting the wrapper as the override
       point.
    """
    problems: list[str] = []
    for doc, heading, class_names in SECTIONS:
        path = ROOT / doc
        if not path.exists():
            problems.append(f"{doc}: listed in SECTIONS but the file is missing")
            continue
        section = _section_lines(path.read_text(encoding="utf-8").splitlines(), heading)
        if section is None:
            problems.append(f"{doc}: no section titled {heading!r} (SECTIONS is stale)")
            continue
        members: set[str] = set()
        missing = [name for name in class_names if not _resolve(name, classes)]
        if missing:
            joined = ", ".join(repr(name) for name in missing)
            problems.append(f"{doc}: class(es) {joined} not found under {'/, '.join(PACKAGES)}/")
            continue
        for name in class_names:
            members |= _resolve(name, classes)

        start, body = section
        ignore = IGNORE.get((doc, heading), frozenset())
        documented = _documented_members(body)
        names = {ident for ident, _ in documented}

        for ident, raw in documented:
            if ident in members or ident in ignore or ident in classes or ident in module_level:
                continue
            problems.append(
                f"{doc}:{start} § {heading}: `{raw}` is not a member of "
                f"{'/'.join(class_names)} (nor of its bases, nor a class or module-level "
                f"symbol). Fix the "
                f"doc, or add it to this section's IGNORE entry if it names a dict "
                f"key rather than a member."
            )

        for ident in sorted(names):
            hook = f"_{ident}_impl"
            if ident.startswith("_") or hook not in members or hook in names:
                continue
            problems.append(
                f"{doc}:{start} § {heading}: `{ident}` is a wrapper — the class "
                f"defines `{hook}`, which is the real override point — but the "
                f"section never names `{hook}`. Readers will override `{ident}` and "
                f"lose whatever the wrapper does (locking, normalization, progress). "
                f"Document the hook here, or stop naming the wrapper."
            )
    return problems


def _check_coverage() -> list[str]:
    """Invariant 2: every contract-shaped heading is registered in SECTIONS."""
    registered = {(doc, heading) for doc, heading, _ in SECTIONS}
    problems: list[str] = []
    for glob in DOC_GLOBS:
        for path in sorted(ROOT.glob(glob)):
            doc = path.relative_to(ROOT).as_posix()
            for line in path.read_text(encoding="utf-8").splitlines():
                match = _HEADING.match(line)
                if not match:
                    continue
                heading = match.group(2)
                if _CONTRACT_HEADING.search(heading) and (doc, heading) not in registered:
                    problems.append(
                        f"{doc}: section {heading!r} looks like a contract table but is "
                        f"not in SECTIONS — add it (with the class it documents) so its "
                        f"member names are checked."
                    )
    return problems


def main() -> int:
    problems = _check_sections(_index_classes(), _index_module_level()) + _check_coverage()
    if problems:
        print("Extension-doc check failed:\n")
        for problem in problems:
            print(f"  - {problem}")
        print(f"\n{len(problems)} problem(s). See the module docstring for what this checks.")
        return 1
    print(f"Extension docs OK ({len(SECTIONS)} contract sections checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
