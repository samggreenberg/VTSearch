#!/usr/bin/env python3
"""Drift gate: the eval framework's default arm vs. the app it is meant to model.

`vtscore.eval` exists to measure *deviations* from the shipped algorithm.  That
only means something if the framework's **default arm** is the shipped
algorithm.  When it isn't, every experiment run against it is measuring a
detector nobody uses, and the damage is silent and retroactive - the numbers
still look fine.

Most of the harness is safe by construction because it *delegates*: the
`max_patch` style calls `pool_box_from_media` / `bad_negative_vecs` /
`media_score_rows` rather than re-deriving them, so it cannot drift.  This gate
covers the parts that can't delegate:

* **ported** - app logic re-implemented in the harness, because the original is
  unreachable (it lives in TypeScript) or unusable (it is wrapped in
  interactive, lock-guarded, single-detector caches).  A copy goes stale the
  moment the original moves.
* **default** - places where the harness resolves "no explicit arm" to whatever
  the app currently defaults to.  When the app's default changes, the harness
  keeps handing out the old one under the name "default".

Each mirror pins a digest of the app-side source.  Changing the app trips the
gate, which tells you which harness code to reconcile.  Once reconciled (or once
you have confirmed nothing is owed), re-pin:

    python scripts/check-eval-app-sync.py --update

Digests ignore comments, docstrings and formatting, so only real logic changes
trip the gate.

Adding a mirror: append a `Mirror(...)` to `MIRRORS` and run `--update`.  If the
harness *intentionally* differs from the app at that point, say so in
`divergence=` - the text is printed whenever the mirror trips, so the next
person reconciling it knows which differences are deliberate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
import textwrap
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PINS_PATH = REPO_ROOT / "scripts" / "eval-app-sync.pins.json"

AUTOPILOT_TS = "frontend/src/app/services/autopilot-state.service.ts"


@dataclass(frozen=True)
class Mirror:
    """One app-side surface the eval harness reproduces rather than calls.

    Attributes:
        id: Stable key into the pins file.  Renaming one re-pins from scratch.
        app: The app-side source. ``py:<dotted.path.to.symbol>`` for Python,
            ``ts:<repo-relative file>::<anchor>`` for a TypeScript block (the
            anchor is matched literally, then brace-matched from the first
            ``{`` after it).
        harness: Where the reproduction lives, as ``<repo-relative file>::<symbol>``.
            Checked for existence, so deleting or renaming the harness side
            trips the gate too.
        kind: ``ported`` or ``default`` - see the module docstring.
        note: What is reproduced, and what to re-check when the app moves.
        divergence: A *declared, intentional* difference from the app.  Not an
            exemption - the digest is still pinned - but it tells whoever
            reconciles this mirror which differences are on purpose.
    """

    id: str
    app: str
    harness: str
    kind: str
    note: str
    divergence: str | None = None


MIRRORS: list[Mirror] = [
    # ------------------------------------------------------------------ ported
    Mirror(
        id="autopilot.phase_machine",
        app=f"ts:{AUTOPILOT_TS}::checkPhaseTransition(",
        harness="vtscore/eval/autopilot_flow.py::next_phase",
        kind="ported",
        note=(
            "The phase ordering and every transition trigger of the simulated Autopilot user. "
            "Lives in TypeScript, so there is nothing to import - it is a hand copy. If you "
            "add, remove or reorder a phase, or change what gates one, port the same change."
        ),
    ),
    Mirror(
        id="autopilot.vote_targets",
        app=f"ts:{AUTOPILOT_TS}::const INITIAL_STATE",
        harness="vtscore/eval/autopilot_flow.py::GOOD_TARGET",
        kind="ported",
        note=(
            "goodToStart / badToStart are copied as GOOD_TARGET / BAD_TARGET, which decide how "
            "many votes the simulation spends before its first learned sort. Pinned literally "
            "by tests_lib/detectors/test_autopilot_flow.py::TestPortedConstants."
        ),
    ),
    Mirror(
        id="progress.smart_status",
        app="py:vtscore.detectors.labeling_progress._compute_smart_status",
        harness="vtscore/eval/autopilot_flow.py::smart_status",
        kind="ported",
        note=(
            "The Smart indicator - error-cost flatness - one of the three gates the phase "
            "machine reads. Re-check the per-class minimum and the flatness threshold."
        ),
        divergence=(
            "The harness takes the error-cost window as an argument instead of reading the "
            "module-level `_cached_steps` MLP cache, which is built for one interactive "
            "detector advancing a vote at a time. The *rules* are copied; only the input "
            "plumbing differs - and only in where the models come from, not in how they are "
            "scored: the caller (`voting_iterations._labelset_error_costs`) re-scores the "
            "whole window against the *current* labelset every step, as `_eval_cached_models` "
            "does. Handing in a history of frozen per-step costs instead would silently change "
            "the statistic the slope measures (issue #2923), which is not a declared divergence."
        ),
    ),
    Mirror(
        id="progress.stable_status",
        app="py:vtscore.detectors.labeling_progress._compute_stable_status",
        harness="vtscore/eval/autopilot_flow.py::stable_status",
        kind="ported",
        note=(
            "The Stable indicator - prediction-flip rate. Re-check the per-class minimum, the "
            "minimum history length, and both the rate and max flip thresholds."
        ),
        divergence=(
            "Same input plumbing divergence as progress.smart_status: flip counts are passed "
            "in rather than read from `_cached_steps`."
        ),
    ),
    Mirror(
        id="progress.span_status",
        app="py:vtscore.detectors.labeling_progress._compute_span_status",
        harness="vtscore/eval/autopilot_flow.py::span_status",
        kind="ported",
        note=(
            "The Span indicator - coverage-atlas breadth - which drives the new -> done "
            "transition. Re-check the green target and the yellow cutoff."
        ),
        divergence=(
            "The app reads its green target from `CoreConfig.autopilot_goal_diversity`; the "
            "harness takes it per-run so a sweep can vary it, defaulting to the same value."
        ),
    ),
    # ----------------------------------------------------------------- default
    Mirror(
        id="training.train_and_threshold",
        app="py:vtscore.detectors.training.train_and_threshold",
        harness="vtscore/eval/voting_iterations.py::_style_train_and_calibrate",
        kind="default",
        note=(
            "The app's canonical train + calibrate pipeline, which the harness reproduces step "
            "for step (fold calibration, full-data fit, fold-anchored threshold). A new stage "
            "here - or a changed head, fold rule or ordering - has to reach the harness or its "
            "default arm trains a detector the app no longer ships. Note "
            "_mlp_train_and_calibrate is the single-vector path and reproduces the same shape."
        ),
    ),
    Mirror(
        id="training.fused_threshold",
        app="py:vtscore.detectors.training._fused_threshold",
        harness="vtscore/eval/voting_iterations.py::_safe_threshold_for_step",
        kind="default",
        note=(
            "How the cross-calibration cut and the population estimate are fused into the "
            "shipped threshold. The harness's reported operating point is only comparable to "
            "the app's if this rule matches."
        ),
    ),
    Mirror(
        id="training.calibration_score_rows",
        app="py:vtscore.detectors.training._calibration_score_rows",
        harness="vtscore/eval/voting_iterations.py::_safe_threshold_for_step",
        kind="default",
        note=(
            "Calibrating in *inference* geometry: each voted bag collapses over the rows the "
            "scorer will max-pool, not the rows the fold model trained on. Changing which rows "
            "the app calibrates over silently moves every threshold the harness reports."
        ),
    ),
    Mirror(
        id="training.blend_schedule_default",
        app="py:vtscore.detectors.training._blend_schedule_for_snap",
        harness="vtscore/eval/voting_iterations.py::simulate_voting_iterations",
        kind="default",
        note=(
            "How the app picks a safe-threshold blend schedule when none is named (per voting "
            "mode). simulate_voting_iterations resolves blend_schedule=None through "
            "production_schedule_for to match; if the app's choice becomes conditional on "
            "something else, that condition has to reach the harness too."
        ),
    ),
    Mirror(
        id="thresholds.rate_cut_no_root",
        app="py:vtscore.training.thresholds._rate_cut",
        harness="vtscore/eval/cut_rules.py::gaussian_cuts",
        kind="default",
        note=(
            "What the 'rate' rule does on a fit whose density crossing has no root between the "
            "component means. This has moved twice already - midpoint (pre-2026-08-06), bare "
            "edge, then continued past the edge at the rule's first-order slope (#2896) - and "
            "each move silently changed what the harness's *_rate arms mean relative to the "
            "app. Pinned so the next move prompts a decision instead of going unnoticed for "
            "days, which is exactly how #2900 happened."
        ),
        divergence=(
            "INTENTIONAL, decided in #2900: gaussian_cuts reports NaN and _safe_gmm_variant_rows "
            "substitutes that fit's MIDPOINT, where production continues past the component "
            "mean. The decomposition family compares cut rules against each other on one fit, so "
            "it wants a neutral, rule-independent stand-in - production's continuation exists "
            "only for 'rate' and would make it incomparable to its cross/priorfree siblings (at "
            "inclusion 0 it would break the rate == priorfree identity every report in "
            "docs/experiments/gmm-cut/ relies on). The divergence is recorded per row in "
            "cut_fallback_kind ('midpoint' vs 'continued'/'degenerate_midpoint'), so an analysis "
            "that needs the shipped path filters on it. The fold-anchored family calls "
            "gmm_cut_from_fit directly and is the faithful stand-in for the app. When re-pinning "
            "this mirror, re-check that the divergence is still the one you want AND that "
            "cut_fallback still fires on the same fits in both families - the flag being "
            "comparable is what keeps fallback_rate aggregates joinable across them."
        ),
    ),
]


class MirrorError(Exception):
    """A mirror could not be resolved at all - the app or harness side moved."""


# --------------------------------------------------------------------- digests


def _normalize_python(source: str) -> str:
    """Source text stripped of comments, docstrings and formatting.

    Token-based rather than AST-based on purpose: `ast.unparse` output is not
    guaranteed stable across the Python versions this repo supports (>=3.10),
    which would make the pins fail for whoever is not on the pinning machine's
    interpreter.  Token text is.

    Magic trailing commas are dropped as well, so that `ruff format` wrapping a
    call across lines - which it will do for a change as innocent as a longer
    variable name - doesn't read as a logic change.
    """
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER)
    out: list[str] = []
    prev_meaningful: int | None = None
    tokens = list(tokenize.generate_tokens(io.StringIO(textwrap.dedent(source) + "\n").readline))
    for i, tok in enumerate(tokens):
        if tok.type in skip:
            continue
        nxt = next((t for t in tokens[i + 1 :] if t.type not in skip), None)
        if tok.type == tokenize.STRING and prev_meaningful in (None, tokenize.NEWLINE, tokenize.INDENT):
            # A string that is an entire statement: a docstring.
            if nxt is not None and nxt.type == tokenize.NEWLINE:
                continue
        if tok.string == "," and nxt is not None and nxt.string in (")", "]", "}"):
            continue
        out.append(tok.string.strip() if tok.type == tokenize.NEWLINE else tok.string)
        prev_meaningful = tok.type
    return "\n".join(s for s in out if s)


_TS_LINE_COMMENT = re.compile(r"//[^\n]*")
_TS_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _normalize_typescript(source: str) -> str:
    """TypeScript block stripped of comments and collapsed whitespace."""
    stripped = _TS_BLOCK_COMMENT.sub(" ", source)
    stripped = _TS_LINE_COMMENT.sub(" ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def _ts_block(path: Path, anchor: str) -> str:
    """The brace-delimited block introduced by *anchor* in *path*.

    Comments are removed before brace-matching so a brace inside a comment
    cannot unbalance the scan.
    """
    text = _TS_BLOCK_COMMENT.sub(" ", path.read_text(encoding="utf-8"))
    text = _TS_LINE_COMMENT.sub(" ", text)
    start = text.find(anchor)
    if start < 0:
        raise MirrorError(f"anchor {anchor!r} not found in {path.relative_to(REPO_ROOT)}")
    if text.find(anchor, start + len(anchor)) >= 0:
        raise MirrorError(f"anchor {anchor!r} is ambiguous in {path.relative_to(REPO_ROOT)} (matches more than once)")
    open_idx = text.find("{", start)
    if open_idx < 0:
        raise MirrorError(f"no block follows anchor {anchor!r} in {path.relative_to(REPO_ROOT)}")
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise MirrorError(f"unbalanced block for anchor {anchor!r} in {path.relative_to(REPO_ROOT)}")


def _py_symbol_source(path: Path, symbol: str) -> str:
    """The source of top-level *symbol* in *path*, found by parsing not importing.

    Parsing keeps the gate fast and dependency-free: it runs before the test
    suite has installed torch, and reading a module for its text should never
    execute it.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
            return "\n".join(lines[start : node.end_lineno])
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == symbol for t in targets):
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise MirrorError(f"{path.relative_to(REPO_ROOT)} has no top-level {symbol!r} - did it move or get renamed?")


def _app_source(mirror: Mirror) -> str:
    """The normalized app-side source *mirror* is pinned against."""
    kind, _, ref = mirror.app.partition(":")
    if kind == "py":
        module_path, _, symbol = ref.rpartition(".")
        path = REPO_ROOT / (module_path.replace(".", "/") + ".py")
        if not path.exists():
            raise MirrorError(f"module {module_path} ({path.relative_to(REPO_ROOT)}) does not exist")
        return _normalize_python(_py_symbol_source(path, symbol))
    if kind == "ts":
        rel, _, anchor = ref.partition("::")
        path = REPO_ROOT / rel
        if not path.exists():
            raise MirrorError(f"{rel} does not exist")
        return _normalize_typescript(_ts_block(path, anchor))
    raise MirrorError(f"unknown app source kind {kind!r} in {mirror.app!r}")


def _digest(mirror: Mirror) -> str:
    return hashlib.sha256(_app_source(mirror).encode("utf-8")).hexdigest()


def _check_harness_side(mirror: Mirror) -> None:
    """The harness counterpart still exists under the name the manifest claims."""
    rel, _, symbol = mirror.harness.partition("::")
    path = REPO_ROOT / rel
    if not path.exists():
        raise MirrorError(f"harness file {rel} does not exist")
    if symbol and symbol not in path.read_text(encoding="utf-8"):
        raise MirrorError(f"harness symbol {symbol!r} not found in {rel} - was it renamed or removed?")


# ----------------------------------------------------------------------- pins


def _load_pins() -> dict[str, str]:
    if not PINS_PATH.exists():
        return {}
    return json.loads(PINS_PATH.read_text(encoding="utf-8"))


def _write_pins(pins: dict[str, str]) -> None:
    PINS_PATH.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class Drift:
    mirror: Mirror
    reason: str
    detail: str = ""
    extras: list[str] = field(default_factory=list)


def check() -> list[Drift]:
    """Every mirror whose app side moved, or whose two sides no longer resolve."""
    pins = _load_pins()
    drifts: list[Drift] = []
    seen: set[str] = set()
    for mirror in MIRRORS:
        if mirror.id in seen:
            raise SystemExit(f"duplicate mirror id {mirror.id!r} in MIRRORS")
        seen.add(mirror.id)
        try:
            _check_harness_side(mirror)
            digest = _digest(mirror)
        except MirrorError as exc:
            drifts.append(Drift(mirror, "unresolvable", str(exc)))
            continue
        pinned = pins.get(mirror.id)
        if pinned is None:
            drifts.append(Drift(mirror, "unpinned", "no digest recorded yet"))
        elif pinned != digest:
            drifts.append(Drift(mirror, "changed", f"pinned {pinned[:12]}, now {digest[:12]}"))
    stale = sorted(set(pins) - {m.id for m in MIRRORS})
    if stale:
        drifts.append(
            Drift(
                MIRRORS[0],
                "stale-pins",
                "pins recorded for mirrors that no longer exist: " + ", ".join(stale),
                extras=stale,
            )
        )
    return drifts


def update() -> int:
    pins: dict[str, str] = {}
    for mirror in MIRRORS:
        _check_harness_side(mirror)
        pins[mirror.id] = _digest(mirror)
    _write_pins(pins)
    print(f"Pinned {len(pins)} eval/app mirrors to {PINS_PATH.relative_to(REPO_ROOT)}")
    return 0


def _report(drifts: list[Drift]) -> None:
    print("The eval framework mirrors app code that has since changed.")
    print("")
    print("The eval default arm has to BE the app's algorithm - that is the only")
    print("thing that makes a deviation arm meaningful. Reconcile each mirror below,")
    print("then re-pin with:  python scripts/check-eval-app-sync.py --update")
    for drift in drifts:
        mirror = drift.mirror
        print("")
        if drift.reason == "stale-pins":
            print(f"  * {drift.detail}")
            print("    Run --update to drop them.")
            continue
        print(f"  * {mirror.id}  [{mirror.kind}, {drift.reason}]")
        print(f"      app:     {mirror.app.partition(':')[2]}")
        print(f"      harness: {mirror.harness}")
        print(f"      {mirror.note}")
        if mirror.divergence:
            print(f"      DECLARED DIVERGENCE: {mirror.divergence}")
        if drift.detail and drift.reason != "changed":
            print(f"      {drift.detail}")
    print("")
    print("If the harness already tracks the change (or is unaffected), --update alone")
    print("is the right answer. If the harness now *intentionally* differs, record why")
    print("in the mirror's divergence= field in scripts/check-eval-app-sync.py.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--update",
        action="store_true",
        help="re-pin every mirror to the current app source, after reconciling the harness",
    )
    args = parser.parse_args(argv)
    if args.update:
        return update()
    drifts = check()
    if drifts:
        _report(drifts)
        return 1
    print(f"Eval/app sync: {len(MIRRORS)} mirrors up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
