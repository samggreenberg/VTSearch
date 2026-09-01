"""The extension-doc drift gate itself: `scripts/check-extension-docs.py`.

Two doc sets state the contract for the same plugin ABCs in their own words,
so each can drift from the code and from the other with nothing failing. The
gate exists because that already happened three times at once (issue #3442).
A gate that stops noticing is worse than no gate — the repo now *believes* its
two extension guides agree — so pin both halves: every invariant fires on a
real defect, and stays quiet on the legitimate shapes the docs actually use.

The historical defects are pinned as regression tests below, reconstructed as
synthetic sections, so a future rewrite of either doc set cannot quietly
reintroduce them.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-extension-docs.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_extension_docs_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()
CLASSES = gate._index_classes()
MODULE_LEVEL = gate._index_module_level()


def check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: str,
    classes: tuple[str, ...] = ("MediaEmbedder",),
    heading: str = "The contract",
    ignore: frozenset[str] = frozenset(),
) -> list[str]:
    """Run the section invariants over a synthetic doc; return the problems.

    The class index still comes from the real tree — the point is to check
    prose against the shipped ABCs — but the doc under test is synthetic, so a
    test can describe a defect without editing a real guide.
    """
    doc = tmp_path / "synthetic.md"
    doc.write_text(f"## {heading}\n\n{body}\n", encoding="utf-8")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "SECTIONS", [("synthetic.md", heading, classes)])
    monkeypatch.setattr(gate, "IGNORE", {("synthetic.md", heading): ignore} if ignore else {})
    return gate._check_sections(CLASSES, MODULE_LEVEL)


class TestRepoIsClean:
    """The gate passes on the repo as committed."""

    def test_no_drift(self):
        assert gate.main() == 0

    def test_runs_fast_enough_to_gate(self):
        """It runs on every `./run-tests.sh`, so it must stay cheap.

        The bound is deliberately loose: this asserts on wall clock, and the
        suite runs under ``pytest -n auto`` on whatever machine is free. It is
        here to catch an order-of-magnitude regression (an accidental import of
        the package under test, a quadratic sweep), not to police tenths.
        """
        start = time.monotonic()
        gate._check_sections(gate._index_classes(), gate._index_module_level())
        assert time.monotonic() - start < 20.0

    def test_every_section_is_reachable(self):
        """Each configured section resolves to a real heading in a real file."""
        for doc, heading, _ in gate.SECTIONS:
            path = REPO_ROOT / doc
            assert path.exists(), f"{doc} is in SECTIONS but missing"
            lines = path.read_text(encoding="utf-8").splitlines()
            assert gate._section_lines(lines, heading) is not None, f"{doc}: no § {heading}"

    def test_both_doc_sets_are_covered(self):
        """Neither tier may drop out of the config wholesale."""
        docs = {doc for doc, _, _ in gate.SECTIONS}
        assert any(d.startswith("docs/EXTENDING-") for d in docs)
        assert any(d.startswith("vtscore/docs/extending/") for d in docs)


class TestMemberExtraction:
    """What counts as a member reference inside a contract table."""

    def test_first_cell_bare_name(self):
        found = gate._documented_members(["| `name` | `str` | Unique key |"])
        assert [ident for ident, _ in found] == ["name"]

    def test_first_cell_strips_signature_and_annotation(self):
        rows = [
            "| `clip(media)` | x | y |",
            "| `supports_text: bool` | x | y |",
            "| `media_type` (property) | x | y |",
        ]
        assert [i for i, _ in gate._documented_members(rows)] == ["clip", "supports_text", "media_type"]

    def test_later_cells_yield_private_names_and_calls(self):
        """The `_patch_forward` defect lived in a description column, not cell one."""
        row = "| `supports_patch_regions` | False | implement `_patch_forward_impl(media)` |"
        assert [i for i, _ in gate._documented_members([row])] == [
            "supports_patch_regions",
            "_patch_forward_impl",
        ]

    def test_later_cells_ignore_bare_types(self):
        """`str` in a signature column is a type, not a member named `str`."""
        row = "| `name` | `str` | Returns a `dict` of `bool` |"
        assert [i for i, _ in gate._documented_members([row])] == ["name"]

    def test_space_before_paren_is_not_a_call(self):
        """`classmethod (str, dict) -> Self` is an annotation, not a member."""
        row = "| `from_config(name, config)` | `classmethod (str, dict) -> Self` | Build |"
        assert [i for i, _ in gate._documented_members([row])] == ["from_config"]

    def test_prose_outside_tables_is_ignored(self):
        assert gate._documented_members(["Do not override `embed_media()` here."]) == []


class TestMemberInvariant:
    """Invariant 1: a documented member must exist on the class."""

    def test_real_member_passes(self, monkeypatch, tmp_path):
        assert check(monkeypatch, tmp_path, "| `_embed_media_impl(media)` | x | y |") == []

    def test_invented_member_fails(self, monkeypatch, tmp_path):
        problems = check(monkeypatch, tmp_path, "| `_embed_nonsense_impl(media)` | x | y |")
        assert len(problems) == 1
        assert "_embed_nonsense_impl" in problems[0]

    def test_inherited_member_passes(self, monkeypatch, tmp_path):
        """`MediaCleaner` inherits from `MediaClipper`; the base's members count."""
        assert check(monkeypatch, tmp_path, "| `clip(media)` | x | y |", ("MediaCleaner",)) == []

    def test_class_name_passes(self, monkeypatch, tmp_path):
        """Docs legitimately tabulate types alongside members."""
        assert check(monkeypatch, tmp_path, "| `FetchedItem` | `path` | y |", ("MediaSource",)) == []

    def test_module_level_function_passes(self, monkeypatch, tmp_path):
        """A free function is a real symbol, just not a method."""
        assert check(monkeypatch, tmp_path, "| `x` | y | set via `set_progress_callback()` |") == []

    def test_ignore_entry_excuses_a_dict_key(self, monkeypatch, tmp_path):
        body = "| `step` | int | A parameter-dict key |"
        assert check(monkeypatch, tmp_path, body, ("MediaClipper",)) != []
        assert check(monkeypatch, tmp_path, body, ("MediaClipper",), ignore=frozenset({"step"})) == []

    def test_unknown_class_is_reported(self, monkeypatch, tmp_path):
        problems = check(monkeypatch, tmp_path, "| `x` | y | z |", ("NoSuchAbc",))
        assert len(problems) == 1 and "NoSuchAbc" in problems[0]

    def test_multi_class_section_accepts_any_member(self, monkeypatch, tmp_path):
        """`detect` is on `Detector`, not on the `Processor` base."""
        body = "| `detect(media)` | `(dict) -> bool` | y |"
        assert check(monkeypatch, tmp_path, body, ("Processor",)) != []
        assert check(monkeypatch, tmp_path, body, ("Processor", "Detector")) == []


class TestWrapperInvariant:
    """Invariant 3: a documented wrapper must be shown beside its hook."""

    def test_wrapper_alone_fails(self, monkeypatch, tmp_path):
        """The `embed_text` defect: the wrapper presented as the override point."""
        problems = check(monkeypatch, tmp_path, "| `embed_text(text)` | `None` | Override to sort |")
        assert len(problems) == 1
        assert "_embed_text_impl" in problems[0] and "wrapper" in problems[0]

    def test_wrapper_beside_its_hook_passes(self, monkeypatch, tmp_path):
        body = "| `embed_text(text)` | x | wraps `_embed_text_impl(text)` |"
        assert check(monkeypatch, tmp_path, body) == []

    def test_hook_alone_passes(self, monkeypatch, tmp_path):
        assert check(monkeypatch, tmp_path, "| `_embed_text_impl(text)` | x | y |") == []

    def test_method_without_a_hook_is_not_a_wrapper(self, monkeypatch, tmp_path):
        """`embed_medias` has no `_embed_medias_impl`, so it is safe to name."""
        assert check(monkeypatch, tmp_path, "| `embed_medias(medias)` | x | y |") == []


class TestCoverageInvariant:
    """Invariant 2: a contract-shaped heading must be registered."""

    def test_repo_has_no_unregistered_sections(self):
        assert gate._check_coverage() == []

    @pytest.mark.parametrize(
        "heading",
        ["MediaFoo abstract interface reference", "The contract", "Capability flags"],
    )
    def test_unregistered_heading_is_flagged(self, monkeypatch, tmp_path, heading):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "EXTENDING-synthetic.md").write_text(f"## {heading}\n", encoding="utf-8")
        monkeypatch.setattr(gate, "ROOT", tmp_path)
        monkeypatch.setattr(gate, "SECTIONS", [])
        problems = gate._check_coverage()
        assert len(problems) == 1 and heading in problems[0]

    def test_ordinary_heading_is_not_flagged(self, monkeypatch, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "EXTENDING-synthetic.md").write_text("## Worked example\n", encoding="utf-8")
        monkeypatch.setattr(gate, "ROOT", tmp_path)
        monkeypatch.setattr(gate, "SECTIONS", [])
        assert gate._check_coverage() == []


class TestHistoricalDrift:
    """The three defects that motivated the gate, pinned so they can't return."""

    def test_embed_text_override_point(self):
        """`embedders.md` told authors to override the L2-normalizing wrapper."""
        text = (REPO_ROOT / "vtscore" / "docs" / "extending" / "embedders.md").read_text()
        assert "| `embed_text(text)` | `None` |" not in text
        assert "`_embed_text_impl(text)`" in text

    def test_patch_forward_hook_name(self):
        """`EXTENDING-media.md` named a hook nothing calls."""
        text = (REPO_ROOT / "docs" / "EXTENDING-media.md").read_text()
        assert "`_patch_forward(image)" not in text
        assert "_patch_forward_impl" in text

    def test_exporter_base_class_name(self):
        """The library index still named the pre-rename alias as the base class.

        Not mechanically checkable — `LabelsetExporter` is a real, permanently
        supported alias, so no member check can tell "outdated preferred name"
        from "deliberate compatibility alias". Pinned as text instead.
        """
        text = (REPO_ROOT / "vtscore" / "docs" / "extending" / "README.md").read_text()
        assert "| `EXPORTER` | `LabelsetExporter` |" not in text
        assert "`ResultsExporter`" in text


class TestCrossLinks:
    """Each doc set names the other, in both directions (issue #3442)."""

    def test_front_door_links_to_the_library_set(self):
        text = (REPO_ROOT / "docs" / "EXTENDING.md").read_text()
        assert "../vtscore/docs/extending/README.md" in text

    def test_library_index_links_back_to_the_front_door(self):
        text = (REPO_ROOT / "vtscore" / "docs" / "extending" / "README.md").read_text()
        assert "../../../docs/EXTENDING.md" in text

    @pytest.mark.parametrize(
        ("doc", "guide"),
        [
            ("docs/EXTENDING-media.md", "media-types"),
            ("docs/EXTENDING-media.md", "embedders"),
            ("docs/EXTENDING-media.md", "clippers"),
            ("docs/EXTENDING-media.md", "converters"),
            ("docs/EXTENDING-plugins.md", "dataset-importers"),
            ("docs/EXTENDING-plugins.md", "results-exporters"),
            ("docs/EXTENDING-plugins.md", "label-importers"),
            ("docs/EXTENDING-plugins.md", "labelset-sources"),
        ],
    )
    def test_app_guide_points_at_its_library_counterpart(self, doc, guide):
        text = (REPO_ROOT / doc).read_text()
        assert f"../vtscore/docs/extending/{guide}.md" in text

    @pytest.mark.parametrize(
        "guide",
        [
            "media-types",
            "embedders",
            "clippers",
            "converters",
            "dataset-importers",
            "results-exporters",
            "label-importers",
            "labelset-sources",
        ],
    )
    def test_library_guide_points_at_its_app_counterpart(self, guide):
        text = (REPO_ROOT / "vtscore" / "docs" / "extending" / f"{guide}.md").read_text()
        assert "App-side counterpart:" in text
        assert "../../../docs/EXTENDING-" in text
