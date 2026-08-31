"""Tests for the DocMarks corpus builder (``scripts/experiments/docmarks/``).

The builder assembles one instance-retrieval corpus out of four document
sources.  Everything network-touching lives behind ``fetch_*`` and is not
exercised here; everything that decides *what the corpus means* is pure and is
pinned below.

Three of these tests guard properties that a plausible-looking refactor would
quietly break, and whose breakage would not show up as a crash — only as
numbers that are wrong in a direction nobody notices:

* **tier nesting** — a study on ``docmarks_s`` and one on ``docmarks_l`` are
  only comparable if the small tier is a subset of the large one.  Sampling
  distractors with anything order-dependent silently breaks that.
* **contamination** — Tobacco800 and UCSF's Tobacco industry are the same
  underlying archive, so scoring Tobacco800 classes against UCSF tobacco pages
  counts correct retrievals as false positives.
* **synthetic box tightness** — a rotated paste's ground-truth box must come
  from the alpha bbox, not the paste rectangle, or every query crop carries a
  third of a page of blank paper.

The scripts are loose modules, not package members, so the directory goes on
``sys.path`` and they are imported by name.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_DOCMARKS = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "docmarks"


@pytest.fixture(scope="module", autouse=True)
def _on_path():
    sys.path.insert(0, str(_DOCMARKS))
    yield
    with pytest.MonkeyPatch.context():
        pass
    if str(_DOCMARKS) in sys.path:
        sys.path.remove(str(_DOCMARKS))


@pytest.fixture(scope="module")
def mods(_on_path):
    return {
        "cfg": importlib.import_module("docmarks_config"),
        "common": importlib.import_module("sources._common"),
        "spods": importlib.import_module("sources.spods"),
        "staver": importlib.import_module("sources.staver"),
        "tobacco800": importlib.import_module("sources.tobacco800"),
        "ucsf": importlib.import_module("sources.ucsf"),
        "artwork": importlib.import_module("sources.artwork"),
        "cluster": importlib.import_module("cluster_marks"),
        "build": importlib.import_module("build_corpus"),
        "synth": importlib.import_module("synth_compose"),
        "embed": importlib.import_module("embed_corpus"),
        "roster": importlib.import_module("roster"),
        "shortlist": importlib.import_module("shortlist"),
        "audit": importlib.import_module("audit_to_corrections"),
        "report": importlib.import_module("make_report"),
    }


def _page(mods, page_id, source, marks=(), path="x.png", w=1000, h=1400):
    Mark, Page = mods["common"].Mark, mods["common"].Page
    return Page(
        page_id=page_id,
        source=source,
        path=path,
        width=w,
        height=h,
        marks=[Mark(kind=k, box=b, class_id=c, provenance=p) for k, b, c, p in marks],
    )


# ---------------------------------------------------------------- primitives


class TestStableRank:
    def test_is_deterministic_and_in_range(self, mods):
        rank = mods["common"].stable_rank
        values = [rank(f"page/{i}", "salt") for i in range(200)]
        assert all(0.0 <= v < 1.0 for v in values)
        assert values == [rank(f"page/{i}", "salt") for i in range(200)]

    def test_salt_changes_the_ordering(self, mods):
        rank = mods["common"].stable_rank
        keys = [f"page/{i}" for i in range(50)]
        assert sorted(keys, key=lambda k: rank(k, "a")) != sorted(keys, key=lambda k: rank(k, "b"))


class TestMaskToBoxes:
    def test_finds_separated_components(self, mods):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:40, 10:50] = 255
        mask[120:160, 130:170] = 255
        boxes = mods["common"].mask_to_boxes(mask, min_area_frac=0.0)
        assert len(boxes) == 2
        assert (10, 10, 40, 30) in boxes

    def test_drops_speckle_below_the_area_floor(self, mods):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:40, 10:50] = 255
        mask[100, 100] = 255  # one pixel
        boxes = mods["common"].mask_to_boxes(mask, min_area_frac=0.001)
        assert len(boxes) == 1

    def test_empty_mask_is_no_boxes(self, mods):
        assert mods["common"].mask_to_boxes(np.zeros((50, 50), dtype=np.uint8)) == []

    def test_inverted_masks_are_detected_not_swallowed(self, mods):
        # SPODS ships 1-bit masks with the mark BLACK on white paper. Read as
        # "non-zero is foreground" this yields one page-sized box per page --
        # which does not crash, it silently produces 1,088 identical rectangles
        # that cluster into a single class and look like a working corpus. On the
        # real data that is exactly what happened: 2,176 marks, 1 class.
        mask = np.full((200, 200), 255, dtype=np.uint8)
        mask[10:40, 10:50] = 0  # the mark, dark on light
        boxes = mods["common"].mask_to_boxes(mask, min_area_frac=0.0)
        assert boxes == [(10, 10, 40, 30)]

    def test_normal_polarity_still_works(self, mods):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:40, 10:50] = 255
        assert mods["common"].mask_to_boxes(mask, min_area_frac=0.0) == [(10, 10, 40, 30)]

    def test_polarity_can_be_forced(self, mods):
        # A genuinely dense mask (body text on a full page) can be forced rather
        # than left to the minority heuristic.
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:80, :] = 255  # 80% lit: "auto" would invert this
        auto = mods["common"].mask_to_boxes(mask, min_area_frac=0.0, polarity="auto")
        forced = mods["common"].mask_to_boxes(mask, min_area_frac=0.0, polarity="light")
        assert auto == [(0, 80, 100, 20)]
        assert forced == [(0, 0, 100, 80)]

    def test_an_unknown_polarity_is_refused(self, mods):
        with pytest.raises(ValueError, match="unknown polarity"):
            mods["common"].mask_to_boxes(np.zeros((10, 10), dtype=np.uint8), polarity="sideways")


class TestMergeOverlapping:
    def test_merges_a_fragmented_stamp_into_one_mark(self, mods):
        # A rubber stamp's mask breaks into a ring plus its inner text; left
        # unmerged each fragment becomes its own "class" and the inventory is
        # nonsense.
        fragments = [(10, 10, 30, 5), (10, 16, 30, 5), (10, 22, 30, 5)]
        merged = mods["common"].merge_overlapping(fragments, gap=6)
        assert merged == [(10, 10, 30, 17)]

    def test_leaves_distant_marks_alone(self, mods):
        boxes = [(10, 10, 20, 20), (500, 500, 20, 20)]
        assert len(mods["common"].merge_overlapping(boxes, gap=6)) == 2


# ------------------------------------------------------------------- sources


class TestSpods:
    @pytest.mark.parametrize(
        "name,expected",
        [("image (417).png", 417), ("image (1).png", 1), ("IMAGE (23).PNG", 23), ("notes.txt", None)],
    )
    def test_page_number(self, mods, name, expected):
        assert mods["spods"].page_number(Path(name)) == expected

    def test_marks_for_page_reads_every_category(self, mods, tmp_path):
        from PIL import Image

        gt = tmp_path / "gt"
        for kind, box in (("logo", (20, 20, 60, 40)), ("stamp", (200, 300, 80, 80))):
            arr = np.zeros((600, 500), dtype=np.uint8)
            x, y, w, h = box
            arr[y : y + h, x : x + w] = 255
            (gt / kind).mkdir(parents=True)
            Image.fromarray(arr).save(gt / kind / "image (7).png")

        marks = mods["spods"].marks_for_page(gt, 7, min_area_frac=0.0)
        by_kind = {m.kind: m.box for m in marks}
        assert by_kind == {"logo": (20, 20, 60, 40), "stamp": (200, 300, 80, 80)}
        # Identity is never invented at parse time: SPODS does not ship it.
        assert all(m.class_id is None for m in marks)
        assert all(m.provenance == "gt" for m in marks)

    def test_find_tree_reports_an_unrecognised_layout(self, mods, tmp_path):
        (tmp_path / "something-else").mkdir()
        with pytest.raises(mods["common"].FetchError, match="layout not recognised"):
            mods["spods"].find_tree(tmp_path)


class TestStaver:
    def test_parse_info_normalises_keys_and_types(self, mods):
        parsed = mods["staver"].parse_info(
            "Number of stamps: 2\nSignature present : yes\nStamp Color: colored\nOverlap: no\n"
        )
        assert parsed["number_of_stamps"] == 2
        assert parsed["signature_present"] is True
        assert parsed["overlap"] is False
        assert parsed["stamp_color"] == "colored"

    def test_expected_stamp_count_accepts_the_known_spellings(self, mods):
        for key in ("number_of_stamps", "stamps", "num_stamps"):
            assert mods["staver"].expected_stamp_count({key: 3}) == 3
        assert mods["staver"].expected_stamp_count({"stamp_color": "black"}) is None


class TestTobacco800:
    GEDI = """<?xml version="1.0"?>
    <GEDI>
      <DL_DOCUMENT src="xyz.tif">
        <DL_PAGE gedi_type="DL_PAGE" src="xyz00001.tif" width="2544" height="3295">
          <DL_ZONE gedi_type="DLLogo" id="1" col="220" row="140" width="600" height="180"/>
          <DL_ZONE gedi_type="DLSignature" id="2" col="300" row="2400"
                   width="500" height="200" AuthorID="Horrigan"/>
          <DL_ZONE gedi_type="DLText" id="3" col="0" row="0" width="10" height="10"/>
        </DL_PAGE>
      </DL_DOCUMENT>
    </GEDI>"""

    def test_parses_logo_and_signature_zones_only(self, mods):
        parsed = mods["tobacco800"].parse_gedi(self.GEDI)
        marks = parsed["xyz00001"]
        assert {m.kind for m in marks} == {"logo", "signature"}

    def test_logo_box_uses_col_row_width_height(self, mods):
        logo = next(m for m in mods["tobacco800"].parse_gedi(self.GEDI)["xyz00001"] if m.kind == "logo")
        assert logo.box == (220, 140, 600, 180)

    def test_signature_identity_becomes_a_class_id(self, mods):
        sig = next(m for m in mods["tobacco800"].parse_gedi(self.GEDI)["xyz00001"] if m.kind == "signature")
        assert sig.class_id == "tobacco800/signature_horrigan"

    def test_a_bare_serial_id_is_not_treated_as_an_identity(self, mods):
        # `id` is a per-zone serial. Reading it as an identity would give every
        # single mark its own singleton class, and the corpus would look full of
        # classes while containing none.
        logo = next(m for m in mods["tobacco800"].parse_gedi(self.GEDI)["xyz00001"] if m.kind == "logo")
        assert logo.class_id is None


class TestUcsf:
    def test_build_query_quotes_multiword_values(self, mods):
        q = mods["ucsf"].build_query(industry="Fossil Fuel", author="LOR, LORILLARD")
        assert 'industry:"Fossil Fuel"' in q
        assert 'author:"LOR, LORILLARD"' in q
        assert "pages:1" in q

    def test_pdf_url_uses_the_split_character_scheme(self, mods):
        assert mods["ucsf"].pdf_url("ffbb0019").endswith("/f/f/b/b/ffbb0019/ffbb0019.pdf")

    def test_pdf_url_rejects_a_short_id(self, mods):
        with pytest.raises(ValueError, match="too short"):
            mods["ucsf"].pdf_url("ab")

    @pytest.mark.parametrize(
        "date,expected",
        [("1996 January 24", 1996), ("1965", 1965), ("2003 December 04", 2003), ("", None), (None, None)],
    )
    def test_year(self, mods, date, expected):
        assert mods["ucsf"].year(date) == expected

    def test_first_value_unwraps_solr_multivalued_fields(self, mods):
        assert mods["ucsf"].first_value({"collection": ["Lorillard Records", "MSA"]}, "collection") == (
            "Lorillard Records"
        )

    def test_an_author_is_a_candidate_pool_never_a_class(self, mods):
        # The metadata says the page is *from* Philip Morris; it has never
        # looked at the mark. Making it a class id would put two different
        # artworks in one class whenever a company redesigned its letterhead,
        # and split one artwork across two classes whenever subsidiaries shared
        # it -- both of them errors the eval exists to measure, written straight
        # into the labels.
        doc = {"id": "ffbb0019", "author": ["PHILIP MORRIS"], "documentdate": "1996 January 24"}
        page = mods["ucsf"].doc_to_page(doc, "/tmp/x.png", 1700, 2200, letterhead_author="PHILIP MORRIS")
        (mark,) = page.marks
        assert mark.class_id is None
        assert mark.provenance == "candidate"
        assert page.meta["letterhead_author"] == "PHILIP MORRIS"

    def test_candidate_carries_a_locatable_band(self, mods):
        page = mods["ucsf"].doc_to_page(
            {"id": "ffbb0019"}, "/tmp/x.png", 1700, 2200, letterhead_author="RJR", band_frac=0.2
        )
        # A mark nobody can see cannot be adjudicated, so the candidate gets a
        # coarse top-of-page strip to cluster on -- never a ground-truth box.
        assert page.marks[0].box == (0, 0, 1700, 440)

    def test_the_year_never_reaches_a_class_id(self, mods):
        doc = {"id": "ffbb0019", "documentdate": "1965 May 3"}
        page = mods["ucsf"].doc_to_page(doc, "/tmp/x.png", 100, 100, letterhead_author="RJR")
        assert page.meta["year"] == 1965
        # Era is a fact about the calendar, not about the mark. A class means
        # "this artwork" and nothing else.
        assert page.marks[0].class_id is None

    def test_a_page_with_no_author_carries_no_marks(self, mods):
        page = mods["ucsf"].doc_to_page({"id": "ffbb0019"}, "/tmp/x.png", 10, 10)
        assert page.marks == []


class TestArtworkVoc:
    def test_parses_pascal_voc_boxes(self, mods):
        xml = """<annotation><object><name>Nike</name>
                 <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>70</ymax></bndbox>
                 </object></annotation>"""
        assert mods["artwork"].parse_voc(xml) == [("Nike", (10, 20, 100, 50))]

    def test_skips_degenerate_boxes(self, mods):
        xml = """<annotation><object><name>X</name>
                 <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>10</xmax><ymax>70</ymax></bndbox>
                 </object></annotation>"""
        assert mods["artwork"].parse_voc(xml) == []


# ------------------------------------------------------------- contamination


class TestContamination:
    def test_tobacco800_may_not_use_ucsf_tobacco_as_distractors(self, mods):
        # Both are IIT-CDIP. A UCSF tobacco page is certain to carry more
        # instances of these same letterheads, so scoring against it counts
        # correct retrievals as false positives.
        assert not mods["cfg"].eligible_distractor("tobacco800", "ucsf", "Tobacco")

    def test_tobacco800_may_use_other_ucsf_industries(self, mods):
        assert mods["cfg"].eligible_distractor("tobacco800", "ucsf", "Opioids")

    def test_spods_may_use_ucsf_freely(self, mods):
        assert mods["cfg"].eligible_distractor("spods", "ucsf", "Tobacco")

    def test_no_source_is_its_own_distractor(self, mods):
        for source in ("spods", "staver", "tobacco800", "ucsf", "synth"):
            assert not mods["cfg"].eligible_distractor(source, source)


# ------------------------------------------------------------ class admission


class TestClassAdmission:
    def _corpus(self, mods, big=12, small=3):
        pages = []
        for i in range(big):
            pages.append(_page(mods, f"spods/{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/a", "gt")]))
        for i in range(small):
            pages.append(_page(mods, f"spods/x{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/b", "gt")]))
        return pages

    def test_survival_curve_counts_classes_per_threshold(self, mods):
        pages = self._corpus(mods)
        inv = mods["build"].class_inventory(pages)
        curve = mods["build"].survival_curve(inv, (2, 5, 10, 20))
        assert curve == {2: 2, 5: 1, 10: 1, 20: 0}

    def test_min_instances_rejects_the_thin_class(self, mods):
        pages = self._corpus(mods)
        inv = mods["build"].class_inventory(pages)
        admitted, rejected = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        assert set(admitted) == {"spods/a"}
        assert "instance(s) < min_instances" in rejected["spods/b"]

    def test_tiny_marks_are_rejected_with_a_reason(self, mods):
        pages = [
            _page(mods, f"spods/{i:03d}", "spods", [("logo", (0, 0, 12, 10), "spods/tiny", "gt")]) for i in range(20)
        ]
        inv = mods["build"].class_inventory(pages)
        admitted, rejected = mods["build"].admit_classes(pages, inv, min_instances=5, min_mark_px=32)
        assert admitted == {}
        assert "min_mark_px" in rejected["spods/tiny"]

    def test_signatures_are_never_queryable(self, mods):
        pages = [
            _page(mods, f"t/{i:03d}", "tobacco800", [("signature", (0, 0, 400, 200), "tobacco800/signature_x", "gt")])
            for i in range(30)
        ]
        inv = mods["build"].class_inventory(pages)
        admitted, rejected = mods["build"].admit_classes(pages, inv, min_instances=5, min_mark_px=32)
        assert admitted == {}
        assert "not queryable" in rejected["tobacco800/signature_x"]

    def test_band_classes_skip_the_mark_size_floor(self, mods):
        # A band's pixel size describes the top-of-page strip, not the mark, so
        # checking it against the 32px mark floor compares the wrong number
        # against the wrong threshold -- and reporting it as median_mark_px
        # would misdescribe the class.
        pages = [
            _page(mods, f"ucsf/{i:03d}", "ucsf", [("logo", (0, 0, 1700, 440), "ucsf/logo_a_0", "clustered_band")])
            for i in range(40)
        ]
        inv = mods["build"].class_inventory(pages)
        admitted, _ = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        assert admitted["ucsf/logo_a_0"]["located_by"] == "band"
        assert admitted["ucsf/logo_a_0"]["median_mark_px"] is None

    def test_unlocated_classes_are_rejected(self, mods):
        pages = [
            _page(mods, f"ucsf/{i:03d}", "ucsf", [("logo", (0, 0, 0, 0), "ucsf/nowhere", "candidate")])
            for i in range(40)
        ]
        inv = mods["build"].class_inventory(pages)
        admitted, rejected = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        assert admitted == {}
        assert "no located instances" in rejected["ucsf/nowhere"]

    def test_admitted_classes_record_their_eligible_distractors(self, mods):
        pages = self._corpus(mods)
        inv = mods["build"].class_inventory(pages)
        admitted, _ = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        eligible = admitted["spods/a"]["eligible_distractor_sources"]
        assert "spods" not in eligible
        assert "ucsf" in eligible


# ------------------------------------------------------------------- roster


class TestRoster:
    def _pages(self, mods, n_a=14, n_b=3):
        pages = [
            _page(mods, f"spods/a{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/a", "clustered")])
            for i in range(n_a)
        ]
        pages += [
            _page(mods, f"spods/b{i:03d}", "spods", [("logo", (0, 0, 12, 10), "spods/b", "clustered")])
            for i in range(n_b)
        ]
        return pages

    def _admit(self, mods, pages, roster=None):
        inv = mods["build"].class_inventory(pages)
        return mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32, roster=roster)

    def test_without_a_roster_the_bars_decide(self, mods):
        admitted, rejected = self._admit(mods, self._pages(mods))
        assert set(admitted) == {"spods/a"}
        assert "spods/b" in rejected

    def test_a_roster_restricts_admission_to_its_own_classes(self, mods):
        roster = mods["roster"].Roster(name="t", classes=["spods/a"])
        admitted, rejected = self._admit(mods, self._pages(mods), roster)
        assert set(admitted) == {"spods/a"}
        assert rejected["spods/b"] == "not on the roster"

    def test_a_roster_class_overrides_the_bars_but_records_why(self, mods):
        # The human who picked it knows something the threshold does not; the
        # override is kept visible in the artifact rather than silently waived.
        roster = mods["roster"].Roster(name="t", classes=["spods/a", "spods/b"])
        admitted, _ = self._admit(mods, self._pages(mods), roster)
        assert set(admitted) == {"spods/a", "spods/b"}
        assert admitted["spods/a"]["caveats"] == []
        assert any("min_instances" in c for c in admitted["spods/b"]["caveats"])
        assert any("min_mark_px" in c for c in admitted["spods/b"]["caveats"])

    def test_classes_start_unverified(self, mods):
        roster = mods["roster"].Roster(name="t", classes=["spods/a"])
        admitted, _ = self._admit(mods, self._pages(mods), roster)
        # Until the membership pass runs, a class is a clustering proposal.
        assert admitted["spods/a"]["audit"]["membership_verified"] is False
        assert admitted["spods/a"]["on_roster"] is True

    def test_check_reports_drift_between_roster_and_corpus(self, mods):
        roster = mods["roster"].Roster(name="t", classes=["spods/a", "spods/gone"])
        present, missing = mods["roster"].check(roster, ["spods/a", "spods/b"])
        assert present == ["spods/a"]
        assert missing == ["spods/gone"]

    def test_roster_round_trips_and_deduplicates(self, mods, tmp_path):
        path = tmp_path / "roster.json"
        mods["roster"].save(mods["roster"].Roster("t", ["b", "a", "b"], notes="why"), path)
        back = mods["roster"].load(path)
        assert back.classes == ["a", "b"]
        assert back.notes == "why"

    def test_known_negatives_come_only_from_verified_sources(self, mods):
        meta = {
            "page_ids": ["spods/a000"],
            "eligible_distractor_sources": ["ucsf", "synth"],
        }
        pages_by_source = {
            "spods": ["spods/a000", "spods/x001"],
            "ucsf": ["ucsf/d1", "ucsf/d2"],
        }
        # SPODS contaminates SPODS by default -- but once SPODS has been
        # exhaustively checked for this class, its non-members become *known*
        # negatives: same scanner, same paper, verified clean, which is the
        # hardest and most useful negative there is.
        split = mods["roster"].eligible_pages(meta, pages_by_source, verified_negative_sources=["spods"])
        assert split["positive"] == ["spods/a000"]
        assert split["known_negative"] == ["spods/x001"]
        assert split["presumed_negative"] == ["ucsf/d1", "ucsf/d2"]

    def test_without_verification_same_source_pages_are_not_usable(self, mods):
        meta = {"page_ids": ["spods/a000"], "eligible_distractor_sources": ["ucsf"]}
        split = mods["roster"].eligible_pages(meta, {"spods": ["spods/a000", "spods/x001"], "ucsf": ["ucsf/d1"]})
        assert split["known_negative"] == []
        assert split["presumed_negative"] == ["ucsf/d1"]


class TestMembershipAudit:
    def _setup(self, mods):
        pages = [
            _page(mods, f"spods/{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/a", "clustered")])
            for i in range(5)
        ]
        classes = {
            "spods/a": {
                "class_id": "spods/a",
                "n_instances": 5,
                "page_ids": [f"spods/{i:03d}" for i in range(5)],
                "audit": {"membership_verified": False, "rejected_page_ids": []},
            }
        }
        return pages, classes

    def test_ok_verifies_without_dropping_anything(self, mods):
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "ok"}
        changes, problems = mods["audit"].apply_membership(pages, classes, [row])
        assert not problems
        assert classes["spods/a"]["n_instances"] == 5
        assert classes["spods/a"]["audit"]["membership_verified"] is True

    def test_rejected_indices_are_removed_from_the_class(self, mods):
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "1, 3"}
        mods["audit"].apply_membership(pages, classes, [row])
        assert classes["spods/a"]["page_ids"] == ["spods/000", "spods/002", "spods/004"]
        assert classes["spods/a"]["audit"]["rejected_page_ids"] == ["spods/001", "spods/003"]

    def test_a_rejected_instance_keeps_its_box_and_page(self, mods):
        # It stops being a positive, but the page stays a *known* negative and
        # the mark is still a real mark a later roster might want.
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "1"}
        mods["audit"].apply_membership(pages, classes, [row])
        dropped = next(p for p in pages if p.page_id == "spods/001")
        assert len(dropped.marks) == 1
        assert dropped.marks[0].class_id is None
        assert dropped.marks[0].box == (0, 0, 200, 120)

    def test_an_out_of_range_index_is_refused_not_silently_clamped(self, mods):
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "9"}
        changes, problems = mods["audit"].apply_membership(pages, classes, [row])
        assert not changes
        assert "outside 0..4" in problems[0]
        assert classes["spods/a"]["n_instances"] == 5

    def test_a_malformed_verdict_is_refused(self, mods):
        pages, classes = self._setup(mods)
        row = {"class_id": "spods/a", "page_ids": classes["spods/a"]["page_ids"], "verdict": "maybe"}
        _changes, problems = mods["audit"].apply_membership(pages, classes, [row])
        assert "must be 'ok' or comma-separated indices" in problems[0]


# -------------------------------------------------------------------- tiers


class TestTiers:
    def _pages(self, mods, n_distractors=500):
        pages = [
            _page(mods, f"spods/{i:03d}", "spods", [("logo", (0, 0, 200, 120), "spods/a", "gt")]) for i in range(20)
        ]
        pages += [_page(mods, f"ucsf/d{i:05d}", "ucsf") for i in range(n_distractors)]
        return pages

    def _assign(self, mods, pages, tiers, pinned=None):
        inv = mods["build"].class_inventory(pages)
        admitted, _ = mods["build"].admit_classes(pages, inv, min_instances=10, min_mark_px=32)
        return mods["build"].assign_tiers(
            pages, admitted, tiers=tiers, tier_order=("s", "m", "l"), salt="test-salt", pinned_cutoffs=pinned
        )

    @staticmethod
    def _members(tier_of, tier_order=("s", "m", "l")):
        """Cumulative membership per tier, since tiers nest."""
        out, running = {}, set()
        for tier in tier_order:
            running = running | {p for p, t in tier_of.items() if t == tier}
            out[tier] = set(running)
        return out

    def test_tiers_are_nested_and_hit_their_budgets(self, mods):
        tier_of, _ = self._assign(mods, self._pages(mods), {"s": 60, "m": 200, "l": 400})
        m = self._members(tier_of)
        assert m["s"] < m["m"] < m["l"]
        assert (len(m["s"]), len(m["m"]), len(m["l"])) == (60, 200, 400)

    def test_every_positive_page_is_in_the_smallest_tier(self, mods):
        tier_of, _ = self._assign(mods, self._pages(mods), {"s": 60, "m": 200, "l": 400})
        assert all(tier_of[f"spods/{i:03d}"] == "s" for i in range(20))

    def test_is_deterministic_for_a_fixed_page_set(self, mods):
        budgets = {"s": 60, "m": 200, "l": 400}
        first, _ = self._assign(mods, self._pages(mods, 500), budgets)
        second, _ = self._assign(mods, self._pages(mods, 500), budgets)
        assert first == second

    def test_pinned_cutoffs_survive_a_growing_source_pool(self, mods):
        # Budgets and cross-build stability genuinely conflict: you cannot hold
        # a page count fixed *and* hold membership fixed when the pool changes
        # size. Pinning the rank cutoffs buys stability and lets the count
        # drift, which is the trade a follow-up build wants so that its numbers
        # stay comparable to the earlier one's.
        budgets = {"s": 60, "m": 200, "l": 400}
        _, cutoffs = self._assign(mods, self._pages(mods, 500), budgets)
        small, _ = self._assign(mods, self._pages(mods, 500), budgets, pinned=cutoffs)
        grown, _ = self._assign(mods, self._pages(mods, 900), budgets, pinned=cutoffs)

        s_small = self._members(small)["s"]
        s_grown = self._members(grown)["s"]
        assert s_small <= s_grown, "pinning must never evict a page from a tier it was already in"

    def test_unpinned_growth_is_documented_as_a_new_corpus_version(self, mods):
        # The converse of the test above, pinned here so nobody "fixes" the
        # default into silent instability without noticing: without pinning, a
        # larger pool re-selects, and that is a new corpus, not the same one.
        budgets = {"s": 60, "m": 200, "l": 400}
        before, _ = self._assign(mods, self._pages(mods, 500), budgets)
        after, _ = self._assign(mods, self._pages(mods, 900), budgets)
        assert self._members(before)["s"] != self._members(after)["s"]

    def test_pages_past_the_largest_budget_are_excluded(self, mods):
        pages = self._pages(mods, 500)
        tier_of, _ = self._assign(mods, pages, {"s": 60, "m": 100, "l": 150})
        assert len(tier_of) == 150
        assert len(pages) > len(tier_of)


# --------------------------------------------------------------- clustering


class TestClustering:
    def test_phash_is_deterministic_and_sized_by_the_block(self, mods):
        from PIL import Image

        rng = np.random.default_rng(42)
        img = Image.fromarray(rng.integers(0, 255, (120, 90), dtype=np.uint8))
        a, b = mods["cluster"].phash(img), mods["cluster"].phash(img)
        assert a.shape == (mods["cluster"].PHASH_BLOCK ** 2,)
        assert np.array_equal(a, b)

    def test_the_hash_is_wide_enough_to_separate_two_ringed_stamps(self, mods):
        # 64 bits could not: a book stamp and an elephant stamp, both circular
        # with a heavy border, merged into one class of 32 and no threshold
        # split them. A stamp's ring is low-frequency and its interior is not,
        # so an 8x8 block encodes "is a round stamp" rather than which one.
        assert mods["cluster"].PHASH_BLOCK >= 16

    def test_the_radial_taper_damps_the_border_not_the_middle(self, mods):
        arr = np.full((64, 64), 10.0)
        arr[2, 2] = 250.0  # corner ink, where a border ring lives
        arr[32, 32] = 250.0  # centre ink, where the mark's identity lives
        out = mods["cluster"]._radial_taper(arr)
        centre_kept = (out[32, 32] - arr.mean()) / (250.0 - arr.mean())
        corner_kept = (out[2, 2] - arr.mean()) / (250.0 - arr.mean())
        assert centre_kept > 0.99
        assert corner_kept < 0.01

    def test_the_taper_leaves_a_flat_crop_flat(self, mods):
        # Fading toward the crop's own mean, not toward white: fading to white
        # would replace the border ring with a different strong edge.
        arr = np.full((32, 32), 7.5)
        assert np.allclose(mods["cluster"]._radial_taper(arr), 7.5)

    def test_phash_is_scale_invariant(self, mods):
        from PIL import Image

        rng = np.random.default_rng(7)
        base = Image.fromarray(rng.integers(0, 255, (200, 200), dtype=np.uint8)).filter(
            __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(3)
        )
        big, small = mods["cluster"].phash(base), mods["cluster"].phash(base.resize((80, 80)))
        assert (big != small).mean() < 0.15

    def test_single_linkage_chains_and_separates(self, mods):
        dist = np.array(
            [
                [0.0, 0.1, 0.9, 0.9],
                [0.1, 0.0, 0.15, 0.9],
                [0.9, 0.15, 0.0, 0.9],
                [0.9, 0.9, 0.9, 0.0],
            ]
        )
        labels = mods["cluster"].single_linkage(dist, threshold=0.2)
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] != labels[0]

    def test_labels_are_dense_and_first_appearance_ordered(self, mods):
        dist = np.ones((4, 4)) - np.eye(4)
        labels = mods["cluster"].single_linkage(dist, threshold=0.0)
        assert labels == [0, 1, 2, 3]

    def test_aspect_gate_forces_apart_marks_of_different_shape(self, mods):
        MarkRef = mods["cluster"].MarkRef
        refs = [
            MarkRef(0, 0, "p/1", "logo", (0, 0, 100, 100)),  # square
            MarkRef(1, 0, "p/2", "logo", (0, 0, 400, 40)),  # wide banner
        ]
        desc = np.ones((2, 64), dtype=bool)  # identical hashes
        dist = mods["cluster"].distance_matrix(desc, refs, backend="phash")
        assert dist[0, 1] == 1.0

    def test_cannot_link_keeps_adjudicated_marks_apart(self, mods):
        # Two crops close enough that the threshold would merge them, which a
        # human has said are different marks. The whole point of recording that
        # is that it survives the clustering that would otherwise overrule it.
        dist = np.array([[0.0, 0.05], [0.05, 0.0]])
        assert mods["cluster"].single_linkage(dist, 0.2) == [0, 0]
        assert mods["cluster"].single_linkage(dist, 0.2, cannot_link=[(0, 1)]) == [0, 1]

    def test_a_separation_propagates_through_a_third_crop(self, mods):
        # a-b are separated; c is near both. Without propagation c would merge
        # with a and then with b, reuniting the pair through the back door.
        dist = np.array(
            [
                [0.0, 0.9, 0.05],
                [0.9, 0.0, 0.05],
                [0.05, 0.05, 0.0],
            ]
        )
        labels = mods["cluster"].single_linkage(dist, 0.2, cannot_link=[(0, 1)])
        assert labels[0] != labels[1]

    def test_pairs_resolve_by_page_id_not_row_index(self, mods):
        MarkRef = mods["cluster"].MarkRef
        refs = [
            MarkRef(0, 0, "spods/001", "logo", (0, 0, 10, 10)),
            MarkRef(1, 0, "spods/002", "logo", (0, 0, 10, 10)),
        ]
        assert mods["cluster"].resolve_pairs(refs, [("spods/001", "spods/002")]) == [(0, 1)]

    def test_a_pair_naming_a_dropped_page_is_skipped(self, mods):
        MarkRef = mods["cluster"].MarkRef
        refs = [MarkRef(0, 0, "spods/001", "logo", (0, 0, 10, 10))]
        # Pages come and go with tier budgets; a stale pair must not refuse the
        # build.
        assert mods["cluster"].resolve_pairs(refs, [("spods/001", "spods/999")]) == []

    def test_adjudications_round_trip_and_deduplicate(self, mods, tmp_path):
        path = tmp_path / "adjudications.json"
        mods["cluster"].save_adjudications(
            [{"left_page_id": "d", "right_page_id": "c"}],
            [{"left_page_id": "b", "right_page_id": "a"}, {"left_page_id": "a", "right_page_id": "b"}],
            path,
        )
        same, different = mods["cluster"].load_adjudications(path)
        # (a, b) and (b, a) are one decision, not two.
        assert same == [("c", "d")]
        assert different == [("a", "b")]

    def test_no_adjudication_file_means_no_constraints(self, mods, tmp_path):
        assert mods["cluster"].load_adjudications(tmp_path / "missing.json") == ([], [])

    def test_a_pair_ruled_both_ways_is_refused(self, mods, tmp_path):
        # Storing both would let whichever is applied last silently win, and the
        # loser is a human decision nobody would know had been discarded.
        with pytest.raises(ValueError, match="both same and different"):
            mods["cluster"].save_adjudications(
                [{"left_page_id": "a", "right_page_id": "b"}],
                [{"left_page_id": "a", "right_page_id": "b"}],
                tmp_path / "adjudications.json",
            )

    def test_must_link_joins_marks_the_threshold_would_split(self, mods):
        # The operating strategy: run strict so the partition over-splits, then
        # repair by hand. A merge has to beat the distance, or the repair does
        # not survive the next re-cluster.
        dist = np.array([[0.0, 0.9], [0.9, 0.0]])
        assert mods["cluster"].single_linkage(dist, 0.1) == [0, 1]
        assert mods["cluster"].single_linkage(dist, 0.1, must_link=[(0, 1)]) == [0, 0]

    def test_a_merge_and_a_separation_that_conflict_are_refused(self, mods):
        dist = np.array([[0.0, 0.9], [0.9, 0.0]])
        with pytest.raises(ValueError, match="both same and different"):
            mods["cluster"].single_linkage(dist, 0.1, must_link=[(0, 1)], cannot_link=[(0, 1)])

    def test_a_merge_carries_its_group_across_a_separation(self, mods):
        # a must-link b; c is separated from a. c must therefore stay apart from
        # b too, or the separation is honoured only against the row that
        # happened to be named in it.
        dist = np.array([[0.0, 0.9, 0.01], [0.9, 0.0, 0.01], [0.01, 0.01, 0.0]])
        labels = mods["cluster"].single_linkage(dist, 0.1, must_link=[(0, 1)], cannot_link=[(0, 2)])
        assert labels[0] == labels[1]
        assert labels[2] != labels[0]

    def test_merge_order_is_independent_of_row_order(self, mods):
        # Once constraints can block a merge, "which merge happened first"
        # decides the outcome, so merges are applied in distance order rather
        # than whatever order the loops produce.
        dist = np.array(
            [
                [0.0, 0.10, 0.15],
                [0.10, 0.0, 0.05],
                [0.15, 0.05, 0.0],
            ]
        )
        assert mods["cluster"].single_linkage(dist, 0.12, cannot_link=[(0, 2)]) == [0, 1, 1]

    def test_class_ids_are_anchored_to_a_page_not_a_counter(self, mods):
        MarkRef = mods["cluster"].MarkRef
        pages = [_page(mods, "spods/00042", "spods", [("logo", (0, 0, 10, 10), None, "gt")])]
        refs = [MarkRef(0, 0, "spods/00042", "logo", (0, 0, 10, 10))]
        classes = mods["cluster"].assign_class_ids(pages, refs, [0], source="spods")
        assert list(classes) == ["spods/logo_00042_0"]
        assert pages[0].marks[0].provenance == "clustered"


# ---------------------------------------------------------------- synthesis


class TestSynthesis:
    def _artwork(self, size=(120, 60)):
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([4, 4, size[0] - 5, size[1] - 5], fill=(20, 30, 200, 255))
        return img

    def test_paste_box_is_tight_around_the_rotated_alpha(self, mods):
        from PIL import Image

        page = Image.new("RGBA", (800, 1000), (255, 255, 255, 255))
        box = mods["synth"].paste_mark(page, self._artwork(), target_px=200, rotation_deg=30.0, position=(0.5, 0.5))
        x, y, w, h = box
        # A 30-degree rotation expands the paste rectangle well beyond the mark.
        # The recorded box must follow the ink, not the rectangle.
        assert w < 200 and h < 200
        crop = np.array(page.crop((x, y, x + w, y + h)))
        assert crop[..., 3].max() > 0
        # And it must be tight: every edge row/column touches ink.
        assert crop[0, :, 3].max() > 0 and crop[-1, :, 3].max() > 0

    def test_paste_lands_inside_the_page(self, mods):
        from PIL import Image

        page = Image.new("RGBA", (600, 800), (255, 255, 255, 255))
        for pos in ((0.0, 0.0), (1.0, 1.0), (0.5, 0.5)):
            x, y, w, h = mods["synth"].paste_mark(page, self._artwork(), target_px=120, rotation_deg=0.0, position=pos)
            assert 0 <= x and 0 <= y and x + w <= page.width and y + h <= page.height

    def test_build_synthetic_pages_records_exact_ground_truth(self, mods, tmp_path):
        from PIL import Image

        bg = tmp_path / "bg.png"
        Image.new("RGB", (900, 1200), "white").save(bg)
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        for name in ("alpha", "beta"):
            self._artwork().save(pool_dir / f"{name}.png")
        pool = mods["artwork"].load_pool_dir(pool_dir)

        pages = mods["synth"].build_synthetic_pages(
            [bg], pool, tmp_path / "out", instances_per_class=3, size_px=(64, 128), rotation_deg=(-5, 5), seed=1
        )
        assert len(pages) == 6
        assert {m.class_id for p in pages for m in p.marks} == {"synth/alpha", "synth/beta"}
        assert all(m.provenance == "synthetic" for p in pages for m in p.marks)
        assert all(m.area() > 0 for p in pages for m in p.marks)

    def test_synthesis_is_reproducible_from_the_seed(self, mods, tmp_path):
        from PIL import Image

        bg = tmp_path / "bg.png"
        Image.new("RGB", (900, 1200), "white").save(bg)
        pool_dir = tmp_path / "pool"
        pool_dir.mkdir()
        self._artwork().save(pool_dir / "alpha.png")
        pool = mods["artwork"].load_pool_dir(pool_dir)

        def run(out):
            return [
                (m.class_id, m.box)
                for p in mods["synth"].build_synthetic_pages(
                    [bg], pool, out, instances_per_class=4, size_px=(64, 128), rotation_deg=(-5, 5), seed=99
                )
                for m in p.marks
            ]

        assert run(tmp_path / "a") == run(tmp_path / "b")

    def test_empty_pool_is_an_error_not_an_empty_corpus(self, mods, tmp_path):
        with pytest.raises(ValueError, match="artwork pool"):
            mods["synth"].build_synthetic_pages(
                [tmp_path / "bg.png"],
                {},
                tmp_path,
                instances_per_class=1,
                size_px=(64, 128),
                rotation_deg=(0, 0),
                seed=1,
            )


# ----------------------------------------------------------------- manifest


class TestManifest:
    def test_round_trips_marks_and_meta(self, mods, tmp_path):
        pages = [
            _page(mods, "spods/001", "spods", [("logo", (1, 2, 3, 4), "spods/a", "clustered")]),
            _page(mods, "ucsf/x#0", "ucsf"),
        ]
        pages[1].meta = {"industry": "Opioids", "decade": "1990s"}
        path = tmp_path / "corpus.jsonl"

        assert mods["common"].write_manifest(pages, path) == 2
        back = list(mods["common"].read_manifest(path))
        assert [p.page_id for p in back] == ["spods/001", "ucsf/x#0"]
        assert back[0].marks[0].box == (1, 2, 3, 4)
        assert back[0].marks[0].provenance == "clustered"
        assert back[1].meta["industry"] == "Opioids"

    def test_written_records_are_stable_json(self, mods, tmp_path):
        pages = [_page(mods, "spods/001", "spods", [("logo", (1, 2, 3, 4), "spods/a", "gt")])]
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        mods["common"].write_manifest(pages, a)
        mods["common"].write_manifest(list(mods["common"].read_manifest(a)), b)
        assert a.read_text() == b.read_text()

    def test_manifest_is_valid_jsonl(self, mods, tmp_path):
        path = tmp_path / "c.jsonl"
        mods["common"].write_manifest([_page(mods, "spods/1", "spods")], path)
        for line in path.read_text().splitlines():
            json.loads(line)


# ------------------------------------------------------------------ report


class TestReport:
    def test_scale_section_bands_against_the_measured_floor(self, mods, tmp_path):
        from PIL import Image

        img = tmp_path / "p.png"
        Image.new("RGB", (1000, 1400), "white").save(img)
        # One sub-floor mark and three above it.
        pages = [
            _page(mods, "spods/001", "spods", [("logo", (0, 0, 20, 18), "spods/a", "gt")], path=str(img)),
            _page(mods, "spods/002", "spods", [("logo", (0, 0, 90, 70), "spods/a", "gt")], path=str(img)),
            _page(mods, "spods/003", "spods", [("logo", (0, 0, 200, 150), "spods/a", "gt")], path=str(img)),
            _page(mods, "spods/004", "spods", [("logo", (0, 0, 300, 200), "spods/a", "gt")], path=str(img)),
        ]
        html = mods["report"].section_scale(pages, {})
        assert "32px" in html
        # One of four marks is below the floor; the share must be reported, not
        # buried, because a class built from sub-floor instances measures the
        # floor rather than the method.
        assert "25% of labelled marks fall below" in html

    def test_scale_section_is_empty_without_labelled_marks(self, mods):
        assert mods["report"].section_scale([_page(mods, "u/1", "ucsf")], {}) == ""

    def test_overview_warns_when_nothing_is_verified(self, mods):
        classes = {"spods/a": {"n_instances": 3, "audit": {"membership_verified": False}}}
        html = mods["report"].section_overview([_page(mods, "spods/1", "spods")], classes, {})
        assert "Nothing here is verified yet" in html

    def test_overview_drops_the_warning_once_verified(self, mods):
        classes = {"spods/a": {"n_instances": 3, "audit": {"membership_verified": True}}}
        html = mods["report"].section_overview([_page(mods, "spods/1", "spods")], classes, {})
        assert "Nothing here is verified yet" not in html

    def test_tables_escape_their_content(self, mods):
        html = mods["report"]._table(["c"], [["<script>x</script>"]])
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_every_provenance_the_builder_emits_has_a_gloss(self, mods):
        # A provenance with no explanation in the report is a label the reader
        # cannot interpret, which defeats the point of tracking provenance.
        emitted = {"gt", "clustered", "clustered_band", "candidate", "synthetic"}
        assert emitted <= set(mods["report"]._PROVENANCE_MEANING)


# ------------------------------------------------------------- embed cells


class TestEmbedCells:
    def test_a_tier_cell_is_cumulative_over_smaller_tiers(self, mods):
        assert mods["embed"].tiers_up_to("s") == {"s"}
        assert mods["embed"].tiers_up_to("m") == {"s", "m"}
        assert mods["embed"].tiers_up_to("l") == {"s", "m", "l"}

    def test_cell_names_match_the_pile_convention(self, mods):
        assert mods["embed"].cell_name("m", "sift_vlad") == "docmarks_m__sift_vlad.pkl"

    def test_load_medias_carries_boxes_as_regions(self, mods, tmp_path):
        from PIL import Image

        img = tmp_path / "p.png"
        Image.new("RGB", (400, 500), "white").save(img)
        page = _page(mods, "spods/001", "spods", [("logo", (10, 20, 60, 40), "spods/a", "gt")], path=str(img))
        medias = mods["embed"].load_medias([page], {}, "sift_vlad")
        (media,) = medias.values()
        assert media["categories"] == ["spods/a"]
        assert media["regions"] == [
            {"label": "spods/a", "x": 10, "y": 20, "width": 60, "height": 40, "provenance": "gt"}
        ]
        assert media["origin_name"] == "spods/001"

    def test_weak_boxless_marks_become_a_category_but_never_a_region(self, mods, tmp_path):
        from PIL import Image

        img = tmp_path / "p.png"
        Image.new("RGB", (400, 500), "white").save(img)
        page = _page(mods, "ucsf/x#0", "ucsf", [("logo", (0, 0, 0, 0), "ucsf/letterhead_rjr", "weak")], path=str(img))
        (media,) = mods["embed"].load_medias([page], {}, "siglip").values()
        # A zero-area region would be indistinguishable from a real box once it
        # is in the media dict, which is precisely the distinction the corpus
        # exists to keep visible.
        assert media["regions"] == []
        assert media["categories"] == ["ucsf/letterhead_rjr"]
        assert media["docmarks"]["provenances"] == ["weak"]

    def test_media_ids_are_stable_under_input_ordering(self, mods, tmp_path):
        from PIL import Image

        paths = []
        for name in ("a", "b", "c"):
            p = tmp_path / f"{name}.png"
            Image.new("RGB", (50, 50), "white").save(p)
            paths.append(p)
        pages = [_page(mods, f"spods/{n}", "spods", path=str(p)) for n, p in zip("abc", paths)]

        forward = mods["embed"].load_medias(pages, {}, "siglip")
        reverse = mods["embed"].load_medias(list(reversed(pages)), {}, "siglip")
        assert {i: m["origin_name"] for i, m in forward.items()} == {i: m["origin_name"] for i, m in reverse.items()}


# -------------------------------------------------------------------- probe


class TestKaggleProbe:
    """``--probe`` must stay a metadata call.

    It used to reach Kaggle by downloading the bundle into ``raw/_probe_*``:
    ~2 GB of transfer, fetched a second time by the real build, never reclaimed
    — under a name and a runbook that both promised "seconds" (issue #3356).
    The cheapness is the whole feature, so it is pinned here.
    """

    @staticmethod
    def _with_creds(monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "someone")
        monkeypatch.setenv("KAGGLE_KEY", "deadbeef")

    @staticmethod
    def _run(monkeypatch, mods, *, stdout="", stderr="", returncode=0, exc=None):
        """Stub ``subprocess.run`` and record the argv it was handed."""
        import subprocess

        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(cmd)
            if exc is not None:
                raise exc
            if returncode:
                raise subprocess.CalledProcessError(returncode, cmd, output=stdout, stderr=stderr)
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=stderr)

        monkeypatch.setattr(mods["common"].subprocess, "run", fake_run)
        return seen

    def test_lists_files_instead_of_downloading(self, monkeypatch, mods, tmp_path):
        self._with_creds(monkeypatch)
        monkeypatch.chdir(tmp_path)
        seen = self._run(monkeypatch, mods, stdout="name,size,creationDate\ngt.zip,44MB,2020-01-01\n")

        mods["common"].kaggle_probe("owner/name")

        (cmd,) = seen
        assert cmd[:4] == ["kaggle", "datasets", "files", "-d"]
        assert "download" not in cmd
        # Nothing may be staged anywhere: no destination is even passed.
        assert not any(str(tmp_path) in str(part) for part in cmd)
        assert list(tmp_path.iterdir()) == []

    def test_a_missing_token_is_reported_before_the_cli_runs(self, monkeypatch, mods, tmp_path):
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.kaggle/kaggle.json under here
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        seen = self._run(monkeypatch, mods)

        with pytest.raises(mods["common"].FetchError, match=r"kaggle\.json"):
            mods["common"].kaggle_probe("owner/name")
        assert seen == []

    def test_a_missing_cli_names_the_install(self, monkeypatch, mods):
        self._with_creds(monkeypatch)
        self._run(monkeypatch, mods, exc=FileNotFoundError("kaggle"))

        with pytest.raises(mods["common"].FetchError, match="pip install kaggle"):
            mods["common"].kaggle_probe("owner/name")

    def test_a_nonzero_exit_quotes_stderr(self, monkeypatch, mods):
        self._with_creds(monkeypatch)
        self._run(monkeypatch, mods, returncode=1, stderr="boom")

        with pytest.raises(mods["common"].FetchError, match="boom"):
            mods["common"].kaggle_probe("owner/name")

    @pytest.mark.parametrize(
        "stdout",
        [
            "403 - Forbidden\n",  # the CLI swallows API errors and still exits 0
            "404 - Not Found\n",
            "",
            "name,size,creationDate\n",  # a header with no rows is not a dataset
        ],
    )
    def test_exit_zero_is_not_taken_as_success(self, monkeypatch, mods, stdout):
        self._with_creds(monkeypatch)
        self._run(monkeypatch, mods, stdout=stdout)

        with pytest.raises(mods["common"].FetchError, match="owner/name"):
            mods["common"].kaggle_probe("owner/name")


class TestReclaimProbeDirs:
    def test_removes_stale_probe_dirs_and_reports_their_size(self, mods, tmp_path):
        stale = tmp_path / "_probe_staver" / "nested"
        stale.mkdir(parents=True)
        (stale / "big.bin").write_bytes(b"x" * 2048)
        keep = tmp_path / "staver"
        keep.mkdir()
        (keep / "real.bin").write_bytes(b"y" * 16)

        dirs, freed = mods["build"]._reclaim_probe_dirs(tmp_path)

        assert (dirs, freed) == (1, 2048)
        assert not (tmp_path / "_probe_staver").exists()
        assert (keep / "real.bin").exists()

    def test_a_missing_raw_root_is_not_an_error(self, mods, tmp_path):
        assert mods["build"]._reclaim_probe_dirs(tmp_path / "never-created") == (0, 0)
