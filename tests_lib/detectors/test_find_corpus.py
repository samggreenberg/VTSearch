"""#4960: Find on a second corpus at every click, with the two domain-shift flags beside the objective.

Eval only: ``find_corpus`` scores a second corpus with the detector each click's
Test gives and changes nothing the session does.  What a study rests on, pinned
here on synthetic medias:

* the in-corpus columns, the picks and the session are byte-identical with and
  without the knob (the way ``test_band_auroc_moves_no_other_column`` pins
  #4044's), so no published number moves under it;
* the Find columns pair click for click, and on the withheld half alone
  (``find_mix=0``) they are the in-corpus columns;
* a corpus with no positive scores NaN F-beta and still counts what it returns;
* a mixed corpus holds exactly the withheld half plus the share;
* the flags are read at the snapshot clicks only, and the training-domain
  overlap is NaN on a patch embedder, which the app refuses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.find_corpus import (
    FIND_ONLY,
    blank_find_columns,
    build_find_corpus,
    find_cell,
    parse_find_mix,
)
from vtscore.eval.voting_columns import FIND_CORPUS_COLUMNS, FIND_FLAG_COLUMNS, TIMING_COLUMNS
from vtscore.eval.voting_iterations import simulate_voting_iterations

_DIM = 16
_FIND = frozenset(FIND_CORPUS_COLUMNS) | frozenset(FIND_FLAG_COLUMNS)


def _medias(n_pos: int, n_neg: int, *, seed: int, shift: float = 0.0, first_id: int = 1, cat: str = "alpha"):
    """Overlapping classes in one space: *cat* around +0.5, the rest around -0.5, all moved by *shift*."""
    rng = np.random.RandomState(seed)
    out: dict[int, dict] = {}
    for i in range(n_pos + n_neg):
        pos = i < n_pos
        emb = (rng.normal(0.5 if pos else -0.5, 1.0, _DIM) + shift).astype(np.float32)
        out[first_id + i] = {"id": first_id + i, "embeddings": {"emb": emb}, "category": cat if pos else "beta"}
    return out


def _cell():
    return _medias(30, 90, seed=0)


def _foreign(n_pos: int = 8, n_neg: int = 52):
    # Ids that collide with the cell's: the corpus must re-key them.
    return _medias(n_pos, n_neg, seed=5, shift=0.7)


_QUERY = np.full(_DIM, 1.0 / math.sqrt(_DIM), dtype=np.float32)


def _cosines(medias):
    return {
        cid: float(np.dot(m["embeddings"]["emb"], _QUERY) / np.linalg.norm(m["embeddings"]["emb"]))
        for cid, m in medias.items()
    }


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def _run(**kw):
    kw.setdefault("max_steps", 20)
    return simulate_voting_iterations(_cell(), "alpha", seed=3, calibrate_count=2, beta=1.0, **kw)


_PATHS = {
    "plain": {},
    "calibration": {"style": "whole_image", "emit_calibration_metrics": True},
}


@pytest.fixture(scope="module", params=sorted(_PATHS))
def paired(request):
    """The same session with and without a Find corpus, and both pick logs."""
    kw = _PATHS[request.param]
    picks_off: list[dict] = []
    picks_on: list[dict] = []
    off = _run(pick_sink=picks_off, **kw)
    on = _run(pick_sink=picks_on, find_corpus=_foreign(), find_flag_steps=[6, 12], **kw)
    return off, on, picks_off, picks_on


class TestTheMix:
    @pytest.mark.parametrize(
        ("given", "want"),
        [(None, FIND_ONLY), ("only", FIND_ONLY), ("ONLY", FIND_ONLY), ("equal", 1.0), ("0", 0.0), (0, 0.0), (0.5, 0.5)],
    )
    def test_spellings(self, given, want):
        assert parse_find_mix(given) == want

    @pytest.mark.parametrize("bad", ["lots", -1, "-0.5", float("inf"), float("nan"), True])
    def test_refuses_what_is_not_a_share(self, bad):
        with pytest.raises(ValueError, match="find_mix"):
            parse_find_mix(bad)


class TestTheCorpus:
    """build_find_corpus on its own: what the mix draws, and under which ids."""

    @staticmethod
    def _build(mix, foreign=None, seed=1, **kw):
        cell = _cell()
        test_ids = sorted(cell)[::2]
        return (
            cell,
            test_ids,
            build_find_corpus(
                cell,
                test_ids,
                _foreign() if foreign is None else foreign,
                "alpha",
                mix=parse_find_mix(mix),
                seed=seed,
                region_aware=False,
                **kw,
            ),
        )

    def test_share_zero_is_the_withheld_half(self):
        _, test_ids, corpus = self._build("0")
        assert corpus.ids == test_ids
        assert (corpus.n_withheld, corpus.n_foreign) == (len(test_ids), 0)

    @pytest.mark.parametrize(("mix", "share"), [("equal", 1.0), (0.5, 0.5), (0.25, 0.25)])
    def test_a_share_holds_exactly_the_withheld_half_plus_the_share(self, mix, share):
        _, test_ids, corpus = self._build(mix)
        want = min(60, int(round(share * len(test_ids))))
        assert corpus.ids[: len(test_ids)] == test_ids
        assert (corpus.n_withheld, corpus.n_foreign) == (len(test_ids), want)
        assert len(corpus.ids) == len(set(corpus.ids)) == len(test_ids) + want

    def test_a_share_past_the_foreign_cell_takes_all_of_it(self):
        _, test_ids, corpus = self._build(10.0)
        assert corpus.n_foreign == 60

    def test_only_is_the_foreign_cell_alone(self):
        _, _, corpus = self._build("only")
        assert (corpus.n_withheld, corpus.n_foreign) == (0, 60)

    def test_the_draw_is_seeded(self):
        _, _, a = self._build(0.5, seed=1)
        _, _, b = self._build(0.5, seed=1)
        _, _, c = self._build(0.5, seed=2)
        emb = lambda corpus: [corpus.clips[cid]["embeddings"]["emb"].tobytes() for cid in corpus.ids]  # noqa: E731
        assert emb(a) == emb(b)
        assert emb(a) != emb(c)

    def test_foreign_medias_are_rekeyed_past_the_cell(self):
        cell, test_ids, corpus = self._build("only")
        foreign_ids = corpus.ids
        assert min(foreign_ids) > max(cell)
        assert all(corpus.clips[cid]["id"] == cid for cid in foreign_ids)
        # The cell's own medias are untouched beside them.
        assert all(corpus.clips[cid] is cell[cid] for cid in cell)

    def test_a_dataset_with_no_such_cell_is_the_whole_dataset_with_no_positive(self):
        banded = {
            1: {"id": 1, "categories": ["car@small"], "evaluable_categories": ["car@small"]},
            2: {"id": 2, "categories": [], "evaluable_categories": ["car@small"]},
        }
        assert find_cell(banded, "bus@medium") == banded
        assert find_cell(banded, "car@small") == banded

    def test_a_designated_cell_drops_the_wrong_size(self):
        banded = {
            1: {"id": 1, "categories": ["car@small"], "evaluable_categories": ["car@small"]},
            2: {"id": 2, "categories": ["car@large"], "evaluable_categories": ["car@large"]},
            3: {"id": 3, "categories": [], "evaluable_categories": ["car@small", "car@large"]},
        }
        assert sorted(find_cell(banded, "car@small")) == [1, 3]

    def test_another_embedder_is_refused(self):
        wide = {cid: {**m, "embeddings": {"emb": np.zeros(_DIM + 1, np.float32)}} for cid, m in _foreign().items()}
        with pytest.raises(ValueError, match="embedder"):
            self._build("only", foreign=wide)

    def test_patch_grids_on_one_side_are_refused(self):
        grid = {cid: {**m, "patch_grid": np.zeros((1, 1, _DIM), np.float16)} for cid, m in _foreign().items()}
        with pytest.raises(ValueError, match="patch grids"):
            self._build("only", foreign=grid)

    def test_a_typed_query_run_needs_the_querys_scores_over_foreign_medias(self):
        cell = _cell()
        with pytest.raises(ValueError, match="find_corpus_scores"):
            self._build("equal", query_scores=_cosines(cell))
        # The withheld half alone needs none: the cell's own sort covers it.
        _, test_ids, corpus = self._build("0", query_scores=_cosines(cell))
        assert corpus.query_scores is not None and sorted(corpus.query_scores) == sorted(test_ids)

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(ValueError, match="no media"):
            self._build("only", foreign={})


class TestTheSessionDoesNotMove:
    def test_the_in_corpus_columns_are_byte_identical(self, paired):
        off, on, _, _ = paired
        assert [r["t"] for r in off] == [r["t"] for r in on]
        for a, b in zip(off, on, strict=True):
            assert set(a) == set(b)
            moved = [k for k in a if k not in _FIND and k not in TIMING_COLUMNS and not _same(a[k], b[k])]
            assert not moved, moved

    def test_the_picks_are_identical(self, paired):
        _, _, picks_off, picks_on = paired
        assert picks_off and len(picks_off) == len(picks_on)
        for a, b in zip(picks_off, picks_on, strict=True):
            assert set(a) == set(b) and all(_same(a[k], b[k]) for k in a)

    def test_off_the_columns_are_nan(self, paired):
        off, _, _, _ = paired
        assert all(all(math.isnan(r[c]) for c in _FIND) for r in off)

    def test_the_door_refuses_the_find_knobs_without_a_corpus(self):
        with pytest.raises(ValueError, match="find_corpus"):
            _run(find_flag_steps=[5])
        with pytest.raises(ValueError, match="find_corpus"):
            _run(find_mix="equal")


class TestTheFindRows:
    def test_every_click_carries_the_find_columns(self, paired):
        _, on, _, _ = paired
        assert on
        for r in on:
            assert set(FIND_CORPUS_COLUMNS) <= set(r)
            assert r["find_n_pos"] == 8 and r["find_n_neg"] == 52
            assert 0 <= r["find_n_flagged"] <= 60
            assert 0.0 <= r["find_fbeta"] <= 1.0

    def test_the_withheld_half_alone_is_the_in_corpus_read(self):
        """``find_mix=0``: the same detector, the same corpus, the same line - so the same numbers."""
        rows = _run(find_corpus=_foreign(), find_mix=0)
        pairs = {
            "find_fbeta": "fbeta",
            "find_fbeta_b025": "fbeta_b025",
            "find_fbeta_b4": "fbeta_b4",
            "find_precision": "precision",
            "find_recall": "recall",
            "find_n_flagged": "n_flagged",
            "find_average_precision": "average_precision",
            "find_n_pos": "n_test_pos",
            "find_n_neg": "n_test_neg",
        }
        tiers = {r["detector_tier"] for r in rows}
        assert {"centroid", "trained"} <= tiers
        for r in rows:
            for f, c in pairs.items():
                assert _same(r[f], r[c]), (r["t"], r["detector_tier"], f)

    def test_a_corpus_with_no_positive_scores_nan_and_counts_what_it_returns(self):
        rows = _run(find_corpus=_foreign(n_pos=0, n_neg=60))
        assert rows
        for r in rows:
            assert all(math.isnan(r[f"find_{c}"]) for c in ("fbeta", "fbeta_b025", "fbeta_b1", "fbeta_b4"))
            assert math.isnan(r["find_recall"]) and math.isnan(r["find_average_precision"])
            assert (r["find_n_pos"], r["find_n_neg"]) == (0, 60)
            assert 0 <= r["find_n_flagged"] <= 60

    def test_a_mixed_corpus_is_the_withheld_half_plus_the_share(self):
        for row in _run(find_corpus=_foreign(), find_mix="equal"):
            n_test = row["n_test_pos"] + row["n_test_neg"]
            assert row["find_n_pos"] + row["find_n_neg"] == n_test + min(60, n_test)

    def test_a_typed_query_run_finds_with_the_typed_query_until_the_head(self):
        foreign = _foreign()
        rows = _run(
            seed_scores=_cosines(_cell()),
            find_corpus=foreign,
            find_corpus_scores=_cosines(foreign),
        )
        query = [r for r in rows if r["detector_tier"] == "query"]
        assert query
        # The typed query's set does not depend on the votes: one read at every such click.
        assert len({(r["find_n_flagged"], r["find_fbeta"]) for r in query}) == 1
        with pytest.raises(ValueError, match="find_corpus_scores"):
            _run(seed_scores=_cosines(_cell()), find_corpus=foreign)


class TestTheFlags:
    def test_read_at_the_first_click_at_or_past_each_listed_one(self, paired):
        _, on, _, _ = paired
        clicks = [r for r in on if r["phase"] != "check"]
        for listed in (6, 12):
            first = min(r["t"] for r in clicks if r["t"] >= listed)
            at = [r for r in on if r["flags_at"] == listed]
            assert at and {r["t"] for r in at} == {first}
            for r in at:
                assert all(not math.isnan(r[c]) for c in FIND_FLAG_COLUMNS), r["t"]
                assert r["shift_shifted"] in (0.0, 1.0) and r["evidence_unsupported"] in (0.0, 1.0)
                assert 0.0 <= r["shift_frac_atypical"] <= 1.0
                assert 0.0 <= r["evidence_frac_unsupported"] <= 1.0
        rest = [r for r in on if math.isnan(r["flags_at"])]
        assert rest and all(all(math.isnan(r[c]) for c in FIND_FLAG_COLUMNS) for r in rest)

    def test_the_blank_row_names_every_column(self):
        blank = blank_find_columns()
        assert set(blank) == _FIND and all(math.isnan(v) for v in blank.values())


_GRID, _PDIM = 2, 8


def _patch_medias(n_pos: int, n_neg: int, *, seed: int, first_id: int = 1, shift: float = 0.0):
    """A patch embedder's medias: a 2x2 grid each, the image vector their mean."""
    rng = np.random.default_rng(seed)
    out: dict[int, dict] = {}
    for i in range(n_pos + n_neg):
        pos = i < n_pos
        grid = (rng.normal(0.6 if pos else -0.6, 1.0, (_GRID, _GRID, _PDIM)) + shift).astype(np.float32)
        cid = first_id + i
        media = {
            "id": cid,
            "media_type": "image",
            "embedder": "dinov3_patch",
            "embeddings": {"dinov3_patch": grid.reshape(-1, _PDIM).mean(axis=0)},
            "patch_grid": grid,
            "category": "alpha" if pos else "beta",
        }
        if pos:
            media["regions"] = [{"box": [0.0, 0.0, 0.5, 0.5], "label": "alpha"}]
        out[cid] = media
    return out


class TestAPatchEmbedder:
    def test_the_overlap_is_nan_and_the_evidence_is_read(self):
        """The app refuses the training-domain overlap on a patch embedder (#3329); the evidence chip it runs."""
        cell = _patch_medias(16, 44, seed=0)
        foreign = _patch_medias(4, 26, seed=9, shift=0.5)
        kw = {"seed": 3, "calibrate_count": 2, "beta": 1.0, "max_steps": 14, "region_voting": True}
        off = simulate_voting_iterations(cell, "alpha", **kw)
        on = simulate_voting_iterations(cell, "alpha", find_corpus=foreign, find_flag_steps=[8], **kw)
        assert on and all(r["style"] == "max_patch" for r in on)
        for a, b in zip(off, on, strict=True):
            assert not [k for k in a if k not in _FIND and k not in TIMING_COLUMNS and not _same(a[k], b[k])]
        at = [r for r in on if r["flags_at"] == 8]
        assert at
        for r in at:
            assert all(math.isnan(r[c]) for c in ("shift_frac_atypical", "shift_z", "shift_shifted"))
            assert not math.isnan(r["evidence_frac_unsupported"]) and not math.isnan(r["evidence_z"])
        assert all(r["find_n_pos"] == 4 and r["find_n_neg"] == 26 for r in on)
