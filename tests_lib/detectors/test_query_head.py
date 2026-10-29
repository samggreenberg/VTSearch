"""A typed-query detector gives its query's own set until it has a trained head (#4758), in the library tier.

Under the label quota (and before the first Good), a detector made from a typed
query is the text sort as a head: the query's unit vector, cut where the text
sort draws its display line at the detector's balance, scored in the dataset's
text space.  These tests pin the head against the sort it is, the tiers
``train_from_labelset`` and ``labelset_train_and_score`` give with and without a
query, the cache, the balance redraw, and the fallback where there is no text
slot.
"""

from __future__ import annotations

import numpy as np
import pytest

import vtscore.detectors.labelset_training as lt
import vtscore.detectors.query_head as qh
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.centroid_head import CENTROID_THRESHOLD, is_centroid_head
from vtscore.detectors.label_quota import TIER_QUERY, served_quota
from vtscore.detectors.labelset_elements import stable_element_id
from vtscore.detectors.model_loading import cached_head_is_current, resolve_or_train_detector
from vtscore.detectors.query_head import (
    detector_text_query,
    fit_query_head,
    head_score_embedder,
    is_query_head,
    is_quota_head,
    query_head,
    query_head_embedder,
    query_head_text,
)
from vtscore.detectors.training import score_rows_with_model, scoring_rows_for_snap
from vtscore.state.core import DetectorContext
from vtscore.training.thresholds import text_sort_cuts

DIM = 8
EMBEDDER = "test_embedder"
QUERY = "a planted thing"


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def _corpus(n: int = 120, seed: int = 4758) -> tuple[dict[int, dict], np.ndarray]:
    """A plain single-vector corpus: a tight cluster around the query's direction and a broad cloud."""
    rng = np.random.default_rng(seed)
    proto = _unit(rng.standard_normal(DIM))
    snap: dict[int, dict] = {}
    for cid in range(1, n + 1):
        vec = proto + (0.2 if cid <= 8 else 1.5) * rng.standard_normal(DIM)
        snap[cid] = {"id": cid, "media_type": "image", "embedder": EMBEDDER, "embeddings": {EMBEDDER: _unit(vec)}}
    return snap, proto


@pytest.fixture
def world(monkeypatch):
    """A corpus whose text slot is the test embedder, a query that embeds to the cluster, and labels that resolve."""
    snap, proto = _corpus()
    embedded: list[tuple[str, str]] = []

    def _embed(text, media_type, embedder):
        embedded.append((text, embedder))
        return proto if text == QUERY else None

    monkeypatch.setattr(qh, "text_embedder_for_snap", lambda s: EMBEDDER if s else None)
    monkeypatch.setattr(qh, "embed_detector_query", _embed)

    def _labelset(goods: list[int], bads: list[int]) -> LabelSet:
        return LabelSet(
            [LabeledElement(md5=f"m{c}", label="good", origin_name=f"m{c}") for c in goods]
            + [LabeledElement(md5=f"m{c}", label="bad", origin_name=f"m{c}") for c in bads]
        )

    def _fake_populate(det_ctx, labelset, *, media_type, snap, on_progress=None):
        det_ctx.embedder = EMBEDDER
        for elem in labelset.elements:
            det_ctx.label_embeddings[stable_element_id(elem)] = _snap[int(elem.md5[1:])]["embeddings"][EMBEDDER]

    _snap = snap
    monkeypatch.setattr(lt, "populate_label_embeddings", _fake_populate)
    return snap, proto, _labelset, embedded


def _ctx(beta: float | None = 1.0) -> DetectorContext:
    ctx = DetectorContext("", name="query", media_type="image")
    ctx.beta = beta
    return ctx


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------


class TestWhichQuery:
    def test_the_detectors_text_query_else_its_first_text_example(self):
        assert detector_text_query({"text_query": "  a dog "}) == "a dog"
        assert detector_text_query({"text_query": "", "examples": [{"type": "media", "value": "x"}, {"type": "text", "value": "a cat"}]}) == "a cat"  # fmt: skip
        assert detector_text_query({"text_query": ""}) == ""
        assert detector_text_query(None) == ""


class TestTheHead:
    def test_it_is_its_own_kind_of_quota_head(self):
        head = query_head(_unit(np.arange(1, DIM + 1, dtype=np.float32)), 0.3, embedder="siglip", text="a dog")
        assert is_query_head(head) and is_quota_head(head) and not is_centroid_head(head)
        assert (query_head_embedder(head), query_head_text(head)) == ("siglip", "a dog")
        assert head_score_embedder(head, "dinov3_patch") == "siglip"
        assert set(head.state_dict()) == {"0.weight", "0.bias"}
        from vtscore.training.mlp import LINEAR_SVM_HEAD, build_model

        trained = build_model(DIM, hidden_dim=LINEAR_SVM_HEAD)
        assert not is_quota_head(trained) and head_score_embedder(trained, "dinov3_patch") == "dinov3_patch"

    @pytest.mark.parametrize("beta", [0.25, 1.0, 4.0])
    def test_it_keeps_the_text_sorts_own_set_at_the_balance(self, beta):
        """The planted answer: cosine to the query, cut where ``text_sort_cuts`` draws the display line."""
        snap, proto = _corpus()
        rows = scoring_rows_for_snap(snap, EMBEDDER)
        head, threshold, line = fit_query_head(
            proto, lambda h: score_rows_with_model(h, rows)[0], beta=beta, embedder=EMBEDDER, text=QUERY
        )
        assert threshold == CENTROID_THRESHOLD and line.rule == "text"
        cosines = rows.matrix.astype(np.float64) @ proto.astype(np.float64)
        drawn = text_sort_cuts(list(cosines), beta=beta).threshold
        scores = np.asarray(score_rows_with_model(head, rows)[0])
        kept = scores >= threshold
        assert np.array_equal(kept, cosines >= drawn - 1e-6) or np.array_equal(kept, cosines >= drawn + 1e-6)
        assert list(np.argsort(-scores, kind="stable")) == list(np.argsort(-cosines, kind="stable"))
        # The count line keeps the cluster; above beta 1 the guarded line may keep nothing on so small a corpus.
        assert int(kept.sum()) < len(snap) and (beta > 1 or int(kept.sum()) > 0)

    def test_a_balance_change_redraws_the_text_line(self):
        snap, proto = _corpus()
        rows = scoring_rows_for_snap(snap, EMBEDDER)
        head, _threshold, line = fit_query_head(
            proto, lambda h: score_rows_with_model(h, rows)[0], beta=1.0, embedder=EMBEDDER, text=QUERY
        )
        scores = np.asarray(score_rows_with_model(head, rows)[0])
        cosines = rows.matrix.astype(np.float64) @ proto.astype(np.float64)
        for beta in (0.25, 4.0):
            keep = int((scores >= line.threshold_at(beta)).sum())
            assert keep == int((cosines >= text_sort_cuts(list(cosines), beta=beta).threshold - 1e-9).sum())


# ---------------------------------------------------------------------------
# The tiers, with and without a typed query
# ---------------------------------------------------------------------------


class TestTrainFromLabelset:
    @pytest.mark.parametrize(("goods", "bads"), [([], []), ([], [40, 41]), ([1], []), ([1, 2], [40, 41, 42, 43, 44])])
    def test_until_the_quota_a_typed_query_detector_is_its_query(self, world, goods, bads):
        snap, _proto, make, embedded = world
        ctx = _ctx()
        ls = make(goods, bads)
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap, text_query=QUERY) is True
        assert is_query_head(ctx.model) and query_head_text(ctx.model) == QUERY
        assert embedded == [(QUERY, EMBEDDER)]
        assert ctx.calibration_cache is None and ctx.labels_line is None and ctx.centroid_line.rule == "text"
        assert served_quota(ctx.model, ls)["tier"] == TIER_QUERY
        assert cached_head_is_current(ctx, ls, QUERY)
        assert not cached_head_is_current(ctx, ls, "another query"), "a head built from an older query"

    def test_at_the_quota_the_trained_head_takes_over(self, world):
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        ls = make([1, 2, 3], [40, 41, 42, 43])
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap, text_query=QUERY) is True
        assert ctx.model is not None and not is_quota_head(ctx.model)

    def test_without_a_query_it_is_the_centroid_and_nothing_as_before(self, world):
        snap, _proto, make, embedded = world
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, make([], [40]), media_type="image", snap=snap) is False
        assert lt.train_from_labelset(ctx, make([1], [40]), media_type="image", snap=snap) is True
        assert is_centroid_head(ctx.model) and embedded == []

    def test_labels_at_the_quota_that_fail_to_resolve_are_not_hidden_by_the_query(self, world, monkeypatch):
        """3 Goods and 4 Bads as written, none resolved: a resolution failure to report, not a detector under the quota."""
        snap, _proto, make, _embedded = world

        def _resolve_nothing(det_ctx, labelset, *, media_type, snap, on_progress=None):
            det_ctx.embedder = EMBEDDER

        monkeypatch.setattr(lt, "populate_label_embeddings", _resolve_nothing)
        ctx = _ctx()
        at = make([1, 2, 3], [40, 41, 42, 43])
        assert lt.train_from_labelset(ctx, at, media_type="image", snap=snap, text_query=QUERY) is False
        assert ctx.model is None
        # Under the quota as written, the query is what resolving would have given anyway.
        assert lt.train_from_labelset(ctx, make([1], [40]), media_type="image", snap=snap, text_query=QUERY)
        assert is_query_head(ctx.model)

    def test_a_query_that_cannot_score_the_text_slot_falls_back_to_the_centroid(self, world, monkeypatch):
        """A query embedded in another width than the snapshot's text-slot vectors: no crash, the centroid."""
        snap, _proto, make, _embedded = world
        monkeypatch.setattr(qh, "embed_detector_query", lambda text, media_type, embedder: np.ones(DIM * 2))
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, make([1], [40]), media_type="image", snap=snap, text_query=QUERY)
        assert is_centroid_head(ctx.model)

    def test_a_dataset_with_no_text_slot_falls_back_to_the_centroid(self, world, monkeypatch):
        snap, _proto, make, _embedded = world
        monkeypatch.setattr(qh, "text_embedder_for_snap", lambda s: None)
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, make([1], [40]), media_type="image", snap=snap, text_query=QUERY) is True
        assert is_centroid_head(ctx.model)
        assert lt.train_from_labelset(_ctx(), make([], []), media_type="image", snap=snap, text_query=QUERY) is False

    def test_a_centroid_cached_before_the_query_is_not_handed_out(self, world):
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        ls = make([1], [40])
        lt.train_from_labelset(ctx, ls, media_type="image", snap=snap)
        assert is_centroid_head(ctx.model) and cached_head_is_current(ctx, ls)
        assert not cached_head_is_current(ctx, ls, QUERY)


class TestLabelsetTrainAndScore:
    def test_a_cold_find_gets_the_query_under_the_quota_scored_in_its_space(self, world):
        snap, proto, make, _embedded = world
        results, threshold, model = lt.labelset_train_and_score(
            _ctx(),
            make([1, 2], [40]),
            media_type="image",
            clips_dict=snap,
            beta=1.0,
            label_quota=True,
            text_query=QUERY,
        )
        assert is_query_head(model) and threshold == CENTROID_THRESHOLD and len(results) == len(snap)
        cosines = {cid: float(m["embeddings"][EMBEDDER] @ proto) for cid, m in snap.items()}
        assert [r["id"] for r in results] == sorted(cosines, key=lambda c: -cosines[c])

    def test_a_cold_find_with_no_label_at_all_gets_the_query(self, world):
        snap, _proto, make, _embedded = world
        _results, _t, model = lt.labelset_train_and_score(
            _ctx(), make([], []), media_type="image", clips_dict=snap, label_quota=True, text_query=QUERY
        )
        assert is_query_head(model)

    def test_the_learned_sort_never_takes_the_query(self, world):
        """``label_quota=False`` is the Train view's sort: it trains from the first Good and Bad, query or not."""
        snap, _proto, make, _embedded = world
        _results, _t, model = lt.labelset_train_and_score(
            _ctx(), make([1], [40]), media_type="image", clips_dict=snap, text_query=QUERY
        )
        assert model is not None and not is_quota_head(model)


class TestResolveOrTrain:
    def test_a_detector_made_from_a_typed_query_finds_before_any_label(self, world):
        snap, _proto, _make, _embedded = world
        det_data = {"name": "q", "text_query": QUERY, "labelset": {"labels": []}, "media_type": "image"}
        sink: list = []
        mlp, threshold, diag = resolve_or_train_detector(
            "", det_data, "image", snap, use_loaded_context=False, ctx_sink=sink, on_progress=lambda *a, **k: None
        )
        assert is_query_head(mlp) and threshold == CENTROID_THRESHOLD and diag is None
        assert sink and sink[0].model is mlp

    def test_a_cached_query_head_is_reused_only_in_its_own_text_space(self, world, monkeypatch):
        import vtscore.state.core as core

        snap, _proto, _make, _embedded = world
        det_data = {"name": "q", "text_query": QUERY, "labelset": {"labels": []}, "media_type": "image"}
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, LabelSet([]), media_type="image", snap=snap, text_query=QUERY)
        cached = ctx.model
        monkeypatch.setattr(core, "get_detector_context", lambda _id: ctx)

        def quiet(*_a: object, **_k: object) -> None:
            return None

        assert resolve_or_train_detector("q", det_data, "image", snap, on_progress=quiet)[0] is cached
        # Another dataset whose text slot is another embedder: the cached head is not handed out.
        monkeypatch.setattr(qh, "text_embedder_for_snap", lambda s: "another_text_embedder")
        assert resolve_or_train_detector("q", det_data, "image", snap, on_progress=quiet)[0] is not cached

    def test_a_detector_with_no_query_and_no_label_still_gives_nothing(self, world):
        snap, _proto, _make, _embedded = world
        det_data = {"name": "q", "text_query": "", "labelset": {"labels": []}, "media_type": "image"}
        mlp, _t, _d = resolve_or_train_detector(
            "", det_data, "image", snap, use_loaded_context=False, on_progress=lambda *a, **k: None
        )
        assert mlp is None


class TestTheBalance:
    """A typed query's head keeps the ranking its line is counted on (#5048), so the state counts that line."""

    LABELLED = {1, 40}

    def _unvoted_scores(self, ctx: DetectorContext, snap: dict[int, dict]) -> list[float]:
        rows = scoring_rows_for_snap(snap, EMBEDDER)
        scores = score_rows_with_model(ctx.model, rows)[0]
        return [s for cid, s in zip(rows.ids, scores, strict=True) if cid not in self.LABELLED]

    def test_the_state_counts_what_the_query_line_keeps_and_a_change_redraws_it(self, world):
        from vtscore.state.core import detector_balance_state, recut_detector_threshold
        from vtscore.training.thresholds import BALANCE_UNCHECKED, balance_count

        snap, _proto, make, _embedded = world
        for cid, media in snap.items():
            media["md5"] = f"m{cid}"  # the labels resolve in this corpus
        ctx = _ctx()
        lt.train_from_labelset(ctx, make([1], [40]), media_type="image", snap=snap, text_query=QUERY)
        assert is_query_head(ctx.model)
        assert ctx.line_ranking is not None and ctx.line_ranking.voted == self.LABELLED, "labelled media marked voted"
        unvoted = self._unvoted_scores(ctx, snap)
        kept = sum(1 for s in unvoted if s >= ctx.threshold)
        assert kept != balance_count(1.0, None), "the cap says something else"
        state = detector_balance_state(ctx, 1.0)
        assert state["status"] == BALANCE_UNCHECKED and state["count"] == kept

        recut = recut_detector_threshold(ctx, beta=4.0)
        assert recut is not None
        ctx.threshold = recut
        assert ctx.threshold == pytest.approx(ctx.centroid_line.threshold_at(4.0))
        assert detector_balance_state(ctx, 4.0)["count"] == sum(1 for s in unvoted if s >= ctx.threshold)

    def test_a_cold_find_keeps_the_ranking_too(self, world):
        from vtscore.state.core import detector_balance_state

        snap, _proto, make, _embedded = world
        for cid, media in snap.items():
            media["md5"] = f"m{cid}"
        ctx = _ctx()
        _results, threshold, model = lt.labelset_train_and_score(
            ctx, make([1], [40]), media_type="image", clips_dict=snap, label_quota=True, text_query=QUERY
        )
        assert is_query_head(model) and ctx.line_ranking is not None
        assert ctx.line_ranking.voted == self.LABELLED
        kept = sum(1 for s in self._unvoted_scores(ctx, snap) if s >= threshold)
        assert detector_balance_state(ctx, 1.0)["count"] == kept


# ---------------------------------------------------------------------------
# The hold past the quota (#4736)
# ---------------------------------------------------------------------------


@pytest.fixture
def single_vector(monkeypatch):
    """The test embedder as a known single-vector space (the registry does not know it: no hold by default)."""
    import vtscore.detectors.training as training

    monkeypatch.setattr(training, "_patch_space_for", lambda name: False if name == EMBEDDER else None)


class TestTheQueryHold:
    """On a single-vector space a typed-query detector keeps its query past the quota until QUERY_HOLD_LABELS labels."""

    def test_the_rule_reads_the_counts_and_the_space_only(self):
        from vtscore.detectors.label_quota import QUERY_HOLD_LABELS, query_holds

        assert QUERY_HOLD_LABELS == 30
        assert query_holds(3, 4, patch_space=False)
        assert query_holds(10, QUERY_HOLD_LABELS - 11, patch_space=False)
        assert not query_holds(10, QUERY_HOLD_LABELS - 10, patch_space=False)
        assert not query_holds(3, 4, patch_space=True), "a patch space's head passes the query by vote 10"
        assert not query_holds(3, 4, patch_space=None), "an unknown space keeps the hand-over at the quota"
        assert not query_holds(3, 4, patch_space=False, hold=0), "the arm before"

    @pytest.mark.usefixtures("single_vector")
    def test_past_the_quota_the_detector_is_still_its_query(self, world):
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        ls = make([1, 2, 3], [40, 41, 42, 43])
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap, text_query=QUERY) is True
        assert is_query_head(ctx.model) and query_head_text(ctx.model) == QUERY
        assert served_quota(ctx.model, ls)["tier"] == TIER_QUERY

    @pytest.mark.usefixtures("single_vector")
    def test_the_cli_says_the_hold_not_the_quota(self, world, capsys):
        """3 Goods and 4 Bads meet the quota: the CLI names the hold and what it owes, not "under the quota" (#4889)."""
        from vtscore.cli import detectors as cli_detectors
        from vtscore.cli import progress as cli_progress

        snap, _proto, make, _embedded = world
        ctx = _ctx()
        ls = make([1, 2, 3], [40, 41, 42, 43])
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap, text_query=QUERY) is True
        previous = cli_progress.get_format()
        cli_progress.set_format("text")
        try:
            cli_detectors._report_centroid("held", ctx, ls)
        finally:
            cli_progress.set_format(previous)
        line = capsys.readouterr().out.strip()
        assert "under the quota" not in line
        assert "until it has 30 labels (23 more labels, Good or Bad, for a trained detector)" in line

    @pytest.mark.usefixtures("single_vector")
    def test_at_the_hold_the_trained_head_takes_over(self, world):
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        ls = make(list(range(1, 9)), list(range(40, 62)))
        assert len(ls.elements) == 30
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap, text_query=QUERY) is True
        assert ctx.model is not None and not is_quota_head(ctx.model)

    @pytest.mark.usefixtures("single_vector")
    def test_without_a_query_the_quota_alone_decides(self, world):
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, make([1, 2, 3], [40, 41, 42, 43]), media_type="image", snap=snap) is True
        assert ctx.model is not None and not is_quota_head(ctx.model)

    def test_a_patch_space_has_no_hold(self, world, monkeypatch):
        import vtscore.detectors.training as training

        monkeypatch.setattr(training, "_patch_space_for", lambda name: True)
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        assert lt.train_from_labelset(
            ctx, make([1, 2, 3], [40, 41, 42, 43]), media_type="image", snap=snap, text_query=QUERY
        )
        assert ctx.model is not None and not is_quota_head(ctx.model)

    @pytest.mark.usefixtures("single_vector")
    def test_a_cold_find_holds_too_and_the_learned_sort_never_does(self, world):
        snap, _proto, make, _embedded = world
        ls = make([1, 2, 3], [40, 41, 42, 43])
        _r, _t, model = lt.labelset_train_and_score(
            _ctx(), ls, media_type="image", clips_dict=snap, label_quota=True, text_query=QUERY
        )
        assert is_query_head(model)
        _r, _t, model = lt.labelset_train_and_score(_ctx(), ls, media_type="image", clips_dict=snap, text_query=QUERY)
        assert model is not None and not is_quota_head(model), "the Train view asks from the head either way"

    def test_a_cached_trained_head_is_not_handed_out_while_the_query_holds(self, world, monkeypatch):
        """The learned sort caches its head under the labels' signature from the quota on; a Find must not take it."""
        import vtscore.detectors.training as training

        snap, _proto, make, _embedded = world
        ctx = _ctx()
        ls = make([1, 2, 3], [40, 41, 42, 43])
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap)  # a head under this signature
        assert ctx.model is not None and not is_quota_head(ctx.model)
        assert cached_head_is_current(ctx, ls, QUERY), "an unknown space hands over at the quota"
        monkeypatch.setattr(training, "_patch_space_for", lambda name: False)
        assert not cached_head_is_current(ctx, ls, QUERY), "a Find there gives the query's set"
        assert cached_head_is_current(ctx, ls, ""), "a detector with no query takes the head at the quota"


# ---------------------------------------------------------------------------
# The miss under the quota (#4800)
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_space(monkeypatch):
    """The test embedder as a known patch space (the Region path's DINOv3 patches)."""
    import vtscore.detectors.training as training

    monkeypatch.setattr(training, "_patch_space_for", lambda name: True if name == EMBEDDER else None)


MISSED = ([1], [40])  # a Good, and a Bad before the Goods quota


class TestTheQueryMiss:
    """On a patch space, once the typed query has missed, the Goods' centroid answers until the quota."""

    def test_the_rule_reads_the_counts_and_the_space_only(self):
        from vtscore.detectors.label_quota import (
            BAD_QUOTA,
            DRY_BAD_QUOTA,
            GOOD_QUOTA,
            QUERY_MISS_BADS,
            query_missed,
        )

        assert QUERY_MISS_BADS == 1, "the first Bad, the owner's choice over the Bads quota"
        assert query_missed(1, QUERY_MISS_BADS, patch_space=True)
        assert query_missed(GOOD_QUOTA - 1, DRY_BAD_QUOTA - 1, patch_space=True)
        assert not query_missed(1, 0, patch_space=True), "no Bad yet: the query is the better set"
        assert not query_missed(0, BAD_QUOTA, patch_space=True), "no Good to centre on"
        assert not query_missed(GOOD_QUOTA, BAD_QUOTA - 1, patch_space=True), "the Goods quota came first"
        assert not query_missed(1, DRY_BAD_QUOTA, patch_space=True), "the second quota's trained head"
        assert not query_missed(1, QUERY_MISS_BADS, patch_space=False), "a single-vector space keeps the query"
        assert not query_missed(1, QUERY_MISS_BADS, patch_space=None), "so does an unknown one"

    @pytest.mark.usefixtures("patch_space")
    def test_once_the_query_has_missed_the_detector_is_the_centroid(self, world):
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        ls = make(*MISSED)
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap, text_query=QUERY) is True
        assert is_centroid_head(ctx.model)
        assert served_quota(ctx.model, ls)["tier"] == "centroid"
        assert cached_head_is_current(ctx, ls, QUERY), "the centroid is what this detector gives now"

    @pytest.mark.usefixtures("patch_space")
    @pytest.mark.parametrize(("goods", "bads"), [([], [40, 41, 42, 43]), ([1], []), ([1, 2], []), ([1, 2, 3], [40])])
    def test_before_the_miss_the_detector_is_its_query(self, world, goods, bads):
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, make(goods, bads), media_type="image", snap=snap, text_query=QUERY)
        assert is_query_head(ctx.model)

    @pytest.mark.usefixtures("single_vector")
    def test_a_single_vector_space_keeps_the_query(self, world):
        snap, _proto, make, _embedded = world
        ctx = _ctx()
        assert lt.train_from_labelset(ctx, make(*MISSED), media_type="image", snap=snap, text_query=QUERY)
        assert is_query_head(ctx.model)

    @pytest.mark.usefixtures("patch_space")
    def test_a_cold_find_gives_the_centroid_too(self, world):
        snap, _proto, make, _embedded = world
        _r, threshold, model = lt.labelset_train_and_score(
            _ctx(), make(*MISSED), media_type="image", clips_dict=snap, label_quota=True, text_query=QUERY
        )
        assert is_centroid_head(model) and threshold == CENTROID_THRESHOLD

    def test_a_query_head_cached_before_the_miss_counted_is_not_handed_out(self, world, monkeypatch):
        import vtscore.detectors.training as training

        snap, _proto, make, _embedded = world
        ctx = _ctx()
        ls = make(*MISSED)
        assert lt.train_from_labelset(ctx, ls, media_type="image", snap=snap, text_query=QUERY)
        assert is_query_head(ctx.model) and cached_head_is_current(ctx, ls, QUERY), "an unknown space keeps it"
        monkeypatch.setattr(training, "_patch_space_for", lambda name: True)
        assert not cached_head_is_current(ctx, ls, QUERY)
