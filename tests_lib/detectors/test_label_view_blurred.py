"""A stored label's ``blurred`` in the labels-detail view (#5696).

An element outside the active dataset has no Blur / Unblur override and no
dataset metadata to read, so the view applies the blur rules to the element's
own stored ``metadata``; an element inside it leaves the answer to the
dataset's blur state, which honours overrides.
"""

from __future__ import annotations

from typing import Any

from vtscore.datasets.labelset import LabeledElement
from vtscore.detectors.labelset_elements import build_element_view
from vtscore.state.media_lookup import build_media_lookup

PORN = {"field": "Porn", "op": ">=", "value": 0.8}

#: The active dataset, holding one item the element below resolves to.
IN_DATASET = {
    7: {
        "id": 7,
        "md5": "a1" * 16,
        "origin": {"importer": "test", "params": {}},
        "origin_name": "alpha.png",
    }
}


def _elem(md5: str, metadata: dict[str, Any] | None) -> LabeledElement:
    return LabeledElement(
        md5=md5,
        label="good",
        origin={"importer": "test", "params": {}},
        origin_name="alpha.png" if md5 == "a1" * 16 else "elsewhere.png",
        filename="x.png",
        metadata=metadata,
    )


def _view(elem: LabeledElement, rules: list[dict[str, Any]]) -> dict[str, Any]:
    return build_element_view(
        elem,
        media_type="image",
        click_times={},
        learned_scores={},
        lookups=build_media_lookup(IN_DATASET),
        blur_rules=rules,
    )


def test_outside_the_dataset_a_matching_rule_blurs() -> None:
    view = _view(_elem("e5" * 16, {"Porn": "0.93"}), [PORN])
    assert view["cid"] is None
    assert view["blurred"] is True


def test_outside_the_dataset_a_rule_that_does_not_match_leaves_it_sharp() -> None:
    assert _view(_elem("e5" * 16, {"Porn": 0.1}), [PORN])["blurred"] is False


def test_outside_the_dataset_no_metadata_leaves_it_sharp() -> None:
    assert _view(_elem("e5" * 16, None), [PORN])["blurred"] is False


def test_with_no_rules_nothing_is_blurred() -> None:
    assert _view(_elem("e5" * 16, {"Porn": 0.99}), [])["blurred"] is False


def test_inside_the_dataset_the_dataset_state_decides() -> None:
    """``null``: the element's own metadata may be stale, and an override may
    have overruled the rules, so the view does not answer for it."""
    view = _view(_elem("a1" * 16, {"Porn": 0.99}), [PORN])
    assert view["cid"] == 7
    assert view["blurred"] is None


def test_rules_default_to_none() -> None:
    """Callers that predate the rules (the eval harness, scripts) still get a view."""
    view = build_element_view(
        _elem("e5" * 16, {"Porn": 0.99}), media_type="image", click_times={}, learned_scores={}, lookups=({}, {}, {})
    )
    assert view["blurred"] is False
