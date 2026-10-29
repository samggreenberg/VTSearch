"""The small modules folded into their callers keep their old names (#5399).

``vtscore.utils.hits``, ``vtscore.utils.scores``, ``vtscore.utils.import_metadata``
and ``vtscore.state.clicks`` were folded into the one module that called each.
Out-of-tree code may import them by the old names, so each old name is a
deprecated alias that hands back the module it was folded into: a stub patched
through either name is the one the library reads.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from vtscore.datasets import labelset
from vtscore.media.torch_utils import torch_setup
from vtscore.state import votes
from vtscore.training.thresholds import knobs

ALIASES = [
    ("vtscore.utils.hits", labelset, ("build_media_hit", "hit_custom_metadata")),
    (
        "vtscore.utils.scores",
        knobs,
        (
            "NON_FINITE_SCORE_SENTINEL",
            "finite_or",
            "scored_mask",
            "scored_only",
            "sigmoid_to_finite_array",
            "sigmoid_to_finite_scores",
        ),
    ),
    (
        "vtscore.utils.import_metadata",
        torch_setup,
        ("fast_packages_distributions", "original_packages_distributions", "seed_packages_distributions"),
    ),
    ("vtscore.state.clicks", votes, ("assign_click_time", "get_vote_click_times", "remove_click_time")),
]


@pytest.mark.parametrize(("old_name", "module", "names"), ALIASES, ids=[a[0] for a in ALIASES])
def test_an_old_module_name_is_the_module_it_was_folded_into_and_warns(old_name, module, names, monkeypatch):
    monkeypatch.delitem(sys.modules, old_name, raising=False)
    with pytest.warns(DeprecationWarning, match=f"{old_name} has moved to {module.__name__}"):
        old = importlib.import_module(old_name)
    assert old is module
    assert sys.modules[old_name] is module
    for name in names:
        assert hasattr(module, name), f"{module.__name__} lost {name}, which {old_name} used to export"


def test_a_stub_patched_through_the_old_name_is_the_one_the_votes_call(monkeypatch):
    monkeypatch.delitem(sys.modules, "vtscore.state.clicks", raising=False)
    with pytest.warns(DeprecationWarning):
        clicks = importlib.import_module("vtscore.state.clicks")

    assigned = []
    monkeypatch.setattr(clicks, "assign_click_time", assigned.append)
    votes.apply_label_with_click_time(7, "good")
    assert assigned == [7]
