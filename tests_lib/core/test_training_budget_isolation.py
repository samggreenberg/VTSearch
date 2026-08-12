"""The session's training budget survives a mid-session ``vtscore.config`` reload.

``tests_lib/core/test_torch_config.py`` reloads ``vtscore.config`` to re-read
env vars at import time, which resets every module-level constant - including
the conftests' ``TRAIN_EPOCHS`` override - to its production default.  Nothing
restored it, so a fully seeded detector fixture trained 200 epochs instead of 30
whenever those tests happened to land earlier on the same xdist worker, and the
resulting threshold divergence read as nondeterminism in the safe-threshold
subsystem (issue #3101).

The reloading file now restores its own module snapshot and the conftests
re-assert the budget per test.  This pins the guarantee those two provide, in
the order that actually exposes it: poison the budget in one test, check it is
back in the next.  ``xdist_group`` keeps the pair on one worker under
``--dist loadgroup``; without it they could be split across processes and the
second test would pass without ever seeing the first one's damage.
"""

from __future__ import annotations

import importlib
import os

import pytest

from vtscore import config

from tests_lib.conftest import TEST_TRAIN_EPOCHS

pytestmark = pytest.mark.xdist_group("training-budget-isolation")


def test_a_reloading_config_resets_the_training_budget():
    """The hazard itself: a reload silently reverts the budget to production.

    The budget is the one that read as nondeterminism, but it is not the only
    session-level value the reload drops - the registered ``CoreConfig``
    builder goes with it, and its absence makes ``from_settings()`` raise.
    """
    importlib.reload(config)
    assert config.TRAIN_EPOCHS == int(os.environ.get("VTSEARCH_TRAIN_EPOCHS", "200"))
    assert config._core_config_builder is None


def test_b_the_next_test_still_gets_the_session_budget():
    """...and the next test is handed the session's budget regardless."""
    assert config.TRAIN_EPOCHS == TEST_TRAIN_EPOCHS
    # Restored, and *callable*: registration alone would still leave
    # ``from_settings()`` raising if the builder came back unusable.
    assert config._core_config_builder is not None
    assert config.CoreConfig.from_settings().data_dir == config.DATA_DIR
