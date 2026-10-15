"""The ``apptitude`` loader's decisions, on a synthetic Rico small enough to reason about (#4356).

What needs pinning is what the loader decides:

* **A label is two signals agreeing.** The developer's Activity name proposes a
  class and at least two crowd summaries must say it too. A disagreement, a
  generic ``MainActivity`` and an uncaptioned screen are all left out of the pool:
  none of them has an accepted label, so none can be anybody's negative.
* **The first class to match the Activity wins**, so ``SignupLoginActivity`` is
  a sign-up proposal and an ``AdActivity`` is nobody's candidate.
* **A screen is not a negative for a class its captions also say.**
* **Byte-identical screens are one media**, the lowest id.
* **Split groups join both packages**: a screen of the system browser opened from
  app A and app A's own screens are one group.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

_PILE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"


@pytest.fixture(scope="module")
def pc():
    if str(_PILE_DIR) not in sys.path:
        sys.path.insert(0, str(_PILE_DIR))
    import pile_config

    return pile_config


@pytest.fixture
def mod(pc):
    pytest.importorskip("PIL")
    from pilebuild.loaders import apptitude

    return apptitude


def _jpg(shade: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (54, 96), (shade, shade, shade)).save(buf, format="JPEG")
    return buf.getvalue()


LOGIN = ["a login page", "screen to sign in", "log in with email", "a page", "app screen"]
LOGIN_AND_SIGNUP = ["login page", "log in or sign up", "sign up for an account", "sign in", "page"]
SETTINGS = ["settings page", "app settings", "list of preferences", "options", "a screen"]
NOTHING = ["a screen", "a page", "an app", "some text", "a list"]

#: rico_id: (activity_name, traced package, summaries or None, pixel shade)
_SCREENS = {
    1: ("com.a/com.a.LoginActivity", "com.a", LOGIN, 10),
    2: ("com.a/com.a.SettingsActivity", "com.a", SETTINGS, 20),
    3: ("com.b/com.b.LoginActivity", "com.b", NOTHING, 30),  # disagreement: activity only
    4: ("com.b/com.b.MainActivity", "com.b", LOGIN, 40),  # captions only: generic activity
    5: ("com.c/com.c.LoginActivity", "com.c", None, 50),  # uncaptioned
    6: ("com.c/com.c.LoginActivity", "com.c", LOGIN_AND_SIGNUP, 60),  # login, also says sign-up
    7: ("com.d/com.d.SignupLoginActivity", "com.d", LOGIN_AND_SIGNUP, 70),  # sign-up wins the activity
    8: ("com.d/com.d.AdActivity", "com.d", LOGIN, 80),  # not on the roster
    9: ("com.android.browser/com.android.browser.PreferencesActivity", "com.e", SETTINGS, 90),
    10: ("com.f/com.f.SettingsActivity", "com.f", SETTINGS, 100),
    11: ("com.f/com.f.SettingsActivity", "com.f", SETTINGS, 100),  # byte-identical to 10
    12: ("com.android.browser/com.android.browser.PreferencesActivity", "com.g", SETTINGS, 120),
}


@pytest.fixture
def root(pc, tmp_path, monkeypatch):
    combined = tmp_path / "combined"
    combined.mkdir()
    (tmp_path / "screen2words").mkdir()
    with open(tmp_path / "ui_details.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["UI Number", "App Package Name", "Interaction Trace Number", "UI Number in Trace"])
        for rid, (_, pkg, _, _) in _SCREENS.items():
            w.writerow([rid, pkg, 0, rid])
    with open(tmp_path / "screen2words" / "screen_summaries.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["screenId", "summary"])
        for rid, (_, _, caps, _) in _SCREENS.items():
            for c in caps or []:
                w.writerow([rid, c])
    for rid, (act, _, _, shade) in _SCREENS.items():
        (combined / f"{rid}.json").write_text(json.dumps({"activity_name": act}))
        (combined / f"{rid}.jpg").write_bytes(_jpg(shade))
    monkeypatch.setattr(pc, "RICO_ROOT", tmp_path)
    return tmp_path


def _load(mod) -> dict[int, dict]:
    medias: dict[int, dict] = {}
    mod.load("apptitude", medias, "siglip")
    return medias


class TestWhatIsInThePool:
    def test_only_agreed_roster_screens(self, mod, root):
        assert sorted(_load(mod)) == [1, 2, 6, 7, 9, 10, 12]

    def test_each_carries_its_one_class(self, mod, root):
        m = _load(mod)
        assert {rid: x["category"] for rid, x in m.items()} == {
            1: "login",
            2: "settings",
            6: "login",
            7: "signup",
            9: "settings",
            10: "settings",
            12: "settings",
        }
        assert all(x["categories"] == [x["category"]] for x in m.values())

    def test_the_first_activity_rule_wins(self, mod):
        assert mod.activity_class("com.d/com.d.SignupLoginActivity") == "signup"
        assert mod.activity_class("com.d/com.d.AdActivity") == "ad"
        assert mod.activity_class("com.d/com.d.MainActivity") is None

    def test_a_byte_identical_copy_is_one_media(self, mod, root):
        m = _load(mod)
        assert 10 in m and 11 not in m


class TestNegatives:
    def test_a_screen_is_not_a_negative_for_a_class_its_captions_say(self, mod, root):
        m = _load(mod)
        assert "signup" not in m[6]["evaluable_categories"]
        assert "login" not in m[7]["evaluable_categories"]
        assert "login" in m[6]["evaluable_categories"]

    def test_a_plain_screen_is_evaluable_for_the_whole_roster(self, mod, root):
        assert _load(mod)[2]["evaluable_categories"] == list(mod.ROSTER)

    def test_the_harness_reads_it_that_way(self, mod, root):
        from vtscore.eval.labels import evaluable_pool, media_is_positive

        m = _load(mod)
        pool = evaluable_pool(m, "signup")
        assert 6 not in pool and 7 in pool and 1 in pool
        assert media_is_positive(m[7], "signup") and not media_is_positive(m[1], "signup")


class TestSplitGroups:
    def test_an_app_is_one_group(self, mod, root):
        m = _load(mod)
        assert m[1]["split_group"] == m[2]["split_group"]

    def test_both_packages_join_a_group(self, mod, root):
        """The system browser opened from com.e and from com.g joins them."""
        m = _load(mod)
        assert m[9]["split_group"] == m[12]["split_group"]
        assert m[9]["split_group"] != m[1]["split_group"]


class TestCheck:
    def test_it_counts_what_a_build_would_read(self, mod, root):
        assert "7 agreed screens" in mod.check("apptitude")

    def test_a_missing_source_fails_loudly(self, mod, root):
        (root / "ui_details.csv").unlink()
        with pytest.raises(SystemExit, match="ui_details.csv"):
            mod.check("apptitude")

    def test_a_captioned_screen_without_a_hierarchy_fails_loudly(self, mod, root):
        (root / "combined" / "6.json").unlink()
        with pytest.raises(SystemExit, match="6"):
            mod.check("apptitude")
