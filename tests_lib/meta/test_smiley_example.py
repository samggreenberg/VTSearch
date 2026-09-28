"""The user guide's worked example agrees with itself.

Every screenshot in the user guide is shot on the Smiley example
(`scripts/screenshots/smiley-example.mjs`): cartoon drawings that
`scripts/screenshots/smiley_fixture.py` makes with the Synthetic Media
generator, and a detector that learns to find the yellow smiley faces among
them (#4240). What the example is lives in three files, in two languages, and
each of these drifting apart breaks the guide without breaking anything else:

- The guide tells a reader which **Size** and **Seed** make the very same
  pictures; `CORPORA` in the fixture script is what the screenshots were
  actually taken on.
- The guide names the datasets, the detector and what to type; the harness
  shoots whatever `smiley-example.mjs` says.
- `HERO_REGION` names the one scene the region-voting shot draws a box in, and
  `VOTES` / `REGION_VOTES` ask for so many pictures of each kind. A generator
  change can leave the named scene without its one yellow smiley, or a corpus
  short of a kind, and the harness would only find out at the next refresh.

So they are checked here, without a browser or an app. The JavaScript is read
by parsing, like `test_slide_book_votes.py` does; the fixture script is
imported, since it is the one place the example's categories are defined.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
GUIDE = REPO / "docs" / "user" / "USER_GUIDE.md"
EXAMPLE = REPO / "scripts" / "screenshots" / "smiley-example.mjs"
FIXTURE = REPO / "scripts" / "screenshots" / "smiley_fixture.py"


def _fixture() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smiley_fixture", FIXTURE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _const(name: str) -> str:
    """The value of ``export const NAME = '...';`` in the example."""
    match = re.search(rf"export const {name} = '([^']*)';", EXAMPLE.read_text())
    assert match, f"smiley-example.mjs no longer declares {name} as a string"
    return match.group(1)


def _votes(name: str) -> tuple[int, dict[str, int]]:
    """``(good, {category: n})`` from ``export const NAME = { good: n, bad: {...} };``."""
    block = re.search(rf"export const {name} = \{{(.*?)\n\}};", EXAMPLE.read_text(), re.DOTALL)
    assert block, f"smiley-example.mjs no longer declares {name}"
    good = re.search(r"good: (\d+)", block.group(1))
    bad = re.search(r"bad: \{([^}]*)\}", block.group(1))
    assert good and bad, f"{name} has no good count or no bad table"
    table = {key: int(n) for key, n in re.findall(r"'?([\w-]+)'?:\s*(\d+)", bad.group(1))}
    assert table, f"{name}'s bad table parsed empty"
    return int(good.group(1)), table


def _guide() -> str:
    """The guide with every run of whitespace folded to one space."""
    return " ".join(GUIDE.read_text().split())


def test_the_guides_follow_along_settings_make_the_screenshots_pictures() -> None:
    match = re.search(
        r"set \*\*Size\*\* to (\d+), and set \*\*Seed\*\* to (\d+) for the training set \(Step 1\) "
        r"and to (\d+) for the new one \(Step 3\)",
        _guide(),
    )
    assert match, "the guide's Step by step no longer says which Size and Seeds make its pictures"
    size, train_seed, test_seed = (int(g) for g in match.groups())
    corpora = _fixture().CORPORA
    assert (size, train_seed) == corpora[_const("TRAIN_DATASET")]
    assert (size, test_seed) == corpora[_const("TEST_DATASET")]


def test_the_guide_names_what_the_harness_shoots() -> None:
    guide = _guide()
    for name in ("TRAIN_DATASET", "TEST_DATASET", "DETECTOR", "DETECTOR_TEXT"):
        assert f"`{_const(name)}`" in guide, f"the guide never names {name} ({_const(name)!r})"


def test_every_corpus_the_harness_imports_is_one_the_fixture_draws() -> None:
    corpora = _fixture().CORPORA
    for name in ("TRAIN_DATASET", "TEST_DATASET", "REGION_DATASET"):
        assert _const(name) in corpora


def test_the_region_shots_scene_holds_one_yellow_smiley() -> None:
    fixture = _fixture()
    hero = _const("HERO_REGION")
    pictures = {p["filename"]: p for p in fixture.pictures(_const("REGION_DATASET"))}
    assert hero in pictures, f"{hero} is not in the region corpus"
    assert len(pictures[hero]["yellow_smileys"]) == 1, f"{hero} no longer holds exactly one yellow smiley"


def test_the_vote_baselines_fit_their_corpora() -> None:
    fixture = _fixture()
    for votes, corpus in (("VOTES", "TRAIN_DATASET"), ("REGION_VOTES", "REGION_DATASET")):
        good, bad = _votes(votes)
        counts: dict[str, int] = {}
        for picture in fixture.pictures(_const(corpus)):
            counts[picture["category"]] = counts.get(picture["category"], 0) + 1
        wanted = {"yellow-smiley": good, **bad}
        short = {cat: (n, counts.get(cat, 0)) for cat, n in wanted.items() if counts.get(cat, 0) < n}
        assert not short, f"{votes} asks for more than {_const(corpus)} holds (wanted, held): {short}"


def test_the_detector_learns_yellow_and_smiling_not_yellow() -> None:
    # The Bads are what make the trained ranking a picture of a detector that
    # learned the concept: if they drift to easy pictures only, the shots stop
    # showing why the votes were needed.
    _good, bad = _votes("VOTES")
    assert bad.get("yellow-face"), "no yellow face that is not smiling among the Bads"
    assert bad.get("orange-smiley") or bad.get("smiley"), "no smiling face that is not yellow among the Bads"
