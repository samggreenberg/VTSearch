"""The document stop's readout (#4488): standing non-Good votes since the last Good."""

from vtscore.detectors.labeling_progress import DRY_RUN_TARGET, document_labeling_status, dry_run_status


def _history(*votes: tuple[int, str]) -> list[tuple[int, str, float]]:
    return [(media_id, label, float(i)) for i, (media_id, label) in enumerate(votes)]


def _status(*votes: tuple[int, str]) -> dict:
    """The readout over *votes*, each media's last vote standing."""
    current = {media_id: label for media_id, label in votes}
    good = {m: None for m, label in current.items() if label == "good"}
    bad = {m: None for m, label in current.items() if label == "bad"}
    return dry_run_status(_history(*votes), good, bad)


def _bads(start: int, n: int) -> list[tuple[int, str]]:
    return [(start + i, "bad") for i in range(n)]


def test_the_target_is_the_walks_dry_run():
    assert DRY_RUN_TARGET == 16


def test_counts_bads_since_the_last_good():
    s = _status((1, "good"), *_bads(10, 3), (2, "good"), *_bads(20, 5))
    assert s["run"] == 5
    assert s["status"] == "red"


def test_red_yellow_green():
    assert _status((1, "good"), *_bads(10, 7))["status"] == "red"
    assert _status((1, "good"), *_bads(10, 8))["status"] == "yellow"
    assert _status((1, "good"), *_bads(10, 15))["status"] == "yellow"
    green = _status((1, "good"), *_bads(10, 16))
    assert (green["status"], green["run"], green["target"]) == ("green", 16, 16)


def test_no_good_is_red_however_long_the_run():
    s = _status(*_bads(10, 30))
    assert (s["status"], s["run"]) == ("red", 30)


def test_a_bad_flipped_to_good_is_a_good():
    """The media's last vote stands; its earlier Bad neither counts nor breaks anything."""
    s = _status((1, "good"), *_bads(10, 4), (10, "good"), *_bads(20, 2))
    assert s["run"] == 2


def test_a_removed_vote_is_skipped():
    """A vote in the history whose media no longer carries it is not counted."""
    history = _history((1, "good"), *_bads(10, 5))
    bad = {m: None for m in range(10, 14)}  # 14 was un-voted
    assert dry_run_status(history, {1: None}, bad)["run"] == 4


def test_a_removed_good_does_not_reset_the_run():
    history = _history((1, "good"), *_bads(10, 3), (2, "good"), *_bads(20, 3))
    bad = {m: None for m in [10, 11, 12, 20, 21, 22]}
    assert dry_run_status(history, {1: None}, bad)["run"] == 6  # 2 was un-voted


def test_a_revote_counts_once():
    history = _history((1, "good"), (10, "bad"), (10, "bad"), (11, "bad"))
    assert dry_run_status(history, {1: None}, {10: None, 11: None})["run"] == 2


def test_document_status_turns_the_lights_off():
    history = _history((1, "good"), *_bads(10, 16))
    bad = {m: None for m in range(10, 26)}
    s = document_labeling_status(history, {1: None}, bad)
    assert s["stop_rule"] == "dry_run"
    assert (s["good_count"], s["bad_count"], s["total_count"]) == (1, 16, 17)
    assert s["dry_run"]["status"] == "green"
    assert {s[k]["status"] for k in ("smart", "stable", "span")} == {"off"}
