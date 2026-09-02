"""Which checkout this run resolves to, and where the shared logging goes."""

from __future__ import annotations

import sys
from pathlib import Path


def log(msg: str) -> None:
    print(f"[pile] {msg}", flush=True)


def assert_vtscore_is_this_checkout() -> None:
    """Refuse to run against a different checkout's ``vtscore``.

    The venv's editable install points at the main checkout. If anything
    resolves ``vtscore`` there instead of here, cells get embedded by whatever
    code that tree happens to be on — silently, and possibly by a different
    embedder implementation. Cheap to assert, expensive to discover later.
    """
    import vtscore  # noqa: PLC0415

    want = Path(__file__).resolve().parents[4]
    got = Path(vtscore.__file__).resolve().parent.parent
    if got != want:
        raise SystemExit(
            f"vtscore resolved to {got}, not this checkout ({want}).\n"
            f"  Something put another checkout ahead on sys.path — usually the venv's\n"
            f"  editable install. Re-run with VTS_REPO={want} set for THIS command\n"
            f"  (note `VAR=x cmd1 && cmd2` sets VAR for cmd1 only)."
        )


def _calibration_path() -> None:
    calib = Path(__file__).resolve().parents[2] / "calibration"
    if str(calib) not in sys.path:
        sys.path.insert(0, str(calib))


def cells_io():
    """Import the calibration harness's pickle IO (drops bytes, keeps patch_grid)."""
    _calibration_path()
    import _cells_io  # noqa: PLC0415

    return _cells_io


def experiment_config():
    """Import the calibration harness's category-selection config."""
    _calibration_path()
    import experiment_config  # noqa: PLC0415

    return experiment_config
