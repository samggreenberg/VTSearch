"""End-to-end subprocess coverage for ``app.py``'s CLI entry point.

Unlike the rest of ``tests/cli/`` (which calls the ``vtscore.cli``
``autodetect_*`` functions directly), these spawn a fresh ``python app.py``
process so the full ``vtsearch.cli_main`` dispatch path - argparse build,
two-pass parsing, ``_apply_*`` overrides, ``_run_autodetect`` ->
``_dispatch_autodetect`` - is exercised the way a real CLI invocation hits
it. They guard the seam between ``app.py``'s ``__main__`` block and
``cli_main.main(app, initialize_server)``.

Marked ``slow`` because each spawns an interpreter that imports the whole
app; ``--dry-run`` keeps it model-free.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import app as app_module
from helpers import make_dataset_file as _make_dataset_file

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_app(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603  # interpreter + test-controlled args
        [sys.executable, "app.py", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.slow
def test_app_autodetect_dry_run_via_subprocess(tmp_path):
    """``python app.py --autodetect --dataset ... --dry-run`` prints the plan
    and exits cleanly, with no exporter side effects."""
    dataset_path = _make_dataset_file(tmp_path, app_module.medias)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"autofind_detectors": [], "detectors_dir": str(tmp_path / "det")}))
    out_path = tmp_path / "results.json"

    result = _run_app(
        "--autodetect",
        "--dataset",
        str(dataset_path),
        "--settings",
        str(settings_path),
        "--exporter",
        "server_json_file",
        "--filepath",
        str(out_path),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert str(dataset_path) in result.stdout
    assert "Exporter: server_json_file" in result.stdout
    # Critical: dry-run must not actually run the exporter.
    assert not out_path.exists()


@pytest.mark.slow
def test_app_autodetect_missing_target_errors_via_subprocess():
    """``--autodetect`` with neither ``--dataset`` nor ``--importer`` exits 2
    with the dispatch helper's specific message (argparse ``parser.error``)."""
    result = _run_app("--autodetect")

    assert result.returncode == 2
    assert "requires either --dataset" in result.stderr
