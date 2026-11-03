"""End-to-end subprocess smoke test for ``app.py``'s CLI entry point.

The bulk of the ``vtsearch.cli.main`` dispatch surface — argparse build,
two-pass parsing, ``_apply_*`` overrides, flag validation, error exits, and
the autodetect/pipeline/server wiring — is covered in-process (and far
faster) by ``test_cli_main.py``, which calls ``cli_main.main`` directly with
the heavy stages mocked.

These ``slow`` subprocess tests remain to guard what an in-process test
can't reach: the real seam between ``app.py``'s ``__main__`` block and
``cli_main.main(app, initialize_server)`` when a fresh interpreter actually
imports the whole app and runs a full ``--autodetect`` dispatch (``--dry-run``
keeps it model-free), and what that import writes to stdout before argv is
parsed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers import make_dataset_file as _make_dataset_file
from vtsearch.state import medias

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_app(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603  # interpreter + test-controlled args
        [sys.executable, "app.py", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env} if env else None,
    )


@pytest.mark.slow
def test_app_autodetect_dry_run_via_subprocess(tmp_path):
    """``python app.py --autodetect --dataset ... --dry-run`` prints the plan
    and exits cleanly, with no exporter side effects."""
    dataset_path = _make_dataset_file(tmp_path, medias)
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
def test_app_pipeline_json_progress_is_ndjson_on_stdout(tmp_path):
    """``--pipeline … --progress-format json`` (#4753): every stdout line is an
    NDJSON event, including those written before argv is parsed (the startup
    banner goes to stderr), so a ``| jq`` consumer never meets bare prose."""
    result = _run_app("--pipeline", str(tmp_path / "missing.yaml"), "--progress-format", "json")

    assert result.returncode == 1, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [e["event"] for e in events] == ["error"]
    assert "Pipeline file not found" in events[0]["message"]


@pytest.mark.slow
def test_server_init_variable_does_not_start_a_server_under_a_cli_run(tmp_path):
    """Every Docker image sets ``VTSEARCH_SERVER_INIT=1`` image-wide (#4970).

    It is meant for gunicorn, which imports ``app.py``; a ``python app.py``
    headless run in the same container must not get the server's startup with
    it - the banners on stdout ahead of the NDJSON stream, the env-var admin
    overrides, the model warm-up.  The data directory is a fresh one so that,
    should this regress, the preload finds nothing to download.
    """
    result = _run_app(
        "--pipeline",
        str(tmp_path / "missing.yaml"),
        "--progress-format",
        "json",
        env={
            "VTSEARCH_SERVER_INIT": "1",
            "VTSEARCH_DATA_DIR": str(tmp_path / "data"),
            "VTSEARCH_DATASET_MAX_AGE_DAYS": "14",
        },
    )

    assert result.returncode == 1, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert [e["event"] for e in events] == ["error"]
    assert "Running in" not in result.stdout + result.stderr
