"""Tests for ``scripts/slurm/vtsearch-tunnel.sh`` and its service unit.

The tunnel grew two knobs for an always-on relay box: ``VTS_BIND`` (which
address the forwarded port is bound to locally) and ``--no-shell`` (hold the
forward open with no TTY, for the unit). Both are one-liners that are easy to
get subtly wrong and impossible to notice when wrong -- an unbound port simply
isn't there from the other device, and a shell-mode unit exits instantly.

Nothing here reaches a cluster. The forward spec is *evaluated out of the
shipped script* rather than restated, so the test cannot drift away from the
line it is checking.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "slurm" / "vtsearch-tunnel.sh"
UNIT = REPO_ROOT / "scripts" / "slurm" / "vtsearch-tunnel.service"


def _forward_spec(*, port: str = "10042", node: str = "gpu07", bind: str | None = None) -> str:
    """Evaluate the script's own ``FORWARD=`` line under controlled variables.

    Extracting the real line keeps this honest: restating the parameter
    expansion here would pass forever even after the script stopped agreeing
    with it.
    """
    line = next(
        ln for ln in SCRIPT.read_text().splitlines() if ln.startswith("FORWARD=")
    )
    preamble = f'PORT={port}; NODE={node}; VTS_BIND={bind!r}' if bind is not None else f"PORT={port}; NODE={node}"
    out = subprocess.run(
        ["bash", "-c", f'set -u; {preamble}; {line}; printf "%s" "$FORWARD"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


class TestForwardSpec:
    def test_unset_bind_keeps_loopback_default(self):
        # No bind address means ssh's own default: loopback only, which is what
        # a laptop in front of you wants and the behaviour that predates the knob.
        assert _forward_spec() == "10042:gpu07:10042"

    def test_bind_address_is_prefixed(self):
        assert _forward_spec(bind="100.83.1.4") == "100.83.1.4:10042:gpu07:10042"

    def test_empty_bind_is_treated_as_unset(self):
        # The unit ships `Environment=VTS_BIND=` as a placeholder, so the empty
        # value has to mean "loopback", not a spec with a stray leading colon.
        assert _forward_spec(bind="") == "10042:gpu07:10042"


class TestScriptShape:
    def test_script_parses(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_no_shell_mode_holds_the_forward_without_a_remote_command(self):
        text = SCRIPT.read_text()
        no_shell = text[text.index('if [ "$NO_SHELL" -eq 1 ]') :]
        no_shell = no_shell[: no_shell.index("fi")]
        # -N (no remote command) is what makes this a tunnel rather than a login;
        # -t would allocate a TTY the unit has no way to provide.
        assert "ssh -N" in no_shell
        assert "-t " not in no_shell

    def test_no_shell_sets_keepalives(self):
        # Without these a dead link hangs until TCP gives up, and the unit sits
        # "active" while forwarding nothing -- the failure this repo keeps hitting.
        text = SCRIPT.read_text()
        assert "ServerAliveInterval" in text
        assert "ServerAliveCountMax" in text

    def test_unknown_argument_is_rejected(self):
        out = subprocess.run(
            ["bash", str(SCRIPT), "--bogus"], capture_output=True, text=True
        )
        assert out.returncode == 2
        assert "Unknown argument" in out.stderr

    def test_both_exec_paths_use_the_shared_forward_spec(self):
        # Two `ssh -L` call sites; neither may rebuild the spec by hand.
        sites = re.findall(r'-L "([^"]+)"', SCRIPT.read_text())
        assert sites, "no -L call sites found"
        assert set(sites) == {"$FORWARD"}


def _sections(text: str) -> dict[str, list[str]]:
    """Split an INI-style unit into {section: [directive lines]}.

    Comments are dropped first: the prose in this unit names ``[Service]`` while
    explaining which section a key belongs in, and a naive split on that literal
    slices the file in the wrong place (as the first version of this test did).
    """
    out: dict[str, list[str]] = {}
    current = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line
            out.setdefault(current, [])
        elif current:
            out[current].append(line)
    return out


class TestServiceUnit:
    def test_start_limit_lives_in_the_unit_section(self):
        # systemd moved this key to [Unit]; left in [Service] it is warned about
        # and ignored, so the unit would quietly stop retrying after 5 restarts.
        sections = _sections(UNIT.read_text())
        assert "StartLimitIntervalSec=0" in sections["[Unit]"]
        assert not any(d.startswith("StartLimit") for d in sections["[Service]"])

    def test_runs_the_script_not_a_bare_ssh(self):
        # Restarting the script rediscovers node and port; restarting an ssh
        # would re-dial whatever node the last allocation happened to land on.
        text = UNIT.read_text()
        exec_start = next(ln for ln in text.splitlines() if ln.startswith("ExecStart="))
        assert exec_start.endswith("vtsearch-tunnel --no-shell")

    def test_restarts_forever(self):
        text = UNIT.read_text()
        assert "Restart=always" in text
        # Between allocations there is legitimately no job, so a tight respin
        # would poll squeue continuously for hours.
        assert re.search(r"^RestartSec=(\d+)", text, re.M).group(1) != "0"
