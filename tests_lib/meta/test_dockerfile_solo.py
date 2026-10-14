"""``docker/Dockerfile.solo`` and its weight-baking script (issue #4868).

The image locks itself to the media type named by its ``MEDIA_TYPE`` build arg
and bakes that type's default embedder through
``scripts/preload_default_embedder.py``. A build only fails when that script
says so: the server-side preload it calls logs a load failure and carries on,
so a script that forgot to check the result would ship an image with no
weights in it, which then downloads them on first boot -- exactly what the
image exists to avoid, and invisible at build time.

So what is pinned here is the script's exit status for each way a build can go
wrong (no arg, unknown type, nothing loaded), with the preload stubbed out --
no download, no torch. Plus two static checks over the build files: the
compose files name real media types, and the Dockerfile wires the same
``MEDIA_TYPE`` into both the bake and the lock.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

import vtscore.embedding
from vtscore.media import all_type_ids

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "preload_default_embedder.py"
_DOCKERFILE = _ROOT / "docker" / "Dockerfile.solo"


@pytest.fixture(scope="module")
def preload():
    spec = importlib.util.spec_from_file_location("preload_default_embedder", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A models dir, and ``VTSEARCH_DATA_DIR`` registered so the script's write is undone."""
    monkeypatch.setenv("VTSEARCH_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("VTSEARCH_DATA_DIR", str(tmp_path / "data"))


@pytest.fixture
def calls(monkeypatch):
    """Stub the server preload: record its calls, load whatever ``result`` holds."""
    state: dict = {"calls": [], "result": None}

    def fake(extra_media_types=None, extra_embedders=None):
        state["calls"].append(extra_media_types)
        wanted = vtscore.embedding.predict_embedders_to_preload(extra_media_types=extra_media_types)
        return wanted if state["result"] is None else state["result"]

    monkeypatch.setattr(vtscore.embedding, "preload_predicted_embedders", fake)
    return state


def test_no_media_type_fails_and_names_the_build_arg(preload, env, calls, capsys):
    assert preload.main([]) == 2
    assert preload.main([""]) == 2
    assert "--build-arg MEDIA_TYPE=" in capsys.readouterr().err
    assert calls["calls"] == []


def test_unknown_media_type_fails_with_the_valid_list(preload, env, calls, capsys):
    assert preload.main(["vidoe"]) == 2
    err = capsys.readouterr().err
    assert "'vidoe'" in err
    assert "video" in err
    assert calls["calls"] == []


def test_missing_models_dir_fails(preload, env, calls, monkeypatch):
    monkeypatch.delenv("VTSEARCH_MODELS_DIR")
    assert preload.main(["audio"]) == 1
    assert calls["calls"] == []


def test_a_type_the_server_warms_nothing_for_has_nothing_to_bake(preload, env, calls, monkeypatch):
    monkeypatch.setattr(vtscore.embedding, "predict_embedders_to_preload", lambda **_: [])
    assert preload.main(["image"]) == 0
    assert calls["calls"] == []


def test_half_types_bake_what_they_embed_as(preload, env, calls):
    """document's pages embed as images; face crops embed with the face model."""
    assert preload.main(["document"]) == 0
    assert preload.main(["face"]) == 0
    assert calls["calls"] == [["document"], ["face"]]


def test_dockerfile_installs_facenet_without_deps() -> None:
    """A face container needs facenet-pytorch, and its pins would downgrade torch."""
    text = re.sub(r"\\\n\s*", " ", _DOCKERFILE.read_text())
    installs = [seg for seg in re.split(r"&&|\n", text) if "pip install" in seg and "facenet-pytorch" in seg]
    assert installs, "docker/Dockerfile.solo does not install facenet-pytorch, so MEDIA_TYPE=face cannot bake"
    assert all("--no-deps" in seg for seg in installs), installs


def test_bakes_through_the_server_preload(preload, env, calls):
    assert preload.main(["audio"]) == 0
    assert calls["calls"] == [["audio"]]


def test_a_preload_that_loads_nothing_fails_the_build(preload, env, calls, capsys):
    """The case the script exists for: the server preload swallowed the error."""
    calls["result"] = []
    assert preload.main(["audio"]) == 1
    assert "failed to bake" in capsys.readouterr().err


def _compose_files_building_solo() -> list[Path]:
    found = []
    for path in sorted((_ROOT / "docker" / "compose").glob("docker-compose*.yml")):
        services = (yaml.safe_load(path.read_text()) or {}).get("services", {})
        if any((svc.get("build") or {}).get("dockerfile") == "docker/Dockerfile.solo" for svc in services.values()):
            found.append(path)
    return found


def test_the_audio_and_video_compose_files_build_the_solo_image() -> None:
    names = {p.name for p in _compose_files_building_solo()}
    assert {"docker-compose.audio.yml", "docker-compose.video.yml"} <= names


@pytest.mark.parametrize("compose", _compose_files_building_solo(), ids=lambda p: p.name)
def test_compose_media_type_is_a_registered_type(compose: Path) -> None:
    """A typo here would only surface after the image's pip install, minutes in."""
    for svc in yaml.safe_load(compose.read_text())["services"].values():
        media_type = svc["build"]["args"]["MEDIA_TYPE"]
        assert media_type in all_type_ids(), f"{compose.name}: MEDIA_TYPE {media_type!r} is not a registered media type"


def test_dockerfile_bakes_and_locks_the_same_media_type() -> None:
    """One ``MEDIA_TYPE`` feeds both, so the baked weights match the lock."""
    text = re.sub(r"\\\n\s*", " ", _DOCKERFILE.read_text())
    assert re.search(r"^ARG MEDIA_TYPE=?\s*$", text, re.MULTILINE)
    assert re.search(r'^RUN .*python -u scripts/preload_default_embedder\.py "\$MEDIA_TYPE"', text, re.MULTILINE)
    assert re.search(r"^ENV .*VTSEARCH_SOLO_MEDIA_TYPE=\$\{MEDIA_TYPE\}", text, re.MULTILINE)
