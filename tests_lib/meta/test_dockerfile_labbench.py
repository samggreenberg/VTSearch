"""``docker/Dockerfile.labbench`` built per media type (issue #4958).

The LabBench image installs only one media type's dependencies, so the media
types it can be built for are spelled in three places that must agree: one
``requirements/labbench-<type>.txt`` each (what the Dockerfile installs),
:data:`HIDDEN_BY_TYPE` in ``scripts/seed_labbench_settings.py`` (what the
seeded settings hide for that type), and the compose files' ``MEDIA_TYPE``
build args. A type in one but not another fails minutes into a build, or
worse, builds an image whose settings were seeded for nothing.

The seeded settings matter just as much and fail just as quietly: a seeded key
or plugin name with a typo hides nothing and locks nothing, and the container
serves anyway. So the seed script's output is pinned here against the plugin
inventory and the embedder registry, with no Docker daemon and no download.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

import vtscore.embedding
from vtscore.media import all_type_ids, embedders_for_type
from vtscore.plugins.inventory import gather_plugins

_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _ROOT / "docker" / "Dockerfile.labbench"
_SCRIPT = _ROOT / "scripts" / "seed_labbench_settings.py"


@pytest.fixture(scope="module")
def seed():
    spec = importlib.util.spec_from_file_location("seed_labbench_settings", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env(monkeypatch, tmp_path):
    """``VTSEARCH_DATA_DIR`` registered with monkeypatch, so the script's write is undone."""
    monkeypatch.setenv("VTSEARCH_DATA_DIR", str(tmp_path / "scratch"))


def _dockerfile_text() -> str:
    return re.sub(r"\\\n\s*", " ", _DOCKERFILE.read_text())


def _requirement_types() -> set[str]:
    return {p.stem.removeprefix("labbench-") for p in (_ROOT / "requirements").glob("labbench-*.txt")}


def _compose_files_building_labbench() -> list[Path]:
    found = []
    for path in sorted((_ROOT / "docker" / "compose").glob("docker-compose*.yml")):
        services = (yaml.safe_load(path.read_text()) or {}).get("services", {})
        if any((svc.get("build") or {}).get("dockerfile") == "docker/Dockerfile.labbench" for svc in services.values()):
            found.append(path)
    return found


# ---------------------------------------------------------------------------
# The supported media types agree everywhere
# ---------------------------------------------------------------------------


def test_requirements_and_seed_name_the_same_media_types(seed) -> None:
    assert _requirement_types() == set(seed.HIDDEN_BY_TYPE), (
        "requirements/labbench-<type>.txt and HIDDEN_BY_TYPE in scripts/seed_labbench_settings.py "
        "must name the same media types: a type with requirements but no seed entry fails the "
        "seed step at the end of the build, and an entry without requirements is unbuildable."
    )


def test_supported_types_are_registered_media_types(seed) -> None:
    assert set(seed.HIDDEN_BY_TYPE) <= set(all_type_ids())


@pytest.mark.parametrize("media_type", sorted(_requirement_types()))
def test_type_requirements_include_the_core(media_type: str) -> None:
    """Each type's file is the core plus that type, so installing it alone is the whole image."""
    text = (_ROOT / "requirements" / f"labbench-{media_type}.txt").read_text()
    assert re.search(r"^-r labbench\.txt\s*$", text, re.MULTILINE)


def test_the_image_and_audio_compose_files_build_labbench() -> None:
    names = {p.name for p in _compose_files_building_labbench()}
    assert {"docker-compose.labbench.yml", "docker-compose.labbench-audio.yml"} <= names


@pytest.mark.parametrize("compose", _compose_files_building_labbench(), ids=lambda p: p.name)
def test_compose_media_type_is_supported(compose: Path) -> None:
    """A typo here would only surface after the image's core pip install, minutes in."""
    for svc in yaml.safe_load(compose.read_text())["services"].values():
        media_type = svc["build"]["args"]["MEDIA_TYPE"]
        assert media_type in _requirement_types(), (
            f"{compose.name}: MEDIA_TYPE {media_type!r} has no requirements/labbench-{media_type}.txt"
        )


# ---------------------------------------------------------------------------
# The Dockerfile wires one MEDIA_TYPE through every step
# ---------------------------------------------------------------------------


def test_default_media_type_is_image() -> None:
    """``docker build -f docker/Dockerfile.labbench .`` still builds the SigLIP image."""
    match = re.search(r"^ARG MEDIA_TYPE=(\S*)\s*$", _dockerfile_text(), re.MULTILINE)
    assert match and match.group(1) == "image"


def test_dockerfile_installs_bakes_seeds_and_locks_the_same_media_type() -> None:
    """One ``MEDIA_TYPE`` feeds all four, so the image is one type end to end."""
    text = _dockerfile_text()
    assert re.search(r'-r "requirements/labbench-\$\{MEDIA_TYPE\}\.txt"', text)
    assert re.search(r'^RUN .*python -u scripts/preload_default_embedder\.py "\$MEDIA_TYPE"', text, re.MULTILINE)
    assert re.search(r'^RUN python scripts/seed_labbench_settings\.py "\$MEDIA_TYPE" /app/data', text, re.MULTILINE)
    assert re.search(r"^ENV .*VTSEARCH_SOLO_MEDIA_TYPE=\$\{MEDIA_TYPE\}", text, re.MULTILINE)


def test_core_requirements_install_before_media_type_is_declared() -> None:
    """Declared later, ``MEDIA_TYPE`` would rebuild the torch layer for every type."""
    text = _dockerfile_text()
    core = re.search(r"-r requirements/labbench\.txt", text)
    arg = re.search(r"^ARG MEDIA_TYPE", text, re.MULTILINE)
    assert core and arg and core.start() < arg.start()


def test_dockerfile_age_off_matches_the_seed(seed) -> None:
    match = re.search(r"VTSEARCH_DATASET_MAX_AGE_DAYS=(\d+)", _dockerfile_text())
    assert match and int(match.group(1)) == seed.DATASET_MAX_AGE_DAYS


# ---------------------------------------------------------------------------
# The seeded settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("media_type", sorted(_requirement_types()))
def test_hidden_plugins_are_registered(seed, media_type: str) -> None:
    """A misspelled name hides nothing, silently."""
    inventory = {family: {entry.name for entry in entries} for family, entries in gather_plugins().items()}
    for family, names in seed.server_settings(media_type)["hidden_plugins"].items():
        assert family in inventory, f"unknown plugin family {family!r}"
        missing = set(names) - inventory[family]
        assert not missing, f"{media_type}: no {family} plugin named {sorted(missing)}"


@pytest.mark.parametrize("media_type", sorted(_requirement_types()))
def test_every_image_hides_the_demo_importer(seed, media_type: str) -> None:
    assert "demo" in seed.server_settings(media_type)["hidden_plugins"]["importers"]


@pytest.mark.parametrize("media_type", sorted(_requirement_types()))
def test_seeded_embedder_lock_is_the_baked_embedder(seed, env, media_type: str) -> None:
    """The lock names what ``preload_default_embedder.py`` bakes, and an embedder of that type."""
    baked = vtscore.embedding.predict_embedders_to_preload(extra_media_types=[media_type])
    lock = seed.baked_embedder(media_type)
    assert lock == baked[0]
    assert lock in {e.name for e in embedders_for_type(media_type)}


def test_main_writes_both_files(seed, env, tmp_path) -> None:
    data = tmp_path / "data"
    assert seed.main(["audio", str(data)]) == 0
    server = json.loads((data / "settings.json").read_text())
    # The DefaultLoginProvider's user lives in the data dir itself.
    user = json.loads((data / "user_settings.json").read_text())
    assert not (data / "default").exists()
    assert server == seed.server_settings("audio")
    assert user == {"solo_embedder_per_media_type": {"audio": seed.baked_embedder("audio")}}


def test_main_rejects_a_missing_argument(seed, env, capsys) -> None:
    assert seed.main([]) == 2
    assert seed.main(["audio"]) == 2
    assert "MEDIA_TYPE" in capsys.readouterr().err


def test_main_rejects_an_unsupported_type_and_lists_the_supported(seed, env, tmp_path, capsys) -> None:
    assert seed.main(["video", str(tmp_path / "data")]) == 2
    err = capsys.readouterr().err
    assert "'video'" in err
    assert "audio" in err and "image" in err
    assert not (tmp_path / "data").exists()
