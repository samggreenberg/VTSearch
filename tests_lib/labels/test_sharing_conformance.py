"""Conformance suite for the sharing capability of ``LabelsetSource`` (#5574).

The contract tests run against **every registered source whose
``supports_sharing`` is true** - today the built-in ``shared_folder`` - so a
future sharing source is held to the same contract the moment it registers.
Each such source needs a driver in :data:`DRIVERS`: the field values that point
it at a scratch store, and a way to write to a container as another identity,
the way a misbehaving recipient would.  A sharing source with no driver fails
:func:`test_every_sharing_source_has_a_driver` rather than going untested.

The derived capability flags, the wrappers' argument and return checks,
:func:`~vtscore.labels.sharing.active_sharing_source`, the digest, and the
``shared_folder`` specifics that are not part of the contract are tested
below the parametrized suite.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from vtscore.config import register_setting_getter
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.labels.sharing import (
    RECIPIENT_KINDS,
    SHARED_VIA_DIRECT,
    SHARED_VIA_OWNER,
    SHARED_VIA_PUBLIC,
    Manifest,
    RecipientSuggestion,
    RemoteNotFound,
    RemotePermissionDenied,
    SharedContainer,
    SharedContainerSummary,
    SharingError,
    SharingUnsupported,
    active_sharing_source,
    build_manifest,
    labelset_digest,
    share_payload,
)
from vtscore.labels.sources import get_labelset_source, list_labelset_sources
from vtscore.labels.sources.base import REQUIRED_SHARING_HOOKS, LabelsetSource, PluginField
from vtscore.labels.sources.shared_folder import simulate_foreign_modification
from vtscore.state.current_user import thread_user

OWNER, RECIPIENT, STRANGER = "alice", "bob", "carol"
MD5_A, MD5_B, MD5_C, MD5_D = ("a" * 32, "b" * 32, "c" * 32, "d" * 32)


# ---------------------------------------------------------------------------
# Drivers: how the suite points each sharing source at a scratch store
# ---------------------------------------------------------------------------


class SharedFolderDriver:
    """The built-in ``shared_folder`` source: a fresh directory per test."""

    def field_values(self, tmp_path: Path) -> dict[str, str]:
        return {"folder": str(tmp_path / "shared")}

    def foreign_modify(self, field_values: dict[str, str], remote_ref: str, **kwargs: Any) -> None:
        simulate_foreign_modification(field_values["folder"], remote_ref, **kwargs)


#: One driver per registered sharing source, keyed by source name.
DRIVERS: dict[str, Any] = {"shared_folder": SharedFolderDriver()}


def _sharing_source_names() -> list[str]:
    return sorted(s.name for s in list_labelset_sources() if s.supports_sharing)


def test_every_sharing_source_has_a_driver():
    names = _sharing_source_names()
    assert "shared_folder" in names
    missing = [n for n in names if n not in DRIVERS]
    assert not missing, f"Sharing sources with no conformance driver (add one to DRIVERS): {missing}"


@pytest.fixture(params=[n for n in _sharing_source_names() if n in DRIVERS])
def sharing(request, tmp_path):
    """``(source, field_values, driver)`` for one registered sharing source."""
    driver = DRIVERS[request.param]
    return get_labelset_source(request.param), driver.field_values(tmp_path), driver


def _detector(name: str = "dogs", labels: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A detector file dict, with the measured numbers an export must leave behind."""
    if labels is None:
        labels = [
            {
                "md5": MD5_A,
                "label": "good",
                "origin": {"importer": "folder", "params": {"path": "/photos"}},
                "origin_name": "a.jpg",
                "metadata": {"vt:provenance": {"v": 1, "surface": "sort", "rank_at_vote": 3, "score_at_vote": 0.91}},
            },
            {"md5": MD5_B, "label": "bad", "origin_name": "b.jpg"},
            {"md5": MD5_C, "label": "good", "region_box": [0.1, 0.2, 0.5, 0.6]},
        ]
    return {
        "name": name,
        "media_type": "image",
        "input_spec": {"clipper": "whole_image", "params": {}},
        "threshold": 0.42,
        "labelset": {"labels": labels},
    }


def _texts(elements) -> list[str]:
    return sorted(json.dumps(el.to_dict(), sort_keys=True) for el in elements)


def _create(source, fv, data=None) -> tuple[str, LabelSet, Any]:
    labelset, meta = share_payload(data or _detector())
    with thread_user(OWNER):
        ref = source.share_create(labelset, meta, fv)
    return ref, labelset, meta


def _refresh(source, fv, ref, who=OWNER) -> SharedContainer:
    with thread_user(who):
        return source.share_refresh(ref, fv)


def _received(source, fv, who) -> list[SharedContainerSummary]:
    with thread_user(who):
        return source.share_list_received(fv)


def _mine(source, fv, who) -> list[SharedContainerSummary]:
    with thread_user(who):
        return source.share_list_mine(fv)


# ---------------------------------------------------------------------------
# The contract, for every sharing source
# ---------------------------------------------------------------------------


class TestCreateAndRefresh:
    def test_round_trips_records_identities_owner_name_meta_and_manifest(self, sharing):
        source, fv, _ = sharing
        ref, labelset, meta = _create(source, fv)
        container = _refresh(source, fv, ref)

        assert isinstance(ref, str) and ref
        assert container.remote_ref == ref
        assert container.owner == OWNER
        assert container.name == "dogs"
        assert container.detector_meta == {
            "media_type": "image",
            "input_spec": {"clipper": "whole_image", "params": {}},
        }
        assert _texts(r.element for r in container.records) == _texts(labelset.elements)
        assert {(r.created_by, r.last_modified_by) for r in container.records} == {(OWNER, OWNER)}
        assert container.manifest == meta.manifest
        assert container.manifest_modified_by == OWNER
        assert build_manifest(container.as_labelset()) == container.manifest
        assert container.recipients == ()
        assert container.public is False

    def test_the_export_payload_carries_no_threshold_or_score(self, sharing):
        source, fv, _ = sharing
        labelset, meta = share_payload(_detector())
        assert "threshold" not in meta.detector_meta
        assert "threshold" not in (labelset.detector_meta or {})

        with thread_user(OWNER):
            ref = source.share_create(labelset, meta, fv)
        container = _refresh(source, fv, ref)

        assert "threshold" not in container.detector_meta
        for record in container.records:
            provenance = (record.element.metadata or {}).get("vt:provenance") or {}
            assert "rank_at_vote" not in provenance
            assert "score_at_vote" not in provenance
        kept = next(r for r in container.records if r.element.md5 == MD5_A)
        assert kept.element.metadata == {"vt:provenance": {"v": 1, "surface": "sort"}}

    def test_an_unknown_ref_is_not_found(self, sharing):
        source, fv, _ = sharing
        _create(source, fv)
        with pytest.raises(RemoteNotFound):
            _refresh(source, fv, "no-such-container")


class TestSave:
    def test_save_rewrites_the_container_in_place(self, sharing):
        source, fv, _ = sharing
        ref, _, _ = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT], fv)

        changed = _detector(
            name="dogs v2",
            labels=[
                {"md5": MD5_A, "label": "good", "origin_name": "a.jpg"},
                {"md5": MD5_B, "label": "good", "origin_name": "b.jpg"},
                {"md5": MD5_D, "label": "bad"},
            ],
        )
        labelset, meta = share_payload(changed)
        with thread_user(OWNER):
            source.share_save(ref, labelset, meta, fv)
        container = _refresh(source, fv, ref)

        assert container.remote_ref == ref
        assert container.name == "dogs v2"
        assert _texts(r.element for r in container.records) == _texts(labelset.elements)
        assert {r.last_modified_by for r in container.records} == {OWNER}
        assert container.manifest == meta.manifest
        assert container.manifest_modified_by == OWNER
        assert container.recipients == (RECIPIENT,)
        assert [s.remote_ref for s in _mine(source, fv, OWNER)] == [ref]


class TestListingsFollowTheCaller:
    def test_mine_is_the_callers_own(self, sharing):
        source, fv, _ = sharing
        ref, _, _ = _create(source, fv)

        mine = _mine(source, fv, OWNER)
        assert [(s.remote_ref, s.name, s.owner, s.shared_via) for s in mine] == [(ref, "dogs", OWNER, SHARED_VIA_OWNER)]
        assert mine[0].media_type == "image"
        assert mine[0].record_count == 3
        assert _received(source, fv, OWNER) == []
        assert _mine(source, fv, RECIPIENT) == []
        assert _received(source, fv, RECIPIENT) == []

    def test_an_added_recipient_receives_it(self, sharing):
        source, fv, _ = sharing
        ref, _, _ = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT], fv)

        received = _received(source, fv, RECIPIENT)
        assert [(s.remote_ref, s.owner, s.shared_via) for s in received] == [(ref, OWNER, SHARED_VIA_DIRECT)]
        assert _received(source, fv, STRANGER) == []
        assert _refresh(source, fv, ref).recipients == (RECIPIENT,)
        assert _refresh(source, fv, ref, who=RECIPIENT).owner == OWNER

    def test_a_removed_recipient_no_longer_does(self, sharing):
        source, fv, _ = sharing
        ref, _, _ = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT, STRANGER], fv)
            source.share_remove_recipients(ref, [RECIPIENT], fv)

        assert _received(source, fv, RECIPIENT) == []
        assert [s.remote_ref for s in _received(source, fv, STRANGER)] == [ref]
        assert _refresh(source, fv, ref).recipients == (STRANGER,)

    def test_public_reaches_everyone_until_it_is_turned_off(self, sharing):
        source, fv, _ = sharing
        ref, _, _ = _create(source, fv)
        with thread_user(OWNER):
            source.share_set_public(ref, True, fv)

        assert [(s.remote_ref, s.shared_via) for s in _received(source, fv, STRANGER)] == [(ref, SHARED_VIA_PUBLIC)]
        assert _refresh(source, fv, ref, who=STRANGER).public is True

        with thread_user(OWNER):
            source.share_set_public(ref, False, fv)
        assert _received(source, fv, STRANGER) == []

    def test_archive_hides_the_container(self, sharing):
        source, fv, _ = sharing
        ref, _, _ = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT], fv)
            source.share_archive(ref, fv)

        assert _mine(source, fv, OWNER) == []
        assert _received(source, fv, RECIPIENT) == []
        with pytest.raises(RemoteNotFound):
            _refresh(source, fv, ref)

    def test_a_private_container_is_closed_to_a_stranger(self, sharing):
        source, fv, _ = sharing
        ref, _, _ = _create(source, fv)
        with pytest.raises((RemotePermissionDenied, RemoteNotFound)):
            _refresh(source, fv, ref, who=STRANGER)

    @pytest.mark.parametrize(
        "change",
        [
            lambda s, ref, fv: s.share_add_recipients(ref, [STRANGER], fv),
            lambda s, ref, fv: s.share_remove_recipients(ref, [RECIPIENT], fv),
            lambda s, ref, fv: s.share_set_public(ref, True, fv),
            lambda s, ref, fv: s.share_archive(ref, fv),
        ],
        ids=["add", "remove", "public", "archive"],
    )
    def test_only_the_owner_changes_who_it_is_shared_with(self, sharing, change):
        source, fv, _ = sharing
        ref, _, _ = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT], fv)

        with thread_user(RECIPIENT), pytest.raises(RemotePermissionDenied):
            change(source, ref, fv)
        assert _refresh(source, fv, ref).recipients == (RECIPIENT,)


class TestForeignModifications:
    """What a recipient's writes look like to the owner's refresh."""

    @pytest.fixture
    def shared(self, sharing):
        source, fv, driver = sharing
        ref, _, meta = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT], fv)
        return source, fv, driver, ref, meta

    def test_a_relabel_carries_the_recipients_identity(self, shared):
        source, fv, driver, ref, meta = shared
        driver.foreign_modify(fv, ref, actor=RECIPIENT, relabel=[MD5_A])
        container = _refresh(source, fv, ref)

        changed = next(r for r in container.records if r.element.md5 == MD5_A)
        assert changed.element.label == "bad"
        assert changed.last_modified_by == RECIPIENT != container.owner
        assert {r.last_modified_by for r in container.records if r.element.md5 != MD5_A} == {OWNER}
        assert container.manifest == meta.manifest
        assert build_manifest(container.as_labelset()) != container.manifest

    def test_an_added_record_is_created_by_the_recipient(self, shared):
        source, fv, driver, ref, _ = shared
        driver.foreign_modify(fv, ref, actor=RECIPIENT, add=[LabeledElement(md5=MD5_D, label="good")])
        container = _refresh(source, fv, ref)

        added = next(r for r in container.records if r.element.md5 == MD5_D)
        assert (added.created_by, added.last_modified_by) == (RECIPIENT, RECIPIENT)
        assert container.manifest is not None
        assert container.manifest.record_count != len(container.records)

    def test_a_deletion_shows_only_in_the_manifest(self, shared):
        source, fv, driver, ref, _ = shared
        driver.foreign_modify(fv, ref, actor=RECIPIENT, delete=[MD5_B])
        container = _refresh(source, fv, ref)

        assert {(r.created_by, r.last_modified_by) for r in container.records} == {(OWNER, OWNER)}
        assert container.manifest is not None
        assert container.manifest.record_count == 3
        assert len(container.records) == 2
        assert build_manifest(container.as_labelset()) != container.manifest

    def test_a_rewritten_manifest_names_its_writer(self, shared):
        source, fv, driver, ref, _ = shared
        driver.foreign_modify(fv, ref, actor=RECIPIENT, delete=[MD5_B], touch_manifest=True)
        container = _refresh(source, fv, ref)

        assert build_manifest(container.as_labelset()) == container.manifest
        assert container.manifest_modified_by == RECIPIENT != container.owner


class TestRecipientSearch:
    def test_suggestions_are_typed(self, sharing):
        source, fv, _ = sharing
        if not source.supports_recipient_search:
            pytest.skip(f"{source.name} does not search recipients")
        ref, _, _ = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT], fv)
            suggestions = source.share_search_recipients(RECIPIENT[:2], fv)
            assert source.share_search_recipients("   ", fv) == []

        assert all(isinstance(s, RecipientSuggestion) and s.kind in RECIPIENT_KINDS for s in suggestions)
        assert OWNER not in [s.id for s in suggestions]


# ---------------------------------------------------------------------------
# Derived capability flags and the wrappers
# ---------------------------------------------------------------------------


def _source_class(name: str, hooks: dict[str, Any], *, plain: bool = False) -> type[LabelsetSource]:
    """A source class overriding *hooks* (and plain ``_do_load`` / ``_do_save`` when *plain*)."""
    body: dict[str, Any] = {
        "name": name,
        "display_name": name,
        "description": name,
        "fields": [PluginField(key="where", label="Where", field_type="text", required=False)],
        **hooks,
    }
    if plain:
        body["_do_load"] = lambda self, fv: [{"md5": MD5_A, "label": "good"}]
        body["_do_save"] = lambda self, labelset, fv: None
    return type(f"{name.title()}LabelsetSource", (LabelsetSource,), body)


def _required_hooks() -> dict[str, Any]:
    """Every required sharing hook, as stubs that return well-typed values."""
    hooks: dict[str, Any] = {h: (lambda self, *args: None) for h in REQUIRED_SHARING_HOOKS}
    hooks["_do_share_create"] = lambda self, labelset, meta, fv: "ref-1"
    hooks["_do_share_refresh"] = lambda self, ref, fv: SharedContainer(remote_ref=ref, name="n", owner="o")
    hooks["_do_share_list_mine"] = lambda self, fv: []
    hooks["_do_share_list_received"] = lambda self, fv: []
    return hooks


class TestDerivedFlags:
    def test_the_built_ins(self):
        plain, folder = get_labelset_source("server_json_file"), get_labelset_source("shared_folder")
        assert (plain.supports_sync, plain.supports_sharing, plain.supports_recipient_search) == (True, False, False)
        assert (folder.supports_sync, folder.supports_sharing, folder.supports_recipient_search) == (False, True, True)

    def test_the_flags_ride_on_to_dict(self):
        entry = get_labelset_source("shared_folder").to_dict()
        assert (entry["supports_sync"], entry["supports_sharing"], entry["supports_recipient_search"]) == (
            False,
            True,
            True,
        )

    def test_a_source_without_the_optional_hook_cannot_search(self):
        source = _source_class("nosearch", _required_hooks())()
        assert source.supports_sharing is True
        assert source.supports_recipient_search is False
        assert source.supports_sync is False
        with pytest.raises(SharingUnsupported):
            source.share_search_recipients("bo", {})

    def test_a_source_with_no_sharing_hooks_is_a_plain_source(self):
        source = _source_class("plainonly", {}, plain=True)()
        assert source.supports_sharing is False
        assert source.supports_recipient_search is False
        assert source.supports_sync is True
        assert [e.md5 for e in source.load_full({}).elements] == [MD5_A]
        source.save(LabelSet([LabeledElement(md5=MD5_A, label="good")]), {})
        labelset, meta = share_payload(_detector())
        with pytest.raises(SharingUnsupported):
            source.share_create(labelset, meta, {})
        with pytest.raises(NotImplementedError):
            source.share_list_received({})

    def test_a_save_only_source_is_not_a_sync_target(self):
        source = _source_class("saveonly", {"_do_save": lambda self, labelset, fv: None})()
        assert source.supports_sync is False

    def test_a_partial_implementation_warns_and_does_not_share(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vtscore.labels.sources.base"):
            cls = _source_class("partial", {"_do_share_create": lambda self, labelset, meta, fv: "ref"})
        assert "_do_share_refresh" in caplog.text
        source = cls()
        assert source.supports_sharing is False
        labelset, meta = share_payload(_detector())
        with pytest.raises(SharingUnsupported):
            source.share_create(labelset, meta, {})

    def test_search_without_the_required_hooks_is_unsupported(self, caplog):
        with caplog.at_level(logging.WARNING, logger="vtscore.labels.sources.base"):
            source = _source_class("searchonly", {"_do_share_search_recipients": lambda self, q, fv: []})()
        assert source.supports_recipient_search is False
        with pytest.raises(SharingUnsupported):
            source.share_search_recipients("bo", {})


class TestWrapperChecks:
    def test_create_must_return_a_ref(self):
        source = _source_class("badref", {**_required_hooks(), "_do_share_create": lambda *a: ""})()
        labelset, meta = share_payload(_detector())
        with pytest.raises(SharingError, match="remote_ref"):
            source.share_create(labelset, meta, {})

    def test_refresh_must_return_a_container(self):
        source = _source_class("baddict", {**_required_hooks(), "_do_share_refresh": lambda self, ref, fv: {}})()
        with pytest.raises(SharingError, match="SharedContainer"):
            source.share_refresh("ref-1", {})

    def test_listings_must_hold_summaries(self):
        source = _source_class("badlist", {**_required_hooks(), "_do_share_list_mine": lambda self, fv: [{}]})()
        with pytest.raises(SharingError, match="SharedContainerSummary"):
            source.share_list_mine({})

    def test_ids_must_be_a_collection_of_strings(self):
        seen: list[list[str]] = []
        hooks = {**_required_hooks(), "_do_share_add_recipients": lambda self, ref, ids, fv: seen.append(ids)}
        source = _source_class("ids", hooks)()
        with pytest.raises(TypeError):
            source.share_add_recipients("ref-1", "bob", {})
        with pytest.raises(ValueError):
            source.share_add_recipients("ref-1", ["bob", ""], {})
        source.share_add_recipients("ref-1", ["bob", "Bob", "bob"], {})
        assert seen == [["bob", "Bob"]]

    def test_a_ref_must_be_a_non_empty_string(self):
        source = _source_class("noref", _required_hooks())()
        with pytest.raises(ValueError):
            source.share_refresh("", {})

    def test_meta_must_be_share_meta(self):
        source = _source_class("nometa", _required_hooks())()
        with pytest.raises(TypeError):
            source.share_create(LabelSet(), {"name": "x"}, {})  # type: ignore[arg-type]

    def test_field_values_are_normalized_before_the_hook(self):
        seen: list[dict[str, Any]] = []
        hooks = {**_required_hooks(), "_do_share_list_mine": lambda self, fv: seen.append(fv) or []}
        source = _source_class("normalized", hooks)()
        source.share_list_mine({"where": "  over there  "})
        assert seen == [{"where": "over there"}]


# ---------------------------------------------------------------------------
# active_sharing_source
# ---------------------------------------------------------------------------


class TestActiveSharingSource:
    def _set(self, cfg):
        register_setting_getter("detector_sharing_source", lambda: cfg)

    def test_off_by_default(self):
        assert active_sharing_source() is None

    def test_a_sharing_source_with_its_fixed_values(self, tmp_path):
        self._set({"source_name": "shared_folder", "field_values": {"folder": str(tmp_path)}})
        active = active_sharing_source()
        assert active is not None
        assert active.source_name == "shared_folder"
        assert active.source is get_labelset_source("shared_folder")
        assert dict(active.field_values) == {"folder": str(tmp_path)}
        assert active.missing_fields == ()

    def test_blank_values_are_not_fixed(self):
        self._set({"source_name": "shared_folder", "field_values": {"folder": "  "}})
        active = active_sharing_source()
        assert active is not None
        assert dict(active.field_values) == {}
        assert active.missing_fields == ("folder",)

    @pytest.mark.parametrize("cfg", [None, {}, {"source_name": ""}, {"source_name": "none"}, "shared_folder"])
    def test_off_without_a_warning(self, cfg, caplog):
        self._set(cfg)
        with caplog.at_level(logging.WARNING, logger="vtscore.labels.sharing"):
            assert active_sharing_source() is None
        assert caplog.records == []

    @pytest.mark.parametrize(
        ("name", "says"), [("no_such_source", "unknown"), ("server_json_file", "does not implement sharing")]
    )
    def test_off_with_one_warning(self, name, says, caplog):
        self._set({"source_name": name, "field_values": {}})
        with caplog.at_level(logging.WARNING, logger="vtscore.labels.sharing"):
            assert active_sharing_source() is None
            assert active_sharing_source() is None
        warnings = [r for r in caplog.records if r.name == "vtscore.labels.sharing"]
        assert len(warnings) == 1
        assert says in warnings[0].getMessage()
        assert name in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# The digest and the manifest
# ---------------------------------------------------------------------------


class TestDigest:
    def _elements(self):
        return [LabeledElement(md5=MD5_A, label="good"), LabeledElement(md5=MD5_B, label="bad")]

    def test_order_does_not_matter(self):
        a, b = self._elements()
        assert labelset_digest(LabelSet([a, b])) == labelset_digest(LabelSet([b, a]))

    def test_content_does(self):
        a, b = self._elements()
        relabelled = LabeledElement(md5=MD5_B, label="good")
        assert labelset_digest(LabelSet([a, b])) != labelset_digest(LabelSet([a, relabelled]))
        assert labelset_digest(LabelSet([a, b])) != labelset_digest(LabelSet([a, b, b]))
        assert labelset_digest(LabelSet([a], detector_meta={"media_type": "image"})) != labelset_digest(LabelSet([a]))

    def test_no_meta_is_empty_meta(self):
        a, _ = self._elements()
        assert labelset_digest(LabelSet([a], detector_meta={})) == labelset_digest(LabelSet([a]))

    def test_the_sync_engine_uses_the_same_digest(self):
        from vtscore.labels.sync import _digest

        labelset = LabelSet(self._elements(), detector_meta={"media_type": "image"})
        assert _digest(labelset) == labelset_digest(labelset)

    def test_build_manifest(self):
        labelset = LabelSet(self._elements())
        assert build_manifest(labelset) == Manifest(record_count=2, digest=labelset_digest(labelset))

    @pytest.mark.parametrize(
        "raw", [None, [], {"record_count": "2", "digest": "x"}, {"record_count": True, "digest": "x"}, {"digest": "x"}]
    )
    def test_a_malformed_manifest_reads_as_none(self, raw):
        assert Manifest.from_dict(raw) is None

    def test_a_manifest_round_trips(self):
        manifest = build_manifest(LabelSet(self._elements()))
        assert Manifest.from_dict(manifest.to_dict()) == manifest

    def test_value_types_check_their_enums(self):
        with pytest.raises(ValueError):
            RecipientSuggestion(id="x", display_name="x", kind="team")
        with pytest.raises(ValueError):
            SharedContainerSummary(
                remote_ref="r", name="n", owner="o", media_type="", record_count=0, updated_at=0.0, shared_via="friend"
            )


# ---------------------------------------------------------------------------
# shared_folder specifics (not part of the contract)
# ---------------------------------------------------------------------------


class TestSharedFolder:
    @pytest.fixture
    def folder(self, tmp_path):
        return get_labelset_source("shared_folder"), {"folder": str(tmp_path / "shared")}

    def test_one_json_file_per_container(self, folder):
        source, fv = folder
        ref, _, _ = _create(source, fv)
        path = Path(fv["folder"]) / f"{ref}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["owner"] == OWNER
        assert doc["manifest_modified_by"] == OWNER
        assert sorted(p.name for p in Path(fv["folder"]).glob("*.json")) == [f"{ref}.json"]

    def test_a_crafted_ref_names_no_container(self, folder):
        source, fv = folder
        _create(source, fv)
        with pytest.raises(RemoteNotFound):
            _refresh(source, fv, "../shared/escape")

    def test_a_recipient_can_still_write_and_is_stamped(self, folder):
        source, fv = folder
        ref, _, meta = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT], fv)
        labelset, their_meta = share_payload(_detector(labels=[{"md5": MD5_A, "label": "bad", "origin_name": "a.jpg"}]))
        with thread_user(RECIPIENT):
            source.share_save(ref, labelset, their_meta, fv)
        container = _refresh(source, fv, ref)
        assert container.manifest_modified_by == RECIPIENT
        assert [(r.element.md5, r.last_modified_by) for r in container.records] == [(MD5_A, RECIPIENT)]

    def test_a_stranger_cannot_write(self, folder):
        source, fv = folder
        ref, labelset, meta = _create(source, fv)
        with thread_user(STRANGER), pytest.raises(RemotePermissionDenied):
            source.share_save(ref, labelset, meta, fv)

    def test_an_unchanged_record_keeps_its_identities_through_an_owner_save(self, folder):
        source, fv = folder
        ref, labelset, meta = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT], fv)
        simulate_foreign_modification(fv["folder"], ref, actor=RECIPIENT, add=[LabeledElement(md5=MD5_D, label="bad")])
        before = {r.element.md5: r for r in _refresh(source, fv, ref).records}

        with thread_user(OWNER):
            source.share_save(ref, labelset, meta, fv)
        after = {r.element.md5: r for r in _refresh(source, fv, ref).records}

        assert set(after) == {MD5_A, MD5_B, MD5_C}
        assert after[MD5_A] == before[MD5_A]

    def test_the_owner_is_never_their_own_recipient(self, folder):
        source, fv = folder
        ref, _, _ = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [OWNER, RECIPIENT, RECIPIENT], fv)
        assert _refresh(source, fv, ref).recipients == (RECIPIENT,)

    def test_search_suggests_identities_seen_in_the_folder(self, folder):
        source, fv = folder
        ref, _, _ = _create(source, fv)
        with thread_user(OWNER):
            source.share_add_recipients(ref, [RECIPIENT, "robert"], fv)
        with thread_user(STRANGER):
            assert [s.id for s in source.share_search_recipients("B", fv)] == [RECIPIENT, "robert"]
            assert [s.id for s in source.share_search_recipients("ali", fv)] == [OWNER]
            assert source.share_search_recipients("carol", fv) == []

    def test_a_file_that_does_not_parse_is_skipped_in_listings(self, folder):
        source, fv = folder
        ref, _, _ = _create(source, fv)
        (Path(fv["folder"]) / "junk.json").write_text("{not json", encoding="utf-8")
        (Path(fv["folder"]) / "other.json").write_text('{"labels": []}', encoding="utf-8")
        assert [s.remote_ref for s in _mine(source, fv, OWNER)] == [ref]

    def test_an_empty_folder_lists_nothing(self, folder):
        source, fv = folder
        assert _mine(source, fv, OWNER) == []
        assert _received(source, fv, OWNER) == []

    def test_a_foreign_modification_of_a_missing_container_is_not_found(self, folder):
        _, fv = folder
        with pytest.raises(RemoteNotFound):
            simulate_foreign_modification(fv["folder"], "missing", actor=RECIPIENT, delete=[MD5_A])
