"""Tests for the centralised plugin inventory (`vtsearch.plugins.inventory`).

Backs the ``python app.py --list-plugins`` CLI: every plugin family must
show up with at least one entry (since the codebase ships built-ins for
each one), and the three output formats must round-trip cleanly.
"""

from __future__ import annotations

import json

import pytest

from vtsearch.plugins.inventory import (
    FAMILIES,
    format_json,
    format_names,
    format_plain,
    gather_plugins,
)


class TestGatherPlugins:
    def test_returns_all_known_families(self):
        inv = gather_plugins()
        assert set(inv.keys()) == set(FAMILIES)

    def test_each_family_has_entries(self):
        # The codebase ships built-ins for every family — if one comes back
        # empty something has broken auto-discovery.
        inv = gather_plugins()
        for family in FAMILIES:
            assert inv[family], f"family {family!r} is empty — discovery regressed?"

    def test_importers_include_server_folder(self):
        inv = gather_plugins()
        names = {entry.name for entry in inv["importers"]}
        assert "server_folder" in names

    def test_exporters_include_server_json_file(self):
        inv = gather_plugins()
        names = {entry.name for entry in inv["exporters"]}
        assert "server_json_file" in names

    def test_converters_carry_source_and_target_types(self):
        inv = gather_plugins()
        for entry in inv["converters"]:
            assert "source_type" in entry.extra
            assert "target_type" in entry.extra

    def test_embedders_carry_media_type(self):
        inv = gather_plugins()
        assert inv["embedders"], "no embedders discovered"
        for entry in inv["embedders"]:
            assert "media_type" in entry.extra


class TestFormatters:
    @pytest.fixture
    def inventory(self):
        return gather_plugins()

    def test_format_plain_is_human_readable(self, inventory):
        out = format_plain(inventory)
        # Every family label appears as a header line.
        assert "Dataset importers" in out
        assert "Results exporters" in out
        # And at least one known built-in name shows up.
        assert "server_folder" in out

    def test_format_json_is_valid_json(self, inventory):
        out = format_json(inventory)
        parsed = json.loads(out)
        assert set(parsed.keys()) == set(FAMILIES)
        # Each entry round-trips name + display_name + description.
        for entry in parsed["importers"]:
            assert "name" in entry and "display_name" in entry

    def test_format_names_global_uses_family_prefix(self, inventory):
        out = format_names(inventory)
        lines = [ln for ln in out.splitlines() if ln]
        assert all(":" in ln for ln in lines)
        assert "importers:server_folder" in lines

    def test_format_names_with_family_strips_prefix(self, inventory):
        out = format_names(inventory, family="importers")
        lines = [ln for ln in out.splitlines() if ln]
        assert all(":" not in ln for ln in lines)
        assert "server_folder" in lines

    def test_format_names_unknown_family_is_empty(self, inventory):
        assert format_names(inventory, family="not_a_family") == ""
