"""The out-of-tree half of the derived-name contract.

``tests/core/test_plugin_derived_names_golden.py`` pins the plugins this
repo ships.  That is necessary but not sufficient: the same derivation names
*third-party* plugins, whose class names this repo has never seen, and a
derived ``name`` is a registry key that appears in entry-point configs, in
persisted settings, and in ``origin.params``.  A refactor that is
name-preserving for the ~20 concrete classes in this tree can still silently
rename somebody else's plugin on their install, with no error.

So this file samples the *input space* rather than the shipped classes: one
plausible third-party class name per historical family suffix, plus the
edge cases where the rules are least obvious (a class named exactly like a
suffix, a suffix appearing in the middle of a name, runs of capitals, family
bases whose names are deliberately **not** strippable).

Every row here is a promise to plugin authors outside this repository.
Changing one is a breaking change; adding one is free.
"""

from __future__ import annotations

import pytest

from vtscore.plugins import _PLUGIN_NAME_SUFFIXES, _default_plugin_name, _snake_case

#: ``(class name, derived plugin name)``.
SUFFIX_CORPUS: tuple[tuple[str, str], ...] = (
    # One plausible third-party class per historical family suffix.  These
    # all collapse to ``acme`` — that is the whole point of the suffix list.
    ("AcmeDataSourceImporter", "acme"),
    ("AcmeDatasetImporter", "acme"),
    ("AcmeLabelsetExporter", "acme"),
    ("AcmeResultsExporter", "acme"),
    ("AcmeLabelImporter", "acme"),
    ("AcmeLabelsetSource", "acme"),
    ("AcmeSeedImporter", "acme"),
    ("AcmeSettingsImporter", "acme"),
    ("AcmeSettingsExporter", "acme"),
    ("AcmeSettingsSource", "acme"),
    ("AcmeMediaConverter", "acme"),
    ("AcmeMediaSource", "acme"),
    ("AcmeImporter", "acme"),
    ("AcmeExporter", "acme"),
    ("AcmeSource", "acme"),
    ("AcmeConverter", "acme"),
    # Family *bases* whose names are deliberately not strippable suffixes.
    # ``SyncSource`` and ``ImporterBase`` are abstract bases a third party
    # may subclass directly, but neither is in the suffix list, so a name
    # ending in one falls through to the generic tail (or to no match at
    # all).  Making them strippable would rename these — see the module
    # docstring of ``vtscore/plugins/__init__.py``.
    ("AcmeSyncSource", "acme_sync"),
    ("AcmeImporterBase", "acme_importer_base"),
    ("AcmeSyncSourceImporter", "acme_sync_source"),
    # A class named exactly like a suffix keeps that suffix (the ``raw !=
    # suffix`` guard), then falls through to any *shorter* suffix that still
    # leaves something behind -- ``MediaSource`` keeps ``Media`` via the
    # generic ``Source``, and ``Importer`` matches nothing at all.
    ("Importer", "importer"),
    ("Exporter", "exporter"),
    ("Source", "source"),
    ("Converter", "converter"),
    ("DatasetImporter", "dataset"),
    ("ImporterBase", "importer_base"),
    ("SyncSource", "sync"),
    ("MediaSource", "media"),
    ("LabelsetExporter", "labelset"),
    # No suffix match: the whole class name is snake-cased verbatim.
    ("AcmePlugin", "acme_plugin"),
    ("Acme", "acme"),
    ("AcmeSourceish", "acme_sourceish"),
    ("AcmeExporterV2", "acme_exporter_v2"),
    # Capital runs and digits, where the snake-caser is least obvious.
    ("HTTPArchiveDatasetImporter", "http_archive"),
    ("S3Exporter", "s3"),
    ("OAuth2SettingsSource", "o_auth2"),
    ("XMLRPCImporter", "xmlrpc"),
    ("Audio2TextMediaConverter", "audio2_text"),
    # A suffix appearing somewhere other than the end: only the trailing
    # one is stripped, and only once.
    ("ImporterExporter", "importer"),
    ("SourceImporter", "source"),
    ("ExporterSource", "exporter"),
    ("DataSourceImporterImporter", "data_source_importer"),
    ("MediaConverterExporter", "media_converter"),
)


class TestThirdPartyNameDerivation:
    @pytest.mark.parametrize(("class_name", "expected"), SUFFIX_CORPUS)
    def test_derived_name_unchanged(self, class_name, expected):
        assert _default_plugin_name(type(class_name, (), {})) == expected

    def test_corpus_has_no_duplicate_rows(self):
        names = [row[0] for row in SUFFIX_CORPUS]
        assert len(names) == len(set(names))


class TestSuffixTableIsOrderIndependent:
    """The suffix scan's list order is not load-bearing.

    ``_default_plugin_name`` walks :data:`_PLUGIN_NAME_SUFFIXES` in order and
    strips the *first* match, so the table carries an implicit "longer /
    more-specific first" rule enforced only by a comment.  These tests prove
    that rule holds structurally, which in turn proves the ordered scan is
    equivalent to picking the **longest** matching suffix — the property any
    refactor of this table has to preserve.
    """

    def test_no_earlier_suffix_is_a_proper_suffix_of_a_later_one(self):
        """Two suffixes can both match a class name only when one is a
        suffix of the other; when that happens the longer one must come
        first, or the ordered scan would strip too little."""
        for i, earlier in enumerate(_PLUGIN_NAME_SUFFIXES):
            for later in _PLUGIN_NAME_SUFFIXES[i + 1 :]:
                assert not later.endswith(earlier), (
                    f"{earlier!r} precedes {later!r} but is a suffix of it: "
                    f"a class named ...{later} would be stripped to ...{later[: -len(earlier)]}"
                )

    def test_ordered_scan_equals_longest_match(self):
        """Exhaustive over the table itself: for every suffix, a class name
        built from it derives the same value under both rules."""
        for suffix in _PLUGIN_NAME_SUFFIXES:
            raw = "Acme" + suffix
            matches = [s for s in _PLUGIN_NAME_SUFFIXES if raw.endswith(s) and raw != s]
            longest = max(matches, key=len)
            assert _default_plugin_name(type(raw, (), {})) == _snake_case(raw[: -len(longest)])

    def test_no_duplicate_entries(self):
        assert len(_PLUGIN_NAME_SUFFIXES) == len(set(_PLUGIN_NAME_SUFFIXES))
