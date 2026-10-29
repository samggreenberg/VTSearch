"""``ResultsExporter.export_labelset_streaming`` and its five built-in implementations (#5285).

The CLI's ``--stream-results --as-labelset`` hands an exporter a lazy stream of
label entries, each naming its detector.  These tests pin the contract on the
base class (default refusal, the derived support flag, the shared batching
helper), the label-entry helpers the CLI builds entries with, and each
built-in's output - above all, that the file exporters' output reads back in
through their own label importers.
"""

from __future__ import annotations

import csv
import io
import json
from unittest.mock import MagicMock, patch

import pytest

from vtscore.datasets.labelset import LabelSet, elements_for_media, label_entry_for_export
from vtscore.exporters import get_exporter
from vtscore.exporters.base import ResultsExporter, iter_stream_batches
from vtscore.labels.importers.server_csv_file import _parse_csv_bytes
from vtscore.labels.json_format import extract_labels

_ORIGIN = {"importer": "server_folder", "params": {"path": "/srv/sounds"}}

_HEADER = {"media_type": "audio", "detectors": ["dog_bark", "cat_meow"], "keep_negatives": True}


def _elements():
    """Three entries in chunk order: two detectors, a good and a bad label."""
    yield {"md5": "a" * 32, "label": "good", "origin": _ORIGIN, "origin_name": "bark1.wav", "detector": "dog_bark"}
    yield {"md5": "b" * 32, "label": "bad", "origin": _ORIGIN, "origin_name": "bark2.wav", "detector": "dog_bark"}
    yield {"md5": "a" * 32, "label": "good", "origin": _ORIGIN, "origin_name": "bark1.wav", "detector": "cat_meow"}


class TestBaseContract:
    def test_default_refuses(self):
        class Plain(ResultsExporter):
            name = "plain_ls"
            fields = []

            def export_labelset(self, labelset, field_values):
                return {"message": "ok"}

        exporter = Plain()
        assert exporter.supports_labelset_streaming is False
        with pytest.raises(NotImplementedError, match="does not support streaming a labelset"):
            exporter.export_labelset_streaming(_HEADER, _elements(), {})

    def test_support_is_derived_from_the_override_and_inherited(self):
        class Streams(ResultsExporter):
            name = "streams_ls"
            fields = []

            def export_labelset_streaming(self, header, elements, field_values):
                return {"message": f"{sum(1 for _ in elements)}"}

        class Child(Streams):
            name = "streams_ls_child"

        assert Streams().supports_labelset_streaming is True
        assert Child().supports_labelset_streaming is True

    def test_it_is_not_a_payload_kind(self):
        """A delivery mode the CLI calls, so the GUI pickers never see it."""

        class Streams(ResultsExporter):
            name = "streams_ls_only"
            fields = []

            def export_labelset_streaming(self, header, elements, field_values):
                return {"message": ""}

        assert Streams().supported_payloads == frozenset()

    def test_the_streaming_built_ins_support_it(self):
        for name in ("gui", "server_json_file", "server_csv_file", "webhook", "email_smtp"):
            assert get_exporter(name).supports_labelset_streaming is True, name
        for name in ("open_url", "portable_detector"):
            assert get_exporter(name).supports_labelset_streaming is False, name


class TestIterStreamBatches:
    def test_groups_and_flushes_the_tail(self):
        assert list(iter_stream_batches(iter(range(5)), 2)) == [[0, 1], [2, 3], [4]]

    def test_an_exact_multiple_has_no_trailing_empty_batch(self):
        assert list(iter_stream_batches(iter(range(4)), 2)) == [[0, 1], [2, 3]]

    def test_an_empty_stream_still_yields_one_batch(self):
        assert list(iter_stream_batches(iter([]), 3)) == [[]]

    def test_it_is_lazy(self):
        """Nothing past the current batch is pulled - the bound the delivery streamers rely on."""
        pulled = []

        def items():
            for i in range(10):
                pulled.append(i)
                yield i

        first = next(iter_stream_batches(items(), 3))
        assert first == [0, 1, 2]
        assert pulled == [0, 1, 2]


class TestLabelEntryHelpers:
    def test_a_hit_converts_like_a_media(self):
        hit = {
            "id": 7,
            "filename": "bark1.wav",
            "category": "dogs",
            "score": 0.91,
            "md5": "a" * 32,
            "origin": _ORIGIN,
            "origin_name": "bark1.wav",
            "custom_metadata": {"asset_id": "A-1"},
            "label": "good",
        }
        (element,) = elements_for_media(hit, "good")
        assert element.to_dict() == {
            "md5": "a" * 32,
            "label": "good",
            "origin": _ORIGIN,
            "origin_name": "bark1.wav",
            "filename": "bark1.wav",
            "category": "dogs",
            "metadata": {"asset_id": "A-1"},
        }

    def test_a_dupe_set_expands_to_its_members(self):
        members = [
            {"md5": "1" * 32, "origin": _ORIGIN, "origin_name": "one.wav", "filename": "one.wav"},
            {"md5": "2" * 32, "origin": _ORIGIN, "origin_name": "two.wav", "filename": "two.wav"},
        ]
        hit = {"md5": "0" * 32, "origin": {"importer": "dupe_set", "members": members}, "label": "bad"}
        assert [e.md5 for e in elements_for_media(hit, "bad")] == ["1" * 32, "2" * 32]

    def test_label_entry_for_export_drops_the_head_measured_numbers(self):
        entry = {
            "md5": "a" * 32,
            "label": "good",
            "metadata": {"vt:provenance": {"flow": "find", "rank_at_vote": 3, "score_at_vote": 0.8}},
        }
        out = label_entry_for_export(entry)
        assert out["metadata"]["vt:provenance"] == {"flow": "find"}
        assert entry["metadata"]["vt:provenance"]["rank_at_vote"] == 3  # copied, not mutated


class TestServerJsonLabelsetStreaming:
    def test_writes_a_labels_file_the_json_importer_reads(self, tmp_path):
        out = tmp_path / "labels.json"
        res = get_exporter("server_json_file").export_labelset_streaming(_HEADER, _elements(), {"filepath": str(out)})

        data = json.loads(out.read_text())
        assert data["format"] == "vtsearch-labelset-stream/v1"
        assert data["detector_meta"] == {"media_type": "audio"}
        assert data["detectors"] == ["dog_bark", "cat_meow"]
        assert data["keep_negatives"] is True
        assert [e["detector"] for e in data["labels"]] == ["dog_bark", "dog_bark", "cat_meow"]  # chunk order kept
        assert [(e["md5"], e["label"]) for e in extract_labels(data)] == [
            ("a" * 32, "good"),
            ("b" * 32, "bad"),
            ("a" * 32, "good"),
        ]
        assert LabelSet.from_dict(data).detector_meta == {"media_type": "audio"}
        assert "3 label(s)" in res["message"] and "2 detector(s)" in res["message"]
        assert list(tmp_path.glob("*.tmp")) == []

    def test_an_empty_stream_is_still_a_valid_labels_file(self, tmp_path):
        out = tmp_path / "labels.json"
        get_exporter("server_json_file").export_labelset_streaming(_HEADER, iter([]), {"filepath": str(out)})
        assert extract_labels(json.loads(out.read_text())) == []

    def test_a_dataset_block_rides_along(self, tmp_path):
        out = tmp_path / "labels.json"
        dataset = {"name": "Sounds", "media_type": "audio", "category": "audio", "id": "ds-1"}
        get_exporter("server_json_file").export_labelset_streaming(
            {**_HEADER, "dataset": dataset}, _elements(), {"filepath": str(out)}
        )
        assert json.loads(out.read_text())["dataset"] == dataset


class TestServerCsvLabelsetStreaming:
    def test_writes_rows_the_csv_importer_reads(self, tmp_path):
        out = tmp_path / "labels.csv"
        res = get_exporter("server_csv_file").export_labelset_streaming(_HEADER, _elements(), {"filepath": str(out)})

        rows = list(csv.reader(io.StringIO(out.read_text())))
        assert rows[0] == ["detector", "label", "md5", "origin_name", "filename", "category", "origin"]
        assert [r[0] for r in rows[1:]] == ["dog_bark", "dog_bark", "cat_meow"]
        entries = _parse_csv_bytes(out.read_bytes())
        assert [(e["md5"], e["label"], e["origin"]) for e in entries] == [
            ("a" * 32, "good", _ORIGIN),
            ("b" * 32, "bad", _ORIGIN),
            ("a" * 32, "good", _ORIGIN),
        ]
        assert "3 label(s)" in res["message"]

    def test_formula_prefixed_names_survive_the_round_trip(self, tmp_path):
        out = tmp_path / "labels.csv"
        entry = {"md5": "c" * 32, "label": "good", "origin_name": "-take2.wav", "filename": "=mc2.wav", "detector": "d"}
        get_exporter("server_csv_file").export_labelset_streaming(_HEADER, iter([entry]), {"filepath": str(out)})
        (parsed,) = _parse_csv_bytes(out.read_bytes())
        assert parsed["origin_name"] == "-take2.wav"
        assert parsed["filename"] == "=mc2.wav"


class TestGuiLabelsetStreaming:
    def test_prints_every_entry_with_its_label_and_detector(self, capsys):
        res = get_exporter("gui").export_labelset_streaming(_HEADER, _elements(), {})
        out = capsys.readouterr().out
        assert "[good] dog_bark" in out
        assert "[bad] dog_bark" in out
        assert "[good] cat_meow" in out
        assert "bark2.wav" in out
        assert "3 label(s)" in res["message"]

    def test_an_empty_stream_says_so(self, capsys):
        get_exporter("gui").export_labelset_streaming(_HEADER, iter([]), {})
        assert "No labels to print." in capsys.readouterr().out


class TestWebhookLabelsetStreaming:
    def _mock_post(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        return resp

    def test_batches_entries_into_separate_posts(self):
        with patch("requests.Session.post", return_value=self._mock_post()) as mock_post:
            res = get_exporter("webhook").export_labelset_streaming(
                _HEADER, _elements(), {"url": "https://ex.com/hook", "batch_size": 2}
            )

        assert mock_post.call_count == 2
        first = mock_post.call_args_list[0].kwargs["json"]
        second = mock_post.call_args_list[1].kwargs["json"]
        assert first["format"] == "vtsearch-labelset-batch/v1"
        assert first["detectors"] == ["dog_bark", "cat_meow"]
        assert [e["md5"] for e in first["labels"]] == ["a" * 32, "b" * 32]
        assert second["batch_index"] == 1
        assert [e["detector"] for e in second["labels"]] == ["cat_meow"]
        assert "3 label(s)" in res["message"] and "2 batch(es)" in res["message"]

    def test_an_empty_run_still_posts_once(self):
        with patch("requests.Session.post", return_value=self._mock_post()) as mock_post:
            get_exporter("webhook").export_labelset_streaming(_HEADER, iter([]), {"url": "https://ex.com/hook"})
        assert mock_post.call_count == 1
        assert mock_post.call_args_list[0].kwargs["json"]["labels"] == []


class TestEmailLabelsetStreaming:
    def _patches(self):
        server = MagicMock()
        smtp_cm = MagicMock()
        smtp_cm.__enter__.return_value = server
        return (
            server,
            patch("vtscore.exporters.email_smtp.smtplib.SMTP", return_value=smtp_cm),
            patch("vtscore.exporters.email_smtp._resolve_mx", return_value="mx.example.com"),
        )

    def test_one_email_per_batch_listing_labels(self):
        server, smtp_p, mx_p = self._patches()
        with smtp_p, mx_p:
            res = get_exporter("email_smtp").export_labelset_streaming(
                _HEADER, _elements(), {"from": "vt@example.com", "to": "user@example.com", "batch_size": 2}
            )

        assert server.sendmail.call_count == 2
        first_message = server.sendmail.call_args_list[0].args[2]
        assert "VTSearch Labels: 2 label(s) (1 good, 1 bad) on audio dataset (part 1)" in first_message
        assert "3 label(s)" in res["message"] and "2 email(s)" in res["message"]

    def test_invalid_recipient_is_refused_before_anything_streams(self):
        pulled = []

        def elements():
            pulled.append(True)
            yield from _elements()

        with pytest.raises(ValueError, match="Recipient email address is invalid"):
            get_exporter("email_smtp").export_labelset_streaming(
                _HEADER, elements(), {"from": "vt@example.com", "to": "not-an-email"}
            )
        assert pulled == []
