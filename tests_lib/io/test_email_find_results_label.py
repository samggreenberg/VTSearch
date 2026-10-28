"""The email exporter keeps the good/bad split of a scored run (#5382).

A find-results payload whose ``hits`` hold both sides of the threshold - the
Find Results dialog's **Both** - labels every hit ``"good"`` or ``"bad"``
(``ResultsExporter.export_find_results``), and a ``--keep-negatives`` stream's
records carry the same key.  The email shows that ``label`` (a ``[good]`` /
``[bad]`` tag in plain text, a ``Label`` column in HTML) and leaves one-sided
output as it was.  An empty detector says "No hits.", which holds for
whichever side was sent.
"""

from __future__ import annotations

import email
from unittest.mock import MagicMock, patch

from vtscore.exporters.email_smtp import (
    EmailResultsExporter,
    _build_batch_html,
    _build_batch_plain,
    _build_html,
    _build_plain_text,
)


def _results(**hits_by_detector):
    return {
        "media_type": "audio",
        "detectors_run": len(hits_by_detector),
        "results": {
            name: {
                "detector_name": name,
                "threshold": 0.5,
                "total_hits": len(hits),
                "hits": hits,
                "negative_hits": [],
            }
            for name, hits in hits_by_detector.items()
        },
    }


_BOTH = [
    {"id": 1, "filename": "a.wav", "score": 0.9, "label": "good"},
    {"id": 2, "filename": "b.wav", "score": 0.2, "label": "bad"},
]
_ONE_SIDED = [{"id": 1, "filename": "a.wav", "score": 0.9}]

_BATCH = [
    ("owl", {"id": 1, "filename": "a.wav", "score": 0.9, "label": "good"}),
    ("owl", {"id": 2, "filename": "b.wav", "score": 0.2, "label": "bad"}),
]


class TestFindResultsLabel:
    def test_plain_text_tags_both_sides(self):
        text = _build_plain_text(_results(owl=_BOTH))
        assert "  [good] Clip #1: a.wav (score: 0.9)" in text
        assert "  [bad] Clip #2: b.wav (score: 0.2)" in text

    def test_html_has_a_label_column(self):
        html = _build_html(_results(owl=_BOTH))
        assert "<tr><th>Label</th><th>Clip</th><th>Filename</th><th>Score</th></tr>" in html
        assert "<tr><td>good</td><td>Clip #1</td>" in html
        assert "<tr><td>bad</td><td>Clip #2</td>" in html

    def test_one_sided_payload_is_unchanged(self):
        text = _build_plain_text(_results(owl=_ONE_SIDED))
        html = _build_html(_results(owl=_ONE_SIDED))
        assert "  Clip #1: a.wav (score: 0.9)" in text
        assert "[good]" not in text
        assert "<tr><th>Clip</th><th>Filename</th><th>Score</th></tr>" in html
        assert "Label" not in html

    def test_empty_detector_says_no_hits(self):
        """A Bad export sends the below-threshold side, so "No positive hits" would be wrong."""
        text = _build_plain_text(_results(owl=[]))
        html = _build_html(_results(owl=[]))
        assert "No hits." in text and "positive" not in text
        assert '<td colspan="3"><em>No hits.</em></td>' in html

    def test_empty_detector_spans_the_label_column_too(self):
        """The column is decided per payload, so an empty table in a Both export spans four."""
        html = _build_html(_results(owl=_BOTH, wren=[]))
        assert html.count("<th>Label</th>") == 2
        assert '<td colspan="4"><em>No hits.</em></td>' in html


class TestStreamedLabel:
    def test_keep_negatives_batch_tags_both_sides(self):
        text = _build_batch_plain("audio", _BATCH, 1, labelled=True)
        html = _build_batch_html("audio", _BATCH, 1, labelled=True)
        assert "  [good] Clip #1: a.wav (score: 0.9)" in text
        assert "  [bad] Clip #2: b.wav (score: 0.2)" in text
        assert "<tr><th>Label</th><th>Clip</th>" in html
        assert "<tr><td>bad</td><td>Clip #2</td>" in html

    def test_unlabelled_batch_is_unchanged(self):
        text = _build_batch_plain("audio", _BATCH, 1)
        html = _build_batch_html("audio", _BATCH, 1)
        assert "  Clip #1: a.wav (score: 0.9)" in text
        assert "[good]" not in text
        assert "Label" not in html

    @staticmethod
    def _stream_plain(keep_negatives):
        """The plain-text part of the one email a two-record stream sends."""
        server = MagicMock()
        smtp_cm = MagicMock()
        smtp_cm.__enter__.return_value = server
        with (
            patch("vtscore.exporters.email_smtp.smtplib.SMTP", return_value=smtp_cm),
            patch("vtscore.exporters.email_smtp._resolve_mx", return_value="mx.example.com"),
        ):
            EmailResultsExporter().export_cli_streaming(
                {"media_type": "audio", "detectors": [], "keep_negatives": keep_negatives},
                iter(_BATCH),
                {"from": "vt@example.com", "to": "user@example.com"},
            )
        message = email.message_from_string(server.sendmail.call_args.args[2])
        plain = next(p for p in message.walk() if p.get_content_type() == "text/plain")
        return plain.get_payload(decode=True).decode("utf-8")

    def test_header_keep_negatives_turns_the_tag_on(self):
        """Every streamed record carries ``label``; the header says whether both sides are in it."""
        assert "[bad] Clip #2" in self._stream_plain(True)
        assert "[good]" not in self._stream_plain(False)
