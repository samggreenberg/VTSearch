"""Regrouping an importer's ``(output, chunk)`` stream per dataset (#4707).

``vtscore.cli._group_output_chunks`` turns the one stream the multi-output
hooks yield into one lazy chunk iterator per output, so the CLI can score each
dataset chunk by chunk without holding the others.  Pure: a fake stream in, the
grouping out.
"""

from __future__ import annotations

import pytest

from vtscore.cli import _group_output_chunks
from vtscore.datasets.importers.base import OutputSpec


def _chunk(*ids: int) -> dict[int, dict]:
    return {i: {"id": i} for i in ids}


class TestGrouping:
    def test_each_output_gets_its_own_chunks_in_order(self):
        audio, image = OutputSpec("audio"), OutputSpec("image")
        stream = iter([(audio, _chunk(1, 2)), (audio, _chunk(3)), (image, _chunk(1))])

        grouped = [(output, list(chunks)) for output, chunks in _group_output_chunks(stream, [audio, image], "imp")]

        assert [(o is audio, c) for o, c in grouped] == [
            (True, [_chunk(1, 2), _chunk(3)]),
            (False, [_chunk(1)]),
        ]
        assert grouped[1][0] is image

    def test_chunks_are_pulled_lazily(self):
        """The importer is read one item ahead of the consumer, never further."""
        audio, image = OutputSpec("audio"), OutputSpec("image")
        pulled: list[str] = []

        def stream():
            for output, chunk in [(audio, _chunk(1)), (audio, _chunk(2)), (image, _chunk(3)), (image, _chunk(4))]:
                pulled.append(f"{output.media_type}:{next(iter(chunk))}")
                yield output, chunk

        grouped = _group_output_chunks(stream(), [audio, image], "imp")
        output, chunks = next(grouped)
        assert output is audio
        assert pulled == ["audio:1"], "only the first chunk, to know which output starts"
        assert next(chunks) == _chunk(1)
        assert pulled == ["audio:1", "audio:2"], "one item of lookahead, to know the output goes on"
        assert next(chunks) == _chunk(2)
        assert pulled == ["audio:1", "audio:2", "image:3"], "the lookahead found the next output"
        assert list(chunks) == []
        output, chunks = next(grouped)
        assert output is image
        assert pulled == ["audio:1", "audio:2", "image:3"], "nothing more was read to start it"
        assert list(chunks) == [_chunk(3), _chunk(4)]

    def test_an_unfinished_output_is_drained_when_the_consumer_moves_on(self):
        audio, image = OutputSpec("audio"), OutputSpec("image")
        stream = iter([(audio, _chunk(1)), (audio, _chunk(2)), (audio, _chunk(3)), (image, _chunk(4))])

        grouped = _group_output_chunks(stream, [audio, image], "imp")
        output, chunks = next(grouped)
        assert next(chunks) == _chunk(1)
        output, chunks = next(grouped)
        assert output is image
        assert list(chunks) == [_chunk(4)]

    def test_empty_chunks_are_skipped_and_an_unyielded_output_comes_last_empty(self):
        audio, image, text = OutputSpec("audio"), OutputSpec("image"), OutputSpec("text")
        stream = iter([(image, _chunk(1)), (image, {}), (audio, {})])

        grouped = [
            (output, list(chunks)) for output, chunks in _group_output_chunks(stream, [audio, image, text], "imp")
        ]

        assert [o.media_type for o, _ in grouped] == ["image", "audio", "text"], "production order, then the rest"
        assert [c for _, c in grouped] == [[_chunk(1)], [], []]

    def test_a_copied_output_is_matched_by_equality(self):
        audio = OutputSpec("audio")
        stream = iter([(OutputSpec("audio"), _chunk(1))])

        grouped = _group_output_chunks(stream, [audio], "imp")
        output, chunks = next(grouped)

        assert output is audio, "the caller's own object comes back, not the importer's copy"
        assert list(chunks) == [_chunk(1)]
        assert list(grouped) == []

    def test_interleaving_is_refused(self):
        audio, image = OutputSpec("audio"), OutputSpec("image")
        stream = iter([(audio, _chunk(1)), (image, _chunk(2)), (audio, _chunk(3))])

        grouped = _group_output_chunks(stream, [audio, image], "imp")
        next(grouped)
        next(grouped)
        with pytest.raises(ValueError, match="'imp' yielded media for 'Audio' after moving on"):
            next(grouped)

    def test_an_output_the_importer_was_not_asked_for_is_refused(self):
        audio = OutputSpec("audio")
        stream = iter([(OutputSpec("image"), _chunk(1))])

        with pytest.raises(ValueError, match="'imp' yielded media for an output it was not asked for"):
            next(_group_output_chunks(stream, [audio], "imp"))

    def test_an_empty_stream_reports_every_output_empty(self):
        audio, image = OutputSpec("audio"), OutputSpec("image")
        grouped = [(o, list(c)) for o, c in _group_output_chunks(iter([]), [audio, image], "imp")]
        assert [(o is audio, c) for o, c in grouped] == [(True, []), (False, [])]
