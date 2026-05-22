"""Tests for ``vtscore.cli._renumber_chunks``.

Every chunked importer/loader yields chunks whose media IDs restart at
1.  The in-process consumer ``consume_chunks_into`` renumbers them as it
drains; the CLI pipeline used to score and merge them as-is, which
collided ``id`` values across chunks in the merged hit lists.  This
helper renumbers at the CLI boundary so the exported ``id`` field is
globally unique regardless of which chunked importer is feeding it.
"""

from __future__ import annotations

from vtscore.cli import _renumber_chunks


class TestRenumberChunks:
    def test_single_chunk_starts_at_one(self):
        chunk = {1: {"id": 1, "name": "a"}, 2: {"id": 2, "name": "b"}}
        out = list(_renumber_chunks(iter([chunk])))
        assert len(out) == 1
        assert sorted(out[0].keys()) == [1, 2]
        assert out[0][1]["name"] == "a"
        assert out[0][2]["name"] == "b"

    def test_two_chunks_get_unique_ids(self):
        # Both chunks re-use ids 1..2 (the standard chunked-importer
        # convention).  After renumbering, ids must be 1..4 globally.
        chunk_a = {1: {"id": 1, "name": "a1"}, 2: {"id": 2, "name": "a2"}}
        chunk_b = {1: {"id": 1, "name": "b1"}, 2: {"id": 2, "name": "b2"}}
        out = list(_renumber_chunks(iter([chunk_a, chunk_b])))
        assert len(out) == 2
        assert sorted(out[0].keys()) == [1, 2]
        assert sorted(out[1].keys()) == [3, 4]
        # The media dict's "id" field also reflects the new id.
        assert out[0][1]["id"] == 1
        assert out[0][2]["id"] == 2
        assert out[1][3]["id"] == 3
        assert out[1][4]["id"] == 4
        # Original names are preserved (no media gets swapped).
        names = [m["name"] for chunk in out for m in chunk.values()]
        assert names == ["a1", "a2", "b1", "b2"]

    def test_three_chunks_of_varying_size(self):
        chunks = [
            {1: {"id": 1, "tag": "x"}},
            {1: {"id": 1, "tag": "y1"}, 2: {"id": 2, "tag": "y2"}, 3: {"id": 3, "tag": "y3"}},
            {1: {"id": 1, "tag": "z1"}, 2: {"id": 2, "tag": "z2"}},
        ]
        out = list(_renumber_chunks(iter(chunks)))
        # 1 + 3 + 2 = 6 globally unique ids.
        all_ids = [cid for chunk in out for cid in chunk.keys()]
        assert all_ids == [1, 2, 3, 4, 5, 6]
        # Tags follow the original order across chunks.
        tags = [m["tag"] for chunk in out for m in chunk.values()]
        assert tags == ["x", "y1", "y2", "y3", "z1", "z2"]

    def test_empty_iterator_yields_nothing(self):
        assert list(_renumber_chunks(iter([]))) == []

    def test_empty_chunk_skipped_but_does_not_advance_counter(self):
        chunks = [{}, {1: {"id": 1, "name": "a"}}, {}, {1: {"id": 1, "name": "b"}}]
        out = list(_renumber_chunks(iter(chunks)))
        # Empty chunks pass through as empty dicts (the renumberer
        # doesn't filter); only present medias consume an id.
        assert [sorted(c.keys()) for c in out] == [[], [1], [], [2]]
        assert out[1][1]["name"] == "a"
        assert out[3][2]["name"] == "b"

    def test_lazy_evaluation_does_not_consume_upstream(self):
        """The helper should be a true generator — it must not buffer."""
        consumed = []

        def source():
            for i in range(3):
                consumed.append(i)
                yield {1: {"id": 1, "i": i}}

        it = _renumber_chunks(source())
        # Pull one element only.
        first = next(it)
        assert first == {1: {"id": 1, "i": 0}}
        # Only the first upstream chunk should have been consumed.
        assert consumed == [0]

    def test_media_dict_id_field_overwritten(self):
        """The renumberer rewrites ``media["id"]`` in place, not just the dict key."""
        chunks = [
            {1: {"id": 1, "name": "a"}},
            {1: {"id": 1, "name": "b"}},
        ]
        out = list(_renumber_chunks(iter(chunks)))
        # ``b`` lives at global id 2 in both the outer dict and the
        # media's own ``id`` field.
        assert out[1][2]["id"] == 2
        assert out[1][2]["name"] == "b"
