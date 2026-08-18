"""Tests for the pre-computed vector guard (:mod:`vtscore.embedding.precomputed`).

Vectors that arrive from outside VTSearch - an ``.npz`` manifest of
pre-computed embeddings, an importer's ``content_vectors`` entry, a re-ingest
source - carry no guarantee about their width, dtype or finiteness.  Adopted
unchecked, none of those defects fail where they were introduced: a wrong-width
row resurfaces as a bare numpy broadcast error inside the matrix builder on an
unrelated request, and a non-finite row never raises at all.

These tests pin the door shut at ingestion, and pin the scoring-path backstop
that catches whatever slips past it by some other route.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.datasets.importers._npz_vectors import (
    read_npz_archive_member_rows,
    read_npz_filenames_and_vectors,
    read_npz_multi_vectors,
    write_npz_multi_vectors,
)
from vtscore.datasets.loader_folder import _resolve_file_embedding
from vtscore.datasets.stages.embedding import _stamp_requested_embedder
from vtscore.embedding.binding import expected_dim_for_embedder
from vtscore.embedding.media_vectors import UNKNOWN_EMBEDDER_KEY
from vtscore.embedding.precomputed import (
    MismatchedVectorError,
    normalize_vector,
    normalize_vector_block,
    require_dim,
    stack_vectors,
    vector_dim,
)


def _write(path, **arrays):
    """Write an ``.npz`` and return its path."""
    np.savez(path, **arrays)
    return path


class TestNormalizeVector:
    def test_pins_dtype_to_float32(self):
        """A float64 research export must not leave the dataset mixed-dtype."""
        out = normalize_vector(np.arange(4, dtype=np.float64), label="v")
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]

    def test_accepts_float16_input(self):
        """Half-precision vectors are legal input; they are widened, not refused."""
        out = normalize_vector(np.ones(4, dtype=np.float16), label="v")
        assert out.dtype == np.float32
        assert np.array_equal(out, np.ones(4, dtype=np.float32))

    def test_accepts_plain_python_list(self):
        assert np.array_equal(normalize_vector([1.0, 2.0], label="v"), np.array([1.0, 2.0], dtype=np.float32))

    def test_flattens_single_row_2d(self):
        """A script that kept the leading axis produced (1, D); accept it."""
        assert normalize_vector(np.ones((1, 5)), label="v").shape == (5,)

    def test_rejects_none(self):
        with pytest.raises(MismatchedVectorError, match="no vector supplied"):
            normalize_vector(None, label="v")

    def test_rejects_multi_row_2d(self):
        with pytest.raises(MismatchedVectorError, match="expected a 1-D vector"):
            normalize_vector(np.ones((3, 4)), label="v")

    def test_rejects_empty(self):
        with pytest.raises(MismatchedVectorError, match="zero-length"):
            normalize_vector(np.zeros(0), label="v")

    def test_rejects_nan(self):
        with pytest.raises(MismatchedVectorError, match="non-finite"):
            normalize_vector(np.array([1.0, np.nan]), label="v")

    def test_rejects_inf(self):
        with pytest.raises(MismatchedVectorError, match="non-finite"):
            normalize_vector(np.array([1.0, np.inf]), label="v")

    def test_rejects_float64_magnitude_that_overflows_float32(self):
        """The finiteness check runs after the cast, so narrowing overflow is caught."""
        with pytest.raises(MismatchedVectorError, match="non-finite"):
            normalize_vector(np.array([1e300, 2.0], dtype=np.float64), label="v")

    def test_rejects_non_numeric(self):
        with pytest.raises(MismatchedVectorError, match="non-numeric"):
            normalize_vector(np.array(["a", "b"]), label="v")

    def test_rejects_wrong_expected_dim_and_names_both_widths(self):
        with pytest.raises(MismatchedVectorError) as exc:
            normalize_vector(np.ones(768), label="row 3", expected_dim=1152, expected_source="embedder 'siglip2_l'")
        msg = str(exc.value)
        assert "768" in msg and "1152" in msg
        assert "row 3" in msg
        assert "siglip2_l" in msg

    def test_is_a_value_error(self):
        """Existing ``except ValueError`` handlers must keep working."""
        assert issubclass(MismatchedVectorError, ValueError)


class TestNormalizeVectorBlock:
    def test_pins_dtype_and_keeps_shape(self):
        out = normalize_vector_block(np.arange(6, dtype=np.float64).reshape(3, 2), label="b")
        assert out.dtype == np.float32
        assert out.shape == (3, 2)

    def test_rejects_1d_block(self):
        """A manifest that lost its row axis is a bug, not a one-row archive."""
        with pytest.raises(MismatchedVectorError, match=r"expected a 2-D \(N, D\) array"):
            normalize_vector_block(np.ones(4), label="b")

    def test_rejects_zero_width(self):
        with pytest.raises(MismatchedVectorError, match="zero-width"):
            normalize_vector_block(np.zeros((3, 0)), label="b")

    def test_names_first_offending_row(self):
        block = np.ones((5, 3))
        block[3, 1] = np.nan
        with pytest.raises(MismatchedVectorError, match="row: index 3"):
            normalize_vector_block(block, label="b")

    def test_rejects_wrong_expected_dim(self):
        with pytest.raises(MismatchedVectorError, match="768-dimensional but 1152"):
            normalize_vector_block(np.ones((2, 768)), label="b", expected_dim=1152)


class TestRequireDimAndStackVectors:
    def test_require_dim_passes_on_match(self):
        require_dim(np.ones(4, dtype=np.float32), 4, label="m")

    def test_require_dim_names_both_widths(self):
        with pytest.raises(MismatchedVectorError, match="5-dimensional but 4 was expected"):
            require_dim(np.ones(5), 4, label="media 7 (cat.jpg)")

    def test_require_dim_rejects_non_1d(self):
        with pytest.raises(MismatchedVectorError, match="expected a 1-D"):
            require_dim(np.ones((2, 4)), 4, label="m")

    def test_vector_dim_reads_trailing_axis(self):
        assert vector_dim(np.ones((3, 7))) == 7
        assert vector_dim([1.0, 2.0]) == 2

    def test_stack_vectors_names_the_odd_row(self):
        with pytest.raises(MismatchedVectorError, match=r"row 1 \(b\.jpg\)"):
            stack_vectors(
                [np.ones(4), np.ones(5)],
                label="training vector",
                row_labels=["a.jpg", "b.jpg"],
            )

    def test_stack_vectors_happy_path_is_float32(self):
        out = stack_vectors([np.ones(3, dtype=np.float64), np.zeros(3)], label="v")
        assert out.shape == (2, 3)
        assert out.dtype == np.float32

    def test_stack_vectors_rejects_empty(self):
        with pytest.raises(MismatchedVectorError, match="no vectors to stack"):
            stack_vectors([], label="v")


class TestNpzFilenamesAndVectorsValidation:
    def test_float64_manifest_is_widened_to_float32(self, tmp_path):
        p = _write(
            tmp_path / "m.npz",
            filenames=np.array(["a.jpg", "b.jpg"]),
            vectors=np.arange(8, dtype=np.float64).reshape(2, 4),
        )
        mapping = read_npz_filenames_and_vectors(p)
        assert {v.dtype for v in mapping.values()} == {np.dtype(np.float32)}

    def test_float16_manifest_is_widened_to_float32(self, tmp_path):
        """A half-precision embed can ship its vectors; they are widened at the door."""
        p = _write(tmp_path / "m.npz", filenames=np.array(["a.jpg"]), vectors=np.ones((1, 4), dtype=np.float16))
        assert read_npz_filenames_and_vectors(p)["a.jpg"].dtype == np.float32

    def test_non_finite_manifest_is_rejected(self, tmp_path):
        p = _write(
            tmp_path / "m.npz",
            filenames=np.array(["a.jpg", "b.jpg"]),
            vectors=np.array([[1.0, 2.0], [np.nan, 3.0]]),
        )
        with pytest.raises(MismatchedVectorError, match="non-finite"):
            read_npz_filenames_and_vectors(p)

    def test_declared_embedder_width_mismatch_is_rejected(self, tmp_path):
        """The manifest contradicts itself: siglip2_l is 1152-dim, these rows are 768."""
        assert expected_dim_for_embedder("siglip2_l") == 1152
        p = _write(
            tmp_path / "m.npz",
            filenames=np.array(["a.jpg"]),
            vectors=np.ones((1, 768), dtype=np.float32),
            embedder_name=np.array("siglip2_l"),
        )
        with pytest.raises(MismatchedVectorError) as exc:
            read_npz_filenames_and_vectors(p)
        assert "siglip2_l" in str(exc.value)
        assert "768" in str(exc.value) and "1152" in str(exc.value)

    def test_matching_declared_embedder_width_is_accepted(self, tmp_path):
        p = _write(
            tmp_path / "m.npz",
            filenames=np.array(["a.jpg"]),
            vectors=np.ones((1, 1152), dtype=np.float32),
            embedder_name=np.array("siglip2_l"),
        )
        assert read_npz_filenames_and_vectors(p)["a.jpg"].shape == (1152,)

    def test_unregistered_embedder_name_skips_the_width_check(self, tmp_path):
        """An unknown name means "nothing to check against", not an error here.

        ``validate_manifest_embedder_name`` is what rejects an unroutable name,
        and it runs in the importer with the media type in hand.
        """
        p = _write(
            tmp_path / "m.npz",
            filenames=np.array(["a.jpg"]),
            vectors=np.ones((1, 7), dtype=np.float32),
            embedder_name=np.array("not_a_real_embedder"),
        )
        assert read_npz_filenames_and_vectors(p)["a.jpg"].shape == (7,)


class TestNpzPerKeyLayoutValidation:
    def test_ragged_per_key_archive_names_the_offending_key(self, tmp_path):
        """The per-key layout has no structural guarantee that rows agree."""
        p = _write(tmp_path / "m.npz", a=np.ones(4), b=np.ones(5))
        with pytest.raises(MismatchedVectorError) as exc:
            read_npz_filenames_and_vectors(p)
        assert "'b'" in str(exc.value)
        assert "5-dimensional but 4" in str(exc.value)

    def test_consistent_per_key_archive_is_accepted_and_widened(self, tmp_path):
        p = _write(tmp_path / "m.npz", a=np.ones(4, dtype=np.float64), b=np.zeros(4, dtype=np.float64))
        mapping = read_npz_filenames_and_vectors(p)
        assert set(mapping) == {"a", "b"}
        assert {v.dtype for v in mapping.values()} == {np.dtype(np.float32)}

    def test_per_key_non_finite_is_rejected(self, tmp_path):
        p = _write(tmp_path / "m.npz", a=np.array([1.0, np.inf]))
        with pytest.raises(MismatchedVectorError, match="non-finite"):
            read_npz_filenames_and_vectors(p)


class TestNpzMultiVectorValidation:
    def test_columns_may_differ_in_width_from_each_other(self, tmp_path):
        """A trio archive's columns are *meant* to differ; only per-column checks apply.

        ``siglip2_l`` really is 1152-dim while ``dinov3_patch`` really is 768-dim,
        so a correct trio archive would trip any cross-column consistency check.
        """
        p = _write(
            tmp_path / "m.npz",
            filenames=np.array(["a.jpg"]),
            vectors_siglip2_l=np.ones((1, 1152), dtype=np.float32),
            vectors_dinov3_patch=np.ones((1, 768), dtype=np.float32),
        )
        result = read_npz_multi_vectors(p)
        assert result is not None
        mapping, _primary = result
        assert mapping["a.jpg"]["siglip2_l"].shape == (1152,)
        assert mapping["a.jpg"]["dinov3_patch"].shape == (768,)

    def test_column_checked_against_its_own_embedder_width(self, tmp_path):
        """``vectors_siglip2_l`` holding 768-dim rows is the mislabelled-column case."""
        p = _write(
            tmp_path / "m.npz",
            filenames=np.array(["a.jpg"]),
            vectors_siglip2_l=np.ones((1, 768), dtype=np.float32),
        )
        with pytest.raises(MismatchedVectorError, match="siglip2_l"):
            read_npz_multi_vectors(p)

    def test_non_finite_column_is_rejected(self, tmp_path):
        p = _write(
            tmp_path / "m.npz",
            filenames=np.array(["a.jpg"]),
            vectors_mystery=np.array([[1.0, np.nan]]),
        )
        with pytest.raises(MismatchedVectorError, match="non-finite"):
            read_npz_multi_vectors(p)

    def test_writer_names_the_offending_file_on_a_width_mismatch(self, tmp_path):
        """``write_npz_multi_vectors`` used a bare ``np.stack``, which named nothing."""
        mapping = {"a.jpg": {"e": np.ones(4)}, "b.jpg": {"e": np.ones(5)}}
        with pytest.raises(MismatchedVectorError, match="b.jpg"):
            write_npz_multi_vectors(tmp_path / "out.npz", mapping)

    def test_multi_vector_round_trip(self, tmp_path):
        mapping = {
            "a.jpg": {"siglip": np.ones(768, dtype=np.float32)},
            "b.jpg": {"siglip": np.zeros(768, dtype=np.float32)},
        }
        out = tmp_path / "out.npz"
        write_npz_multi_vectors(out, mapping, "siglip")
        result = read_npz_multi_vectors(out)
        assert result is not None
        read_back, primary = result
        assert primary == "siglip"
        assert np.array_equal(read_back["a.jpg"]["siglip"], np.ones(768, dtype=np.float32))


class TestArchiveMemberManifestValidation:
    def _manifest(self, tmp_path, vectors, **extra):
        return _write(
            tmp_path / "manifest.npz",
            vectors=vectors,
            members=np.array(["x.jpg", "y.jpg"][: len(vectors)]),
            archives=np.array("shards.tar"),
            **extra,
        )

    def test_float64_rows_are_widened(self, tmp_path):
        p = self._manifest(tmp_path, np.arange(8, dtype=np.float64).reshape(2, 4))
        rows = read_npz_archive_member_rows(p)
        assert all(r["vector"].dtype == np.float32 for r in rows)

    def test_non_finite_rows_are_rejected(self, tmp_path):
        p = self._manifest(tmp_path, np.array([[1.0, 2.0], [3.0, np.nan]]))
        with pytest.raises(MismatchedVectorError, match="non-finite"):
            read_npz_archive_member_rows(p)

    def test_declared_embedder_width_mismatch_is_rejected(self, tmp_path):
        p = self._manifest(tmp_path, np.ones((2, 768), dtype=np.float32), embedder_name=np.array("siglip2_l"))
        with pytest.raises(MismatchedVectorError, match="siglip2_l"):
            read_npz_archive_member_rows(p)


class TestContentVectorAdoption:
    """``content_vectors`` / ``custom_metadata_map`` are the plugin-importer channel."""

    def test_content_vector_is_widened_to_float32(self):
        vec, name = _resolve_file_embedding("a.jpg", "a.jpg", None, {"a.jpg": np.ones(4, dtype=np.float64)}, "siglip")
        assert vec.dtype == np.float32
        assert name == "siglip"

    def test_non_finite_content_vector_is_rejected(self):
        with pytest.raises(MismatchedVectorError, match="non-finite"):
            _resolve_file_embedding("a.jpg", "a.jpg", None, {"a.jpg": np.array([1.0, np.nan])}, "")

    def test_non_finite_custom_metadata_vector_is_rejected(self):
        with pytest.raises(MismatchedVectorError, match="non-finite"):
            _resolve_file_embedding("a.jpg", "a.jpg", {"embedding": np.array([np.inf])}, None, "")

    def test_per_embedder_dict_is_validated_column_by_column(self):
        """A trio import supplies a ready-made ``{embedder: vector}`` dict."""
        value = {"siglip": np.ones(4, dtype=np.float64), "dinov3_patch": np.ones(8, dtype=np.float64)}
        vec, _name = _resolve_file_embedding("a.jpg", "a.jpg", None, {"a.jpg": value}, "siglip")
        assert set(vec) == {"siglip", "dinov3_patch"}
        assert vec["siglip"].shape == (4,) and vec["dinov3_patch"].shape == (8,)
        assert all(v.dtype == np.float32 for v in vec.values())

    def test_bad_column_in_a_per_embedder_dict_is_rejected(self):
        value = {"siglip": np.ones(4), "dinov3_patch": np.array([np.nan, 1.0])}
        with pytest.raises(MismatchedVectorError, match="dinov3_patch"):
            _resolve_file_embedding("a.jpg", "a.jpg", None, {"a.jpg": value}, "siglip")

    def test_absent_vector_still_returns_none(self):
        assert _resolve_file_embedding("a.jpg", "a.jpg", None, {}, "") == (None, "")


class TestStampRequestedEmbedder:
    """Re-keying a nameless vector under a named embedder is an assertion, not a rename.

    A manifest that ships vectors but no ``embedder_name`` stores them under the
    blank sentinel key.  When the load names an embedder, the stage re-keys them
    under that name - which claims the vectors live in that embedder's space.  A
    manifest whose vectors came from a different model makes the claim false, and
    the media then looks identical to a correctly-labelled one.
    """

    @staticmethod
    def _nameless(dim: int) -> dict:
        return {1: {"id": 1, "filename": "a.jpg", "embeddings": {UNKNOWN_EMBEDDER_KEY: np.ones(dim, np.float32)}}}

    def test_matching_width_is_stamped(self):
        medias = self._nameless(1152)
        _stamp_requested_embedder(medias, "siglip2_l")
        assert medias[1]["embedder"] == "siglip2_l"
        assert set(medias[1]["embeddings"]) == {"siglip2_l"}

    def test_wrong_width_is_rejected_and_names_both_widths(self):
        medias = self._nameless(768)
        with pytest.raises(MismatchedVectorError) as exc:
            _stamp_requested_embedder(medias, "siglip2_l")
        msg = str(exc.value)
        assert "a.jpg" in msg
        assert "768" in msg and "1152" in msg

    def test_wrong_width_leaves_the_media_untouched(self):
        """A rejected stamp must not half-apply: the sentinel key stays put."""
        medias = self._nameless(768)
        with pytest.raises(MismatchedVectorError):
            _stamp_requested_embedder(medias, "siglip2_l")
        assert set(medias[1]["embeddings"]) == {UNKNOWN_EMBEDDER_KEY}
        assert "embedder" not in medias[1]

    def test_unknown_embedder_width_skips_the_check(self):
        medias = self._nameless(7)
        _stamp_requested_embedder(medias, "not_a_real_embedder")
        assert medias[1]["embedder"] == "not_a_real_embedder"

    def test_importer_set_name_is_never_overwritten(self):
        medias = {1: {"id": 1, "embedder": "clip", "embeddings": {"clip": np.ones(512, np.float32)}}}
        _stamp_requested_embedder(medias, "siglip2_l")
        assert medias[1]["embedder"] == "clip"
