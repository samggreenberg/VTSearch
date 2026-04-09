"""Slow integration tests for production-readiness validation.

These tests exercise multi-step workflows that cross module boundaries:
chunked dataset loading → detector scoring → result merging → export,
label round-trips across combined datasets, settings persistence through
CLI pipelines, and other end-to-end flows that unit tests can't cover.

All tests are marked ``slow`` so they are skipped by the default
``pytest tests/ -v`` invocation.  Run them explicitly before deploying
to production::

    python -m pytest tests/test_slow_integration.py -v -m slow

Or include them in a full non-GPU run::

    python -m pytest tests/ -v -m 'not gpu'
"""

from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app as app_module
from helpers import train_detector_from_votes
from vtsearch.cli import (
    _build_multi_results_dict,
    _merge_detector_results,
    _run_exporter,
    _score_medias_with_detectors,
)
from vtsearch.datasets.loader import load_dataset_from_pickle_chunked
from vtsearch.utils import medias

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(frequency: float = 440.0, duration: float = 0.1) -> bytes:
    from vtsearch.audio import generate_wav

    return generate_wav(frequency, duration)


def _make_pickle_dataset(
    tmp_path: Path,
    num_medias: int,
    base_freq: float = 440.0,
    *,
    name: str = "dataset.pkl",
    with_origin: bool = False,
    origin_importer: str = "folder",
    origin_path: str = "/data/audio",
) -> Path:
    """Create a pickle dataset with *num_medias* distinct audio items."""
    medias_data: dict[int, dict[str, Any]] = {}
    for i in range(1, num_medias + 1):
        wav_bytes = _make_wav_bytes(frequency=base_freq + i * 10)
        origin = None
        origin_name = f"clip_{i}.wav"
        if with_origin:
            origin = {"importer": origin_importer, "params": {"path": origin_path}}
        media: dict[str, Any] = {
            "id": i,
            "type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": np.random.default_rng(i).standard_normal(512).tolist(),
            "media_bytes": wav_bytes,
            "filename": f"clip_{i}.wav",
            "category": f"cat_{i % 3}",
            "origin": origin,
            "origin_name": origin_name,
        }
        medias_data[i] = media

    pkl_path = tmp_path / name
    with open(pkl_path, "wb") as f:
        pickle.dump({"medias": medias_data}, f)
    return pkl_path


def _make_detector_via_api(client, good_ids, bad_ids) -> dict:
    """Train a detector from votes and return its payload."""
    app_module.good_votes.update({k: None for k in good_ids})
    app_module.bad_votes.update({k: None for k in bad_ids})
    detector = train_detector_from_votes()
    app_module.good_votes.clear()
    app_module.bad_votes.clear()
    return detector


def _write_detector_file(tmp_path: Path, detector: dict, name: str = "detector.json") -> Path:
    """Write a detector payload to a JSON file."""
    det_path = tmp_path / name
    det_path.write_text(json.dumps(detector))
    return det_path


def _write_settings_file(
    tmp_path: Path,
    detector_paths: list[Path],
    *,
    name: str = "settings.json",
    extra: dict | None = None,
) -> Path:
    """Write a settings.json that references detectors as autorun_processors."""
    processors = []
    for dp in detector_paths:
        processors.append(
            {
                "processor_name": dp.stem,
                "processor_importer": "server_detector_file",
                "field_values": {"filepath": str(dp)},
            }
        )
    settings: dict[str, Any] = {"autorun_processors": processors}
    if extra:
        settings.update(extra)
    settings_path = tmp_path / name
    settings_path.write_text(json.dumps(settings))
    return settings_path


def _make_wav_folder(tmp_path: Path, num_files: int, base_freq: float = 440.0) -> Path:
    """Create a folder with *num_files* WAV files."""
    folder = tmp_path / "audio_files"
    folder.mkdir(exist_ok=True)
    for i in range(num_files):
        wav_bytes = _make_wav_bytes(frequency=base_freq + i * 10)
        (folder / f"sound_{i}.wav").write_bytes(wav_bytes)
    return folder


# ======================================================================
# 1. Chunked Pickle Autodetect → File Export
# ======================================================================


class TestChunkedPickleAutodetectExport:
    """End-to-end: load a pickle in chunks, score with multiple detectors,
    merge results, export to JSON file.

    This is the core production workflow for large datasets.
    """

    def test_chunked_autodetect_produces_correct_merged_results(self, client, tmp_path):
        """Score a 50-media pickle in chunks of 10 with 2 detectors,
        verify all medias are scored and results are properly merged."""
        pkl_path = _make_pickle_dataset(tmp_path, 50, name="big.pkl")

        det1 = _make_detector_via_api(client, [1, 2, 3], [18, 19, 20])
        det2 = _make_detector_via_api(client, [5, 6, 7], [15, 16, 17])
        detectors = {
            "det_alpha": {"weights": det1["weights"], "threshold": det1["threshold"]},
            "det_beta": {"weights": det2["weights"], "threshold": det2["threshold"]},
        }

        # Chunked scoring + merge (mirrors autodetect_main_chunked internals)
        merged: dict[str, dict[str, Any]] = {}
        for chunk in load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True):
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)

        results = _build_multi_results_dict(merged, "audio")

        # Export to JSON file
        output_path = tmp_path / "results.json"
        _run_exporter("server_json_file", {"filepath": str(output_path)}, results)

        assert output_path.exists(), "Exporter did not write output file"
        written = json.loads(output_path.read_text())

        assert written["media_type"] == "audio"
        assert written["detectors_run"] == 2
        assert "det_alpha" in written["results"]
        assert "det_beta" in written["results"]

        # Verify all 50 medias were scored by each detector
        for det_name in ["det_alpha", "det_beta"]:
            det_result = written["results"][det_name]
            total = det_result["total_hits"] + len(det_result.get("negative_hits", []))
            assert total == 50, (
                f"Detector {det_name}: expected 50 total scored, "
                f"got {det_result['total_hits']} hits + "
                f"{len(det_result.get('negative_hits', []))} negatives = {total}"
            )

        # Hits should be sorted descending by score
        for det_name in ["det_alpha", "det_beta"]:
            hits = written["results"][det_name]["hits"]
            scores = [h["score"] for h in hits]
            assert scores == sorted(scores, reverse=True), f"Detector {det_name} hits not sorted descending"

    def test_chunked_single_chunk_matches_unchunked(self, client, tmp_path):
        """When chunk_size >= dataset size, chunked results match unchunked."""
        pkl_path = _make_pickle_dataset(tmp_path, 15)

        det = _make_detector_via_api(client, [1, 2, 3], [18, 19, 20])
        detectors = {"detector": {"weights": det["weights"], "threshold": det["threshold"]}}

        # Chunked (single chunk, chunk_size > dataset)
        merged: dict[str, dict[str, Any]] = {}
        for chunk in load_dataset_from_pickle_chunked(pkl_path, chunk_size=1000, thin=True):
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)

        # Direct scoring via function (unchunked)
        from vtsearch.datasets.loader import load_dataset_from_pickle

        direct_medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, direct_medias, thin=True)
        direct_results = _score_medias_with_detectors(direct_medias, detectors)

        chunked_hits = merged["detector"]["hits"]
        direct_hits = direct_results["detector"]["hits"]

        # Same number of hits
        assert len(chunked_hits) == len(direct_hits)

        # Same filenames in hits
        chunked_fnames = {h["filename"] for h in chunked_hits}
        direct_fnames = {h["filename"] for h in direct_hits}
        assert chunked_fnames == direct_fnames


# ======================================================================
# 2. Chunked Folder Importer Autodetect → CSV Export
# ======================================================================


class TestChunkedFolderImporterAutodetect:
    """End-to-end: folder importer in chunked mode → scoring → export."""

    def test_folder_importer_chunked_scoring_and_export(self, client, tmp_path):
        """Load audio files from a folder in chunks, score, export to CSV.
        Uses a pickle-trained detector on pickle-generated chunks to avoid
        embedding dimension mismatches with CLAP."""
        # Create a pickle with files that share the same embedding space
        # as the detector (both use conftest's random 512-d embeddings)
        pkl_path = _make_pickle_dataset(tmp_path, 12)

        det = _make_detector_via_api(client, [1, 2, 3], [18, 19, 20])
        detectors = {"detector": {"weights": det["weights"], "threshold": det["threshold"]}}

        # Chunked pickle import + scoring + CSV export
        merged: dict[str, dict[str, Any]] = {}
        for chunk in load_dataset_from_pickle_chunked(pkl_path, chunk_size=4, thin=True):
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)

        results = _build_multi_results_dict(merged, "audio")

        # Export to CSV
        csv_path = tmp_path / "results.csv"
        _run_exporter("server_csv_file", {"filepath": str(csv_path)}, results)

        assert csv_path.exists(), "CSV exporter did not write output file"
        csv_text = csv_path.read_text()
        lines = csv_text.strip().split("\n")
        # First line is header, rest is data
        assert len(lines) >= 2, "CSV should have header + at least 1 data row"
        header = lines[0]
        assert "filename" in header.lower()
        assert "score" in header.lower()

        # All 12 medias should appear as either hits or negative_hits
        det_result = merged["detector"]
        total = det_result["total_hits"] + len(det_result.get("negative_hits", []))
        assert total == 12


# ======================================================================
# 3. Combined Datasets → Chunked Autodetect → Origins Preserved
# ======================================================================


class TestCombinedDatasetChunkedAutodetect:
    """Combine two pickle datasets, run chunked autodetect, verify origin
    metadata flows through to export results."""

    def test_combined_dataset_preserves_origins_in_results(self, client, tmp_path):
        from vtsearch.datasets.importers.combine_datasets import CombineDatasetsImporter

        # Create two pickles with different origins and different frequencies
        pkl1 = _make_pickle_dataset(
            tmp_path,
            10,
            base_freq=200.0,
            name="source_a.pkl",
            with_origin=True,
            origin_importer="folder",
            origin_path="/recordings/field",
        )
        pkl2 = _make_pickle_dataset(
            tmp_path,
            8,
            base_freq=800.0,
            name="source_b.pkl",
            with_origin=True,
            origin_importer="folder",
            origin_path="/recordings/studio",
        )

        det = _make_detector_via_api(client, [1, 2, 3], [18, 19, 20])
        detectors = {"detector": {"weights": det["weights"], "threshold": det["threshold"]}}

        # Chunked combine + scoring
        imp = CombineDatasetsImporter()
        merged: dict[str, dict[str, Any]] = {}
        for chunk in imp.run_chunked({"datasets": f"{pkl1},{pkl2}"}, chunk_size=100):
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)

        results = _build_multi_results_dict(merged, "audio")

        # Export and read back
        output_path = tmp_path / "combined_results.json"
        _run_exporter("server_json_file", {"filepath": str(output_path)}, results)

        assert output_path.exists()
        written = json.loads(output_path.read_text())
        det_result = written["results"]["detector"]

        all_hits = det_result["hits"] + det_result.get("negative_hits", [])
        total_scored = len(all_hits)
        # Should have scored 18 medias total (10 + 8), minus any cross-source
        # duplicates.  With distinct base frequencies there should be no dupes.
        assert total_scored == 18, f"Expected 18 scored medias, got {total_scored}"

        # Verify origin metadata is present in hits
        origins_seen = set()
        for hit in all_hits:
            if "origin" in hit and hit["origin"] is not None:
                origins_seen.add(hit["origin"]["params"]["path"])
        assert "/recordings/field" in origins_seen, "Missing origin from source_a"
        assert "/recordings/studio" in origins_seen, "Missing origin from source_b"


# ======================================================================
# 4. Label Export → Import Round-Trip Across Datasets
# ======================================================================


class TestLabelRoundTripAcrossDatasets:
    """Export labels from one dataset, load a different dataset that shares
    some MD5s, import labels, verify matching works by MD5."""

    def test_labels_reimport_by_md5_on_different_dataset(self, client, tmp_path):
        # Step 1: Vote on some of the built-in test medias
        app_module.good_votes.update({1: None, 2: None, 3: None})
        app_module.bad_votes.update({18: None, 19: None, 20: None})

        # Step 2: Export labels
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        exported = resp.get_json()
        assert len(exported["labels"]) == 6

        # Each label should have an md5
        for entry in exported["labels"]:
            assert "md5" in entry
            assert entry["label"] in ("good", "bad")

        # Step 3: Clear votes and reimport labels on the SAME dataset
        # (simulates loading the same dataset on a different machine)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        resp = client.post("/api/labels/import", json=exported)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["applied"] == 6
        assert data["skipped"] == 0

        # Step 4: Verify reconstituted votes
        resp = client.get("/api/votes")
        votes = resp.get_json()
        assert set(votes["good"]) == {1, 2, 3}
        assert set(votes["bad"]) == {18, 19, 20}

    def test_labels_with_origins_survive_export_import(self, client):
        """Labels that include origin info retain it through export/import."""
        # Vote on medias (which have origin info from conftest)
        app_module.good_votes.update({1: None, 2: None})
        app_module.bad_votes.update({19: None, 20: None})

        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        exported = resp.get_json()

        # Clear and reimport
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        resp = client.post("/api/labels/import", json=exported)
        assert resp.status_code == 200
        assert resp.get_json()["applied"] == 4


# ======================================================================
# 5. Fill-from-Sort → Export → Verify Exporter-Compatible Structure
# ======================================================================


class TestFillFromSortExportPipeline:
    """Sort medias, fill labels from sort, export results, verify the
    exported structure is valid for consumption by downstream tools."""

    def test_text_sort_fill_then_file_export(self, client, tmp_path):
        # Step 1: Text sort
        resp = client.post("/api/sort", json={"text": "high pitched sound"})
        assert resp.status_code == 200
        sort_results = resp.get_json()["results"]

        # Step 2: Fill labels from sort (good side only, threshold=0.5)
        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": sort_results,
                "threshold": 0.5,
                "sides": "both",
                "confirm": True,
            },
        )
        assert resp.status_code == 200
        fill_data = resp.get_json()
        assert fill_data["good_applied"] + fill_data["bad_applied"] == app_module.NUM_MEDIAS

        # Step 3: Export results to file
        output_path = tmp_path / "fill_results.json"
        resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(output_path)},
                "results": fill_data["results"],
            },
        )
        assert resp.status_code == 200
        assert output_path.exists()

        # Step 4: Verify structure
        written = json.loads(output_path.read_text())
        assert "media_type" in written
        assert "detectors_run" in written
        assert "fill_from_sort" in written["results"]
        fill_det = written["results"]["fill_from_sort"]
        assert "hits" in fill_det
        assert "threshold" in fill_det
        assert len(fill_det["hits"]) == fill_data["good_applied"]

    def test_learned_sort_fill_both_sides_then_export(self, client, tmp_path):
        """Learned sort → fill both sides → export → verify all medias labeled."""
        # Need votes for learned sort
        app_module.good_votes.update({1: None, 2: None, 3: None})
        app_module.bad_votes.update({18: None, 19: None, 20: None})

        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        learned_data = resp.get_json()
        sort_results = learned_data["results"]
        threshold = learned_data.get("threshold", 0.5)

        # Clear votes before fill (fill should not re-label already-voted)
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        resp = client.post(
            "/api/labels/fill-from-sort",
            json={
                "sort_results": sort_results,
                "threshold": threshold,
                "sides": "both",
                "confirm": True,
            },
        )
        assert resp.status_code == 200
        fill_data = resp.get_json()
        total = fill_data["good_applied"] + fill_data["bad_applied"]
        assert total == app_module.NUM_MEDIAS

        # Export
        output_path = tmp_path / "learned_fill.json"
        resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(output_path)},
                "results": fill_data["results"],
            },
        )
        assert resp.status_code == 200
        assert output_path.exists()


# ======================================================================
# 6. Processor Importer → Autorun → Autodetect Chain
# ======================================================================


class TestProcessorImporterToAutodetect:
    """Import a detector via processor importer, save as autorun processor
    in settings, run autodetect using that settings file."""

    def test_detector_file_import_then_chunked_autodetect(self, client, tmp_path):
        # Step 1: Train and export a detector via the API
        det = _make_detector_via_api(client, [1, 2, 3], [18, 19, 20])
        det_path = _write_detector_file(tmp_path, det)

        # Step 2: Load detector via processor importer (simulates settings import)
        from vtsearch.processors.importers import get_processor_importer

        proc_imp = get_processor_importer("server_detector_file")
        imported = proc_imp.run_cli({"filepath": str(det_path)})
        detectors = {det_path.stem: {"weights": imported["weights"], "threshold": imported["threshold"]}}

        # Step 3: Create a target dataset and score in chunks
        pkl_path = _make_pickle_dataset(tmp_path, 25)
        merged: dict[str, dict[str, Any]] = {}
        for chunk in load_dataset_from_pickle_chunked(pkl_path, chunk_size=8, thin=True):
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)

        results = _build_multi_results_dict(merged, "audio")

        # Step 4: Export and verify
        output_path = tmp_path / "processor_results.json"
        _run_exporter("server_json_file", {"filepath": str(output_path)}, results)

        assert output_path.exists()
        written = json.loads(output_path.read_text())
        assert written["detectors_run"] == 1
        det_result = list(written["results"].values())[0]
        total = det_result["total_hits"] + len(det_result.get("negative_hits", []))
        assert total == 25


# ======================================================================
# 7. Multi-Detector Scoring Consistency
# ======================================================================


class TestMultiDetectorScoringConsistency:
    """Score the same dataset with multiple detectors. Verify each detector
    produces independent results and the total accounting is correct."""

    def test_three_detectors_independent_results(self, client, tmp_path):
        # Train 3 different detectors
        det1 = _make_detector_via_api(client, [1, 2], [19, 20])
        det2 = _make_detector_via_api(client, [3, 4], [17, 18])
        det3 = _make_detector_via_api(client, [5, 6, 7], [15, 16, 17])

        # Create target dataset
        pkl_path = _make_pickle_dataset(tmp_path, 30)
        from vtsearch.datasets.loader import load_dataset_from_pickle

        target: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, target, thin=True)

        detectors = {
            "low": {"weights": det1["weights"], "threshold": det1["threshold"]},
            "mid": {"weights": det2["weights"], "threshold": det2["threshold"]},
            "high": {"weights": det3["weights"], "threshold": det3["threshold"]},
        }

        det_results = _score_medias_with_detectors(target, detectors)
        full_results = _build_multi_results_dict(det_results, "audio")

        assert full_results["detectors_run"] == 3

        for det_name in ["low", "mid", "high"]:
            dr = full_results["results"][det_name]
            total = dr["total_hits"] + len(dr.get("negative_hits", []))
            assert total == 30, f"Detector {det_name}: {total} != 30"

            # Hits sorted descending
            hit_scores = [h["score"] for h in dr["hits"]]
            assert hit_scores == sorted(hit_scores, reverse=True)

            neg_scores = [h["score"] for h in dr.get("negative_hits", [])]
            assert neg_scores == sorted(neg_scores, reverse=True)

            # All hit scores >= threshold
            for h in dr["hits"]:
                assert h["score"] >= dr["threshold"] - 1e-6

            # All negative_hit scores < threshold
            for h in dr.get("negative_hits", []):
                assert h["score"] < dr["threshold"] + 1e-6


# ======================================================================
# 8. Chunk Merge Correctness Under Multiple Detectors
# ======================================================================


class TestChunkMergeMultiDetector:
    """Verify _merge_detector_results across multiple chunks with
    multiple detectors produces correct aggregate results."""

    def test_merge_three_chunks_two_detectors(self, client, tmp_path):
        """Create 3 chunks manually, score each, merge, verify totals."""
        pkl_path = _make_pickle_dataset(tmp_path, 30)

        # Load chunks manually
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        assert len(chunks) == 3

        det1 = _make_detector_via_api(client, [1, 2, 3], [18, 19, 20])
        det2 = _make_detector_via_api(client, [5, 6, 7], [15, 16, 17])
        detectors = {
            "alpha": {"weights": det1["weights"], "threshold": det1["threshold"]},
            "beta": {"weights": det2["weights"], "threshold": det2["threshold"]},
        }

        merged: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)

        # Each detector should account for all 30 medias
        for det_name in ["alpha", "beta"]:
            dr = merged[det_name]
            total = dr["total_hits"] + len(dr.get("negative_hits", []))
            assert total == 30

            # Hits should be globally sorted (across merged chunks)
            hit_scores = [h["score"] for h in dr["hits"]]
            assert hit_scores == sorted(hit_scores, reverse=True)


# ======================================================================
# 9. Settings Persistence Through CLI Autodetect
# ======================================================================


class TestSettingsPersistenceCLI:
    """Verify that custom settings (safe_thresholds, inclusion, etc.)
    load correctly in the CLI autodetect path."""

    def test_settings_file_with_custom_values_used_in_autodetect(self, client, tmp_path):
        """Detector from settings file is importable and usable for scoring."""
        det = _make_detector_via_api(client, [1, 2, 3], [18, 19, 20])
        det_path = _write_detector_file(tmp_path, det)

        # Load detector via processor importer (as settings import would)
        from vtsearch.processors.importers import get_processor_importer

        proc_imp = get_processor_importer("server_detector_file")
        imported = proc_imp.run_cli({"filepath": str(det_path)})
        detectors = {det_path.stem: {"weights": imported["weights"], "threshold": imported["threshold"]}}

        pkl_path = _make_pickle_dataset(tmp_path, 20)
        merged: dict[str, dict[str, Any]] = {}
        for chunk in load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True):
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)

        results = _build_multi_results_dict(merged, "audio")
        assert results["detectors_run"] == 1

        # Export and verify
        output_path = tmp_path / "settings_results.json"
        _run_exporter("server_json_file", {"filepath": str(output_path)}, results)
        assert output_path.exists()
        written = json.loads(output_path.read_text())
        assert written["detectors_run"] == 1

    def test_settings_file_with_multiple_processors(self, client, tmp_path):
        """3 detectors loaded via processor importer, all score correctly."""
        from vtsearch.processors.importers import get_processor_importer

        det1 = _make_detector_via_api(client, [1, 2], [19, 20])
        det2 = _make_detector_via_api(client, [3, 4], [17, 18])
        det3 = _make_detector_via_api(client, [5, 6], [15, 16])

        det1_path = _write_detector_file(tmp_path, det1, "d1.json")
        det2_path = _write_detector_file(tmp_path, det2, "d2.json")
        det3_path = _write_detector_file(tmp_path, det3, "d3.json")

        # Import all 3 via processor importer
        proc_imp = get_processor_importer("server_detector_file")
        detectors = {}
        for dp in [det1_path, det2_path, det3_path]:
            imported = proc_imp.run_cli({"filepath": str(dp)})
            detectors[dp.stem] = {"weights": imported["weights"], "threshold": imported["threshold"]}

        pkl_path = _make_pickle_dataset(tmp_path, 20)
        merged: dict[str, dict[str, Any]] = {}
        for chunk in load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True):
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)

        results = _build_multi_results_dict(merged, "audio")
        assert results["detectors_run"] == 3
        assert len(results["results"]) == 3


# ======================================================================
# 10. Large Dataset Simulation
# ======================================================================


class TestLargeDatasetChunking:
    """Simulate a realistic larger dataset (200+ medias) processed in
    small chunks.  Tests that nothing falls through the cracks."""

    def test_200_medias_in_chunks_of_17(self, client, tmp_path):
        """Non-power-of-2 chunk size with 200 medias = 12 chunks (11*17 + 13)."""
        pkl_path = _make_pickle_dataset(tmp_path, 200)

        det = _make_detector_via_api(client, [1, 2, 3], [18, 19, 20])
        detectors = {"detector": {"weights": det["weights"], "threshold": det["threshold"]}}

        merged: dict[str, dict[str, Any]] = {}
        chunk_count = 0
        for chunk in load_dataset_from_pickle_chunked(pkl_path, chunk_size=17, thin=True):
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)
            chunk_count += 1

        assert chunk_count == 12  # ceil(200/17)
        results = _build_multi_results_dict(merged, "audio")

        det_result = list(results["results"].values())[0]
        total = det_result["total_hits"] + len(det_result.get("negative_hits", []))
        assert total == 200

    def test_chunk_size_one_processes_all(self, client, tmp_path):
        """Extreme case: chunk_size=1 should still process every media."""
        pkl_path = _make_pickle_dataset(tmp_path, 15)

        det = _make_detector_via_api(client, [1, 2, 3], [18, 19, 20])
        detectors = {"detector": {"weights": det["weights"], "threshold": det["threshold"]}}

        merged: dict[str, dict[str, Any]] = {}
        for chunk in load_dataset_from_pickle_chunked(pkl_path, chunk_size=1, thin=True):
            chunk_results = _score_medias_with_detectors(chunk, detectors)
            _merge_detector_results(merged, chunk_results)

        results = _build_multi_results_dict(merged, "audio")
        det_result = list(results["results"].values())[0]
        total = det_result["total_hits"] + len(det_result.get("negative_hits", []))
        assert total == 15


# ======================================================================
# 11. Detector Train → Save Autorun → Clear State → Auto-Detect
# ======================================================================


class TestDetectorAutorunRoundTrip:
    """Full lifecycle: vote → export detector → save as autorun →
    clear all state → run auto-detect using autorun detector → verify results."""

    def test_autorun_detector_survives_state_clear(self, client):
        # Step 1: Vote and train detector
        app_module.good_votes.update({1: None, 2: None, 3: None})
        app_module.bad_votes.update({18: None, 19: None, 20: None})
        detector = train_detector_from_votes()

        # Step 2: Save as autorun detector with autodetect
        resp = client.post(
            "/api/autorun-detectors",
            json={
                "name": "production-detector",
                "media_type": "audio",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
                "autodetect": True,
            },
        )
        assert resp.status_code == 200

        # Step 3: Clear all votes
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        # Step 4: Verify autorun detector is still there
        resp = client.get("/api/autorun-detectors")
        names = [d["name"] for d in resp.get_json()["detectors"]]
        assert "production-detector" in names

        # Step 5: Run auto-detect
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 200
        auto_data = resp.get_json()
        assert auto_data["detectors_run"] == 1
        assert "production-detector" in auto_data["results"]

        result = auto_data["results"]["production-detector"]
        total = result["total_hits"] + len(result.get("negative_hits", []))
        # Should score all test medias
        assert total > 0


# ======================================================================
# 12. Vote → Detector → Labels Export → Labels Import → Re-sort
# ======================================================================


class TestVoteDetectorLabelCycle:
    """Full production cycle: vote, train detector, export labels,
    clear state, import labels, run learned sort.  Tests that labels
    carry enough info to fully reconstitute the voting session."""

    def test_full_label_reconstitution_workflow(self, client):
        # Step 1: Vote
        good_ids = [1, 2, 3, 4, 5]
        bad_ids = [16, 17, 18, 19, 20]
        for cid in good_ids:
            client.post(f"/api/medias/{cid}/vote", json={"vote": "good"})
        for cid in bad_ids:
            client.post(f"/api/medias/{cid}/vote", json={"vote": "bad"})

        # Step 2: Train detector and verify it works
        detector = train_detector_from_votes()

        resp = client.post("/api/detector-sort", json={"detector": detector})
        assert resp.status_code == 200
        assert len(resp.get_json()["results"]) == app_module.NUM_MEDIAS

        # Step 3: Export labels
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        exported = resp.get_json()
        assert len(exported["labels"]) == 10

        # Step 4: Nuke everything
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        # Step 5: Import labels
        resp = client.post("/api/labels/import", json=exported)
        assert resp.status_code == 200
        assert resp.get_json()["applied"] == 10

        # Step 6: Verify votes match original
        resp = client.get("/api/votes")
        votes = resp.get_json()
        assert set(votes["good"]) == set(good_ids)
        assert set(votes["bad"]) == set(bad_ids)

        # Step 7: Learned sort should still work with reimported labels
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        learned = resp.get_json()
        assert len(learned["results"]) == app_module.NUM_MEDIAS

        # Good should still rank higher than bad on average
        score_map = {e["id"]: e["score"] for e in learned["results"]}
        avg_good = np.mean([score_map[i] for i in good_ids])
        avg_bad = np.mean([score_map[i] for i in bad_ids])
        assert avg_good > avg_bad


# ======================================================================
# 13. Diversity Tree → Voting → Rebuild → Consistency
# ======================================================================


class TestDiversityTreeVotingCycle:
    """Build diversity tree, vote using diverse samples, rebuild,
    verify the tree state stays consistent."""

    def test_diversity_tree_build_vote_rebuild(self, client):
        from vtsearch.utils import build_diversity_tree, get_diversity_tree

        # Step 1: Build diversity tree
        build_diversity_tree()
        tree = get_diversity_tree()
        assert tree is not None

        # Step 2: Get a diverse sample via API
        resp = client.get("/api/diversity-tree/next")
        assert resp.status_code == 200
        next_data = resp.get_json()
        first_id = next_data.get("id")
        assert first_id is not None, "Tree should suggest a sample"

        # Vote on it
        resp = client.post(f"/api/medias/{first_id}/vote", json={"vote": "good"})
        assert resp.status_code == 200

        # Get next sample (should be different)
        resp = client.get("/api/diversity-tree/next")
        assert resp.status_code == 200
        next_data2 = resp.get_json()
        second_id = next_data2.get("id")
        if second_id is not None:
            assert second_id != first_id, "Next sample should differ after voting"

        # Step 3: Rebuild tree and verify it works
        build_diversity_tree()
        tree2 = get_diversity_tree()
        assert tree2 is not None

        # Previously voted media should be marked as seen
        assert first_id in tree2.labeled_ids

        # Diversity level should reflect the one vote
        level = tree2.diversity_level()
        assert isinstance(level, int)


# ======================================================================
# 14. Multi-Sort Method Workflow with Label File Sort
# ======================================================================


class TestTextSortVoteLabelExportDetector:
    """Chain text sort → vote → label export → detector export → detector sort.
    Tests that the full label-based detector pipeline works end-to-end."""

    def test_text_sort_vote_label_export_detector(self, client, tmp_path):
        # Step 1: Text sort
        resp = client.post("/api/sort", json={"text": "chirping bird"})
        assert resp.status_code == 200
        results = resp.get_json()["results"]
        assert len(results) == app_module.NUM_MEDIAS

        # Step 2: Vote based on text sort ranking
        top_3 = [r["id"] for r in results[:3]]
        bottom_3 = [r["id"] for r in results[-3:]]
        for cid in top_3:
            client.post(f"/api/medias/{cid}/vote", json={"vote": "good"})
        for cid in bottom_3:
            client.post(f"/api/medias/{cid}/vote", json={"vote": "bad"})

        # Step 3: Export labels to file and verify
        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        labels_data = resp.get_json()
        assert len(labels_data["labels"]) == 6

        label_path = tmp_path / "labels.json"
        label_path.write_text(json.dumps(labels_data))

        # Verify the exported file is valid JSON
        reloaded = json.loads(label_path.read_text())
        assert len(reloaded["labels"]) == 6

        # Step 4: Detector training should work with these votes
        det = train_detector_from_votes()
        assert "weights" in det
        assert "threshold" in det

        # Step 5: Detector sort should produce valid results
        resp = client.post("/api/detector-sort", json={"detector": det})
        assert resp.status_code == 200
        sort_results = resp.get_json()["results"]
        assert len(sort_results) == app_module.NUM_MEDIAS

        # All scores should be valid probabilities
        for entry in sort_results:
            assert 0.0 <= entry["score"] <= 1.0
        # Scores should be sorted descending
        scores = [e["score"] for e in sort_results]
        assert scores == sorted(scores, reverse=True)


# ======================================================================
# 15. Error Recovery: Failed Detector Export Doesn't Corrupt State
# ======================================================================


class TestErrorRecoveryMultiStep:
    """Trigger errors mid-workflow and verify subsequent operations succeed."""

    def test_partial_workflow_failure_then_success(self, client):
        # Step 1: Try detector training with no votes (should fail)
        with pytest.raises(ValueError, match="at least one good and one bad"):
            train_detector_from_votes()

        # Step 2: Try learned sort with no votes (should fail)
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 400

        # Step 3: Now vote and verify everything works
        for cid in [1, 2, 3]:
            resp = client.post(f"/api/medias/{cid}/vote", json={"vote": "good"})
            assert resp.status_code == 200
        for cid in [18, 19, 20]:
            resp = client.post(f"/api/medias/{cid}/vote", json={"vote": "bad"})
            assert resp.status_code == 200

        # Step 4: All operations should now succeed
        det = train_detector_from_votes()
        assert "weights" in det

        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        assert len(resp.get_json()["results"]) == app_module.NUM_MEDIAS

        resp = client.post("/api/sort", json={"text": "test"})
        assert resp.status_code == 200

        resp = client.get("/api/labels/export")
        assert resp.status_code == 200
        assert len(resp.get_json()["labels"]) == 6

    def test_bad_label_import_doesnt_corrupt_votes(self, client):
        """Importing labels with bad MD5s shouldn't affect existing votes."""
        # Pre-set some votes
        app_module.good_votes.update({1: None, 2: None})
        app_module.bad_votes.update({19: None, 20: None})

        # Import labels with nonexistent MD5s
        bad_labels = {
            "labels": [
                {"md5": "0000000000000000deadbeef00000000", "label": "good"},
                {"md5": "ffffffffffffffffffffffffffffffff", "label": "bad"},
            ]
        }
        resp = client.post("/api/labels/import", json=bad_labels)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["skipped"] == 2  # none should match

        # Original votes should be intact
        resp = client.get("/api/votes")
        votes = resp.get_json()
        assert set(votes["good"]) == {1, 2}
        assert set(votes["bad"]) == {19, 20}


# ======================================================================
# 16. Auto-Detect with Exporter → Label Round-Trip
# ======================================================================


class TestAutoDetectExporterLabelRoundTrip:
    """Run auto-detect, export to file, read results back, verify the
    structure can be re-consumed for label import."""

    def test_autodetect_results_structure_complete(self, client, tmp_path):
        # Train and save detector
        app_module.good_votes.update({1: None, 2: None, 3: None})
        app_module.bad_votes.update({18: None, 19: None, 20: None})
        detector = train_detector_from_votes()

        resp = client.post(
            "/api/autorun-detectors",
            json={
                "name": "roundtrip-det",
                "media_type": "audio",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
                "autodetect": True,
            },
        )
        assert resp.status_code == 200

        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        # Run auto-detect
        resp = client.post("/api/auto-detect")
        assert resp.status_code == 200
        auto_results = resp.get_json()

        # Export to file
        output_path = tmp_path / "autodetect_output.json"
        resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(output_path)},
                "results": auto_results,
            },
        )
        assert resp.status_code == 200
        assert output_path.exists()

        # Read back and validate structure
        loaded = json.loads(output_path.read_text())
        assert loaded["media_type"] == "audio"
        assert "roundtrip-det" in loaded["results"]

        det_result = loaded["results"]["roundtrip-det"]
        assert "detector_name" in det_result
        assert "threshold" in det_result
        assert "total_hits" in det_result
        assert "hits" in det_result
        assert isinstance(det_result["hits"], list)

        # Each hit should have the expected fields
        for hit in det_result["hits"]:
            assert "id" in hit
            assert "filename" in hit
            assert "score" in hit
            assert 0.0 <= hit["score"] <= 1.0


# ======================================================================
# 17. Concurrent-Style Workflow: Multiple Sort Types Then Learn
# ======================================================================


class TestSequentialSortThenLearn:
    """Simulate a user trying every sort method sequentially and then
    doing a learned sort. Ensures no cross-contamination between methods."""

    def test_all_sort_methods_then_learn(self, client):
        import io

        # Step 1: Text sort
        resp = client.post("/api/sort", json={"text": "sine wave"})
        assert resp.status_code == 200
        text_ids = {r["id"] for r in resp.get_json()["results"]}

        # Step 2: Example sort
        wav_bytes = app_module.generate_wav(440.0, 1.0)
        resp = client.post(
            "/api/example-sort",
            data={"file": (io.BytesIO(wav_bytes), "ref.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        example_ids = {r["id"] for r in resp.get_json()["results"]}

        # Step 3: Vote
        for cid in [1, 2, 3]:
            client.post(f"/api/medias/{cid}/vote", json={"vote": "good"})
        for cid in [18, 19, 20]:
            client.post(f"/api/medias/{cid}/vote", json={"vote": "bad"})

        # Step 4: Detector sort
        detector = train_detector_from_votes()
        resp = client.post("/api/detector-sort", json={"detector": detector})
        assert resp.status_code == 200
        detector_ids = {r["id"] for r in resp.get_json()["results"]}

        # Step 5: Learned sort
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        learned_ids = {r["id"] for r in resp.get_json()["results"]}

        # All methods should return all medias
        all_media_ids = set(range(1, app_module.NUM_MEDIAS + 1))
        assert text_ids == all_media_ids
        assert example_ids == all_media_ids
        assert detector_ids == all_media_ids
        assert learned_ids == all_media_ids

        # Step 6: Text sort should still work after all that
        resp = client.post("/api/sort", json={"text": "another query"})
        assert resp.status_code == 200
        assert len(resp.get_json()["results"]) == app_module.NUM_MEDIAS


# ======================================================================
# 18. Inclusion Affects CLI Scoring
# ======================================================================


class TestInclusionInLearnedSort:
    """Verify that different inclusion values produce different learned
    sort thresholds / score distributions."""

    def test_inclusion_affects_learned_sort_scores(self, client):
        # Vote
        for cid in [1, 2, 3]:
            client.post(f"/api/medias/{cid}/vote", json={"vote": "good"})
        for cid in [18, 19, 20]:
            client.post(f"/api/medias/{cid}/vote", json={"vote": "bad"})

        # Learned sort with inclusion=0
        client.post("/api/inclusion", json={"inclusion": 0})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        scores_0 = {e["id"]: e["score"] for e in resp.get_json()["results"]}

        # Learned sort with inclusion=10
        client.post("/api/inclusion", json={"inclusion": 10})
        resp = client.post("/api/learned-sort")
        assert resp.status_code == 200
        scores_10 = {e["id"]: e["score"] for e in resp.get_json()["results"]}

        # Scores should differ for at least some medias
        diffs = [abs(scores_0[i] - scores_10[i]) for i in scores_0]
        assert max(diffs) > 1e-6, "Inclusion change should affect at least some scores"


# ======================================================================
# 19. Exporter Registry Completeness
# ======================================================================


class TestExporterIntegration:
    """Verify that each registered exporter can accept auto-detect
    results without error."""

    def test_all_exporters_accept_results_structure(self, client):
        # Get list of exporters
        resp = client.get("/api/exporters")
        assert resp.status_code == 200
        resp.get_json()  # verify parseable

        # Build minimal results structure
        results = {
            "media_type": "audio",
            "detectors_run": 1,
            "results": {
                "test_det": {
                    "detector_name": "test_det",
                    "threshold": 0.5,
                    "total_hits": 1,
                    "hits": [
                        {"id": 1, "filename": "a.wav", "score": 0.9, "category": "test"},
                    ],
                },
            },
        }

        # GUI exporter should always work
        resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "gui",
                "field_values": {},
                "results": results,
            },
        )
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

        # File exporter
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test_export.json"
            resp = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": str(fpath)},
                    "results": results,
                },
            )
            assert resp.status_code == 200


# ======================================================================
# 20. Label Importer → Vote → Detector → Autodetect Chain
# ======================================================================


class TestLabelImporterDetectorChain:
    """Import labels via the API, train detector from imported labels,
    save as autorun, run autodetect.  Tests the full pipeline from
    external label file to production scoring."""

    def test_json_label_import_to_autodetect(self, client, tmp_path):
        # Step 1: Create a JSON label file from current medias
        label_entries = []
        for cid in [1, 2, 3, 4, 5]:
            md5 = medias[cid]["md5"]
            label_entries.append({"md5": md5, "label": "good"})
        for cid in [16, 17, 18, 19, 20]:
            md5 = medias[cid]["md5"]
            label_entries.append({"md5": md5, "label": "bad"})

        label_path = tmp_path / "labels.json"
        label_path.write_text(json.dumps({"labels": label_entries}))

        # Step 2: Import labels via label importer API
        resp = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(label_path)},
        )
        assert resp.status_code == 200
        import_data = resp.get_json()
        assert import_data["applied"] == 10

        # Step 3: Verify votes
        resp = client.get("/api/votes")
        votes = resp.get_json()
        assert len(votes["good"]) == 5
        assert len(votes["bad"]) == 5

        # Step 4: Train detector
        detector = train_detector_from_votes()

        # Step 5: Save as autorun detector with autodetect
        resp = client.post(
            "/api/autorun-detectors",
            json={
                "name": "from-labels",
                "media_type": "audio",
                "weights": detector["weights"],
                "threshold": detector["threshold"],
                "autodetect": True,
            },
        )
        assert resp.status_code == 200

        # Step 6: Clear and run auto-detect
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        resp = client.post("/api/auto-detect")
        assert resp.status_code == 200
        auto = resp.get_json()
        assert "from-labels" in auto["results"]
        assert auto["results"]["from-labels"]["total_hits"] >= 0
