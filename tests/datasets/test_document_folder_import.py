"""Folder Ingest on PDFs, end to end (#5680).

Every way of asking for a folder of PDFs must end in one image dataset holding
each page once: the document row the Add Dataset dialog sends, a bare
``media_type`` of ``image`` or of ``document``, and a multi-dataset output.
The first used to give every page twice, the third an empty load.
"""

from __future__ import annotations

import time

import pytest

from vtscore.concurrency.progress import loading_tasks

fitz = pytest.importorskip("fitz")

_DOCUMENT_ROW = [{"source_type": "document", "converter": "document2image", "params": {}}]


def _write_pdf(path, pages: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 100), f"{path.stem} page {i + 1}", fontsize=20)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def pdf_folder(tmp_path):
    folder = tmp_path / "pdfs"
    _write_pdf(folder / "a.pdf", 3)
    _write_pdf(folder / "sub" / "b.pdf", 2)
    return folder


@pytest.fixture(autouse=True)
def _bulk_embed_through_the_stub(monkeypatch):
    """Route the image/text embedders' own bulk paths through the stubbed ``embed_media``."""
    from vtscore.media import all_embedders

    for emb in all_embedders():
        monkeypatch.setattr(emb, "_embed_media_bulk_impl", lambda ms, emb=emb: [emb.embed_media(m) for m in ms])


def _finish(resp) -> dict:
    """Wait for the import *resp* started; return its one task."""
    assert resp.status_code == 200, resp.get_data(as_text=True)
    (task_id,) = resp.get_json()["task_ids"]
    deadline = time.monotonic() + 60
    while not loading_tasks.is_finished(task_id) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert loading_tasks.is_finished(task_id), "import never finished"
    (task,) = [t for t in loading_tasks.list_tasks() if t["task_id"] == task_id]
    return task


def _medias(client, dataset_id: str) -> list[dict]:
    headers = {"X-Dataset-Id": dataset_id}
    ids = [m["id"] for m in client.get("/api/medias/ids", headers=headers).get_json()]
    return client.post("/api/medias/batch", json={"ids": ids}, headers=headers).get_json()


@pytest.mark.parametrize(
    "request_fields",
    [
        {"media_type": "image", "source_specs": _DOCUMENT_ROW},
        {"media_type": "image", "source_specs": _DOCUMENT_ROW, "reference_files": "true"},
        {"media_type": "image"},
        {"media_type": "document"},
        {"outputs": [{"media_type": "image", "category": "document", "source_specs": _DOCUMENT_ROW}]},
    ],
    ids=["document-row", "document-row-referenced", "bare-image", "bare-document", "multi-dataset-output"],
)
def test_a_pdf_folder_becomes_one_page_per_image(client, pdf_folder, request_fields):
    task = _finish(
        client.post(
            "/api/dataset/import/server_folder",
            json={"path": str(pdf_folder), "autofind": "false", **request_fields},
        )
    )

    assert task["error"] is None
    medias = _medias(client, task["dataset_id"])
    assert len(medias) == 5, "3 + 2 pages, each once"
    assert {m["media_type"] for m in medias} == {"image"}
    assert {m["custom_metadata"]["Derived Via"] for m in medias} == {"Document → Images"}

    headers = {"X-Dataset-Id": task["dataset_id"]}
    for media in medias:
        thumb = client.get(f"/api/medias/{media['id']}/thumbnail", headers=headers)
        assert thumb.status_code == 200 and thumb.mimetype.startswith("image/")


def test_a_document_demo_with_no_converter_named_loads_its_pages(client, tmp_path, monkeypatch):
    """The demo's own type has no embedder, so "no conversion" used to load nothing."""
    import vtscore.datasets.downloader as downloader
    import vtscore.datasets.loader_demo as loader_demo

    for category in ("Tobacco", "Food"):
        # Named per category: identical pages would be merged as duplicates.
        _write_pdf(tmp_path / "docs" / category / f"{category.lower()}.pdf", 2)
    monkeypatch.setattr(downloader, "download_ucsf_documents", lambda categories, **kwargs: tmp_path / "docs")
    monkeypatch.setattr(loader_demo, "EMBEDDINGS_DIR", tmp_path / "embeddings")

    task = _finish(client.post("/api/dataset/load-demo", json={"name": "ucsf_documents_a", "autofind": "false"}))

    assert task["error"] is None
    assert task["media_type"] == "image"
    medias = _medias(client, task["dataset_id"])
    assert len(medias) == 4
    assert {m["media_type"] for m in medias} == {"image"}


def test_a_demo_whose_download_fails_says_why(client, tmp_path, monkeypatch):
    """Not "Import produced no medias": the downloader's own account of what went wrong."""
    import vtscore.datasets.downloader as downloader
    import vtscore.datasets.loader_demo as loader_demo

    def refuse(categories, **kwargs):
        raise RuntimeError("No UCSF Industry Documents could be downloaded: downloading x.pdf failed (HTTP 403).")

    monkeypatch.setattr(downloader, "download_ucsf_documents", refuse)
    monkeypatch.setattr(loader_demo, "EMBEDDINGS_DIR", tmp_path / "embeddings")

    task = _finish(
        client.post(
            "/api/dataset/load-demo",
            json={"name": "ucsf_documents_a", "converter": "document2image", "autofind": "false"},
        )
    )

    assert "HTTP 403" in (task["error"] or "")
