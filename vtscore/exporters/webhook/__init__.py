"""Webhook exporter – POSTs auto-detect results to an arbitrary URL.

Requires only ``requests``, which is already a core dependency.
"""

from __future__ import annotations

from typing import Any, Iterator

import requests

from vtscore.exporters.base import PluginField, LabelsetExporter, resolve_stream_batch_size


class WebhookLabelsetExporter(LabelsetExporter):
    """POST the results JSON to a user-specified URL.

    Enables integration with automation platforms (Zapier, n8n, Make,
    custom services) without writing a dedicated exporter.  The full
    results dict is sent as the JSON request body.
    """

    name = "webhook"
    display_name = "Webhook (HTTP POST)"
    description = "POST the results as JSON to a URL."
    icon = "\U0001f310"
    fields = [
        PluginField(
            key="url",
            label="Webhook URL",
            field_type="url",
            description="The URL to POST the results JSON to.",
            placeholder="https://example.com/webhook",
        ),
        PluginField(
            key="auth_header",
            label="Authorization Header",
            field_type="password",
            description="Optional Bearer token or API key sent as the Authorization header.",
            required=False,
        ),
        PluginField(
            key="batch_size",
            label="Batch size (streaming)",
            field_type="number",
            required=False,
            default="500",
            min="1",
            step="1",
            description=(
                "When used with --stream-results, hits are POSTed in batches of this many so a "
                "run larger than RAM never buffers the whole result set. Ignored for one-shot exports."
            ),
        ),
    ]

    def _headers(self, field_values: dict[str, Any]) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        auth_header = field_values.get("auth_header", "")
        if auth_header:
            headers["Authorization"] = auth_header
        return headers

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        url = field_values["url"]
        headers = self._headers(field_values)

        resp = requests.post(url, json=results, headers=headers, timeout=30, allow_redirects=False)
        resp.raise_for_status()

        if "labels" in results:
            total_items = len(results.get("labels", []))
            detail = f"Posted {total_items} label(s)"
        else:
            total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
            detail = f"Posted {total_hits} hit(s) across {results.get('detectors_run', 0)} detector(s)"
        return {
            "message": (f"{detail} to {url} (HTTP {resp.status_code})."),
            "status_code": resp.status_code,
            "url": url,
        }

    @property
    def supports_streaming(self) -> bool:
        return True

    def export_cli_streaming(
        self,
        header: dict[str, Any],
        records: Iterator[tuple[str, dict[str, Any]]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """POST hits in fixed-size batches as scored chunks stream in.

        Instead of buffering the whole result set into a single request body
        (which defeats ``--stream-results``), hits are accumulated up to
        ``batch_size`` and each full batch is POSTed as its own request, then
        dropped. Peak memory stays bounded by ``batch_size`` regardless of how
        many hits the run produces. Each batch body is a JSON object carrying
        the run metadata, a zero-based ``batch_index``, and a ``hits`` array
        (each hit with its ``detector`` name merged in). A single request is
        always sent even when zero hits match, so the receiver learns the run
        happened.
        """
        url = field_values["url"]
        headers = self._headers(field_values)
        batch_size = resolve_stream_batch_size(field_values.get("batch_size"))

        meta = {
            "format": "vtsearch-hits-batch/v1",
            "media_type": header.get("media_type", "unknown"),
            "detectors": header.get("detectors", []),
            "keep_negatives": bool(header.get("keep_negatives", False)),
        }

        total_hits = 0
        batches = 0

        def _post(hits: list[dict[str, Any]]) -> None:
            nonlocal batches
            payload = {**meta, "batch_index": batches, "hits": hits}
            resp = requests.post(url, json=payload, headers=headers, timeout=30, allow_redirects=False)
            resp.raise_for_status()
            batches += 1

        batch: list[dict[str, Any]] = []
        for detector_name, hit in records:
            batch.append({"detector": detector_name, **hit})
            total_hits += 1
            if len(batch) >= batch_size:
                _post(batch)
                batch = []
        # Flush the trailing partial batch; also fire once for an empty run so
        # the receiver always gets at least one POST.
        if batch or batches == 0:
            _post(batch)

        return {
            "message": (f"Streamed {total_hits} hit(s) to {url} in {batches} batch(es)."),
            "url": url,
        }


EXPORTER = WebhookLabelsetExporter()
