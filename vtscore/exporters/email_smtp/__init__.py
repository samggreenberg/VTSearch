"""Email exporter – sends auto-detect results directly via MX lookup.

Connects directly to the recipient's mail server (via DNS MX record lookup)
so no SMTP credentials or server configuration are needed.  The sender
address is supplied by the caller - a real domain you control is required,
because most MX hosts reject mail whose sender domain has no DNS records.

Requires the ``dnspython`` package for MX record resolution.
"""

from __future__ import annotations

import re
import smtplib
from collections import defaultdict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Iterator

from vtscore.exporters.base import PluginField, ResultsExporter, iter_stream_batches, resolve_stream_batch_size

# Pragmatic address check: a non-empty local part, a single ``@``, and a
# dotted domain, none of which may contain whitespace.  Not full RFC 5322
# (that's effectively un-regexable), but enough to reject the addresses the
# bare ``"@" in`` test let through — ``"@example.com"`` (empty local part),
# ``"foo@"`` (no domain), ``"foo@bar"`` (undotted domain) — before we spend
# a DNS MX lookup and hand the address to a remote SMTP server.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _is_valid_email(addr: str) -> bool:
    """Return ``True`` if *addr* looks like a deliverable email address."""
    return bool(_EMAIL_RE.match(addr or ""))


def _build_plain_text(results: dict[str, Any]) -> str:
    """Render results as a human-readable plain-text summary."""
    lines: list[str] = [
        "Auto-Detect Results",
        "===================",
        f"Media Type:    {results.get('media_type', 'unknown')}",
        f"Detectors Run: {results.get('detectors_run', 0)}",
        "",
    ]
    for det_result in results.get("results", {}).values():
        lines.append(f"--- {det_result['detector_name']} ---")
        lines.append(f"Threshold: {det_result['threshold']}  |  Total Hits: {det_result['total_hits']}")
        if det_result["hits"]:
            for hit in det_result["hits"]:
                lines.append(f"  Clip #{hit['id']}: {hit.get('filename', 'N/A')} (score: {hit['score']})")
        else:
            lines.append("  No positive hits found.")
        lines.append("")
    return "\n".join(lines)


def _build_html(results: dict[str, Any]) -> str:
    """Render results as a minimal HTML e-mail body."""
    from html import escape

    rows = ""
    for det_result in results.get("results", {}).values():
        hits_html = ""
        for hit in det_result["hits"]:
            hits_html += (
                f"<tr><td>Clip #{hit['id']}</td>"
                f"<td>{escape(str(hit.get('filename', 'N/A')))}</td>"
                f"<td>{hit['score']}</td></tr>"
            )
        if not hits_html:
            hits_html = '<tr><td colspan="3"><em>No positive hits found.</em></td></tr>'
        rows += (
            f"<h3>{escape(str(det_result['detector_name']))}</h3>"
            f"<p>Threshold: {det_result['threshold']} &mdash; "
            f"Total Hits: {det_result['total_hits']}</p>"
            f"<table border='1' cellpadding='4' cellspacing='0'>"
            f"<tr><th>Clip</th><th>Filename</th><th>Score</th></tr>"
            f"{hits_html}</table>"
        )
    return (
        f"<html><body>"
        f"<h2>Auto-Detect Results</h2>"
        f"<p><strong>Media Type:</strong> {escape(str(results.get('media_type', 'unknown')))}<br>"
        f"<strong>Detectors Run:</strong> {results.get('detectors_run', 0)}</p>"
        f"{rows}"
        f"</body></html>"
    )


def _default_subject(results: dict[str, Any]) -> str:
    """Auto-generated subject used when the user leaves the field blank."""
    media_type = results.get("media_type", "unknown")
    total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
    return f"VTSearch Auto-Detect: {total_hits} hit(s) on {media_type} dataset"


def _labelset_counts(labels: list[dict[str, Any]]) -> tuple[int, int]:
    """Return ``(good, bad)`` counts for a labelset's entries."""
    good = sum(1 for e in labels if e.get("label") == "good")
    return good, len(labels) - good


def _label_row_values(entry: dict[str, Any]) -> tuple[str, str, str]:
    """Return the ``(label, name, origin)`` cells shown for one label entry.

    ``origin_name`` beats ``filename`` because it is what the item is called
    inside its source, which is the name a recipient can act on; the origin
    itself is rendered through :meth:`~vtscore.datasets.origin.Origin.display`
    so the email says where the item came from rather than dumping a dict.
    """
    from vtscore.datasets.origin import Origin

    origin = entry.get("origin")
    try:
        origin_str = Origin.from_dict(origin).display() if origin else ""
    except Exception:  # noqa: BLE001 - a malformed origin must not sink the email
        origin_str = ""
    name = entry.get("origin_name") or entry.get("filename") or entry.get("md5", "")
    return str(entry.get("label", "")), str(name), origin_str


def _build_labelset_plain(labelset: dict[str, Any]) -> str:
    """Render a labelset as a human-readable plain-text listing."""
    labels = labelset.get("labels", []) or []
    good, bad = _labelset_counts(labels)
    lines: list[str] = [
        "VTSearch Labels",
        "===============",
        f"Labels: {len(labels)}  ({good} good, {bad} bad)",
        "",
    ]
    for entry in labels:
        label, name, origin_str = _label_row_values(entry)
        lines.append(f"  [{label or '?'}] {name}" + (f"  ({origin_str})" if origin_str else ""))
    if not labels:
        lines.append("  No labels in this export.")
    return "\n".join(lines)


def _build_labelset_html(labelset: dict[str, Any]) -> str:
    """Render a labelset as a minimal HTML table."""
    from html import escape

    labels = labelset.get("labels", []) or []
    good, bad = _labelset_counts(labels)
    rows = ""
    for entry in labels:
        label, name, origin_str = _label_row_values(entry)
        rows += f"<tr><td>{escape(label)}</td><td>{escape(name)}</td><td>{escape(origin_str)}</td></tr>"
    if not rows:
        rows = '<tr><td colspan="3"><em>No labels in this export.</em></td></tr>'
    return (
        f"<html><body>"
        f"<h2>VTSearch Labels</h2>"
        f"<p><strong>Labels:</strong> {len(labels)} "
        f"({good} good, {bad} bad)</p>"
        f"<table border='1' cellpadding='4' cellspacing='0'>"
        f"<tr><th>Label</th><th>Name</th><th>Origin</th></tr>"
        f"{rows}</table>"
        f"</body></html>"
    )


def _default_labelset_subject(labelset: dict[str, Any]) -> str:
    """Auto-generated subject used when the user leaves the field blank."""
    labels = labelset.get("labels", []) or []
    good, bad = _labelset_counts(labels)
    return f"VTSearch Labels: {len(labels)} label(s) ({good} good, {bad} bad)"


def _group_batch_by_detector(batch: list[tuple[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Group a streamed ``(detector_name, hit)`` batch into ``{detector: [hits]}``."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detector_name, hit in batch:
        grouped[detector_name].append(hit)
    return grouped


def _build_batch_plain(media_type: str, batch: list[tuple[str, dict[str, Any]]], part: int) -> str:
    """Render a streamed batch of hits as a plain-text summary."""
    lines: list[str] = [
        f"Auto-Detect Results (streamed, part {part})",
        "=========================================",
        f"Media Type: {media_type}",
        f"Hits in this part: {len(batch)}",
        "",
    ]
    for detector_name, hits in _group_batch_by_detector(batch).items():
        lines.append(f"--- {detector_name} ---")
        for hit in hits:
            clip_id = hit.get("id")
            label = f"Clip #{clip_id}" if clip_id is not None else "Clip"
            lines.append(f"  {label}: {hit.get('filename', 'N/A')} (score: {hit.get('score', 'N/A')})")
        lines.append("")
    return "\n".join(lines)


def _build_batch_html(media_type: str, batch: list[tuple[str, dict[str, Any]]], part: int) -> str:
    """Render a streamed batch of hits as a minimal HTML e-mail body."""
    from html import escape

    rows = ""
    for detector_name, hits in _group_batch_by_detector(batch).items():
        hits_html = ""
        for hit in hits:
            clip_id = hit.get("id")
            label = f"Clip #{clip_id}" if clip_id is not None else "Clip"
            hits_html += (
                f"<tr><td>{escape(label)}</td>"
                f"<td>{escape(str(hit.get('filename', 'N/A')))}</td>"
                f"<td>{escape(str(hit.get('score', 'N/A')))}</td></tr>"
            )
        rows += (
            f"<h3>{escape(str(detector_name))}</h3>"
            f"<table border='1' cellpadding='4' cellspacing='0'>"
            f"<tr><th>Clip</th><th>Filename</th><th>Score</th></tr>"
            f"{hits_html}</table>"
        )
    return (
        f"<html><body>"
        f"<h2>Auto-Detect Results (streamed, part {part})</h2>"
        f"<p><strong>Media Type:</strong> {escape(str(media_type))}<br>"
        f"<strong>Hits in this part:</strong> {len(batch)}</p>"
        f"{rows}"
        f"</body></html>"
    )


def _group_labels_by_detector(batch: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group a streamed batch of label entries into ``{detector: [entries]}``."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in batch:
        grouped[str(entry.get("detector", ""))].append(entry)
    return grouped


def _labelset_batch_subject(media_type: str, batch: list[dict[str, Any]], part: int) -> str:
    """Auto-generated subject for one part of a streamed labelset."""
    good, bad = _labelset_counts(batch)
    return f"VTSearch Labels: {len(batch)} label(s) ({good} good, {bad} bad) on {media_type} dataset (part {part})"


def _build_labelset_batch_plain(media_type: str, batch: list[dict[str, Any]], part: int) -> str:
    """Render a streamed batch of label entries as a plain-text listing, per detector."""
    good, bad = _labelset_counts(batch)
    lines: list[str] = [
        f"VTSearch Labels (streamed, part {part})",
        "======================================",
        f"Media Type: {media_type}",
        f"Labels in this part: {len(batch)}  ({good} good, {bad} bad)",
        "",
    ]
    for detector_name, entries in _group_labels_by_detector(batch).items():
        lines.append(f"--- {detector_name} ---")
        for entry in entries:
            label, name, origin_str = _label_row_values(entry)
            lines.append(f"  [{label or '?'}] {name}" + (f"  ({origin_str})" if origin_str else ""))
        lines.append("")
    if not batch:
        lines.append("  No labels in this part.")
    return "\n".join(lines)


def _build_labelset_batch_html(media_type: str, batch: list[dict[str, Any]], part: int) -> str:
    """Render a streamed batch of label entries as a minimal HTML table per detector."""
    from html import escape

    good, bad = _labelset_counts(batch)
    tables = ""
    for detector_name, entries in _group_labels_by_detector(batch).items():
        rows = ""
        for entry in entries:
            label, name, origin_str = _label_row_values(entry)
            rows += f"<tr><td>{escape(label)}</td><td>{escape(name)}</td><td>{escape(origin_str)}</td></tr>"
        tables += (
            f"<h3>{escape(detector_name)}</h3>"
            f"<table border='1' cellpadding='4' cellspacing='0'>"
            f"<tr><th>Label</th><th>Name</th><th>Origin</th></tr>"
            f"{rows}</table>"
        )
    if not batch:
        tables = "<p><em>No labels in this part.</em></p>"
    return (
        f"<html><body>"
        f"<h2>VTSearch Labels (streamed, part {part})</h2>"
        f"<p><strong>Media Type:</strong> {escape(str(media_type))}<br>"
        f"<strong>Labels in this part:</strong> {len(batch)} ({good} good, {bad} bad)</p>"
        f"{tables}"
        f"</body></html>"
    )


def _resolve_mx(domain: str) -> str:
    """Return the highest-priority MX host for *domain*."""
    import dns.resolver  # lazy import – dnspython is optional

    answers = dns.resolver.resolve(domain, "MX", lifetime=10)
    best = min(answers, key=lambda r: r.preference)
    return str(best.exchange).rstrip(".")


class EmailResultsExporter(ResultsExporter):
    """Send a scored run or a labelset by e-mail via direct MX delivery.

    Looks up the recipient domain's MX record and connects directly - no
    SMTP credentials or server configuration required.  The caller must
    supply a sender address on a domain they control; receiving MX hosts
    reject mail whose sender domain has no DNS records.
    """

    name = "email_smtp"
    display_name = "Send by Email"
    description = "Email the results or labels to any address."
    icon = "📧"
    fields = [
        PluginField(
            key="from",
            label="Sender email",
            field_type="email",
            description="The email address the results will be sent from.",
            hint=("Must be on a domain you control - most MX hosts reject mail from non-existent domains."),
            placeholder="vtsearch@your-domain.example",
        ),
        PluginField(
            key="to",
            label="Recipient email",
            field_type="email",
            description="The email address to send the results to.",
            placeholder="recipient@example.com",
        ),
        PluginField(
            key="subject",
            label="Subject",
            field_type="text",
            required=False,
            description="Subject line for the email. Leave blank for an auto-generated summary.",
            hint=(
                "Template variables: {YYYYMMDD-HHMMSS}, {YYYYMMDD}, {YYYY}, {MM}, {DD}, {detector_name}, {username}."
            ),
            placeholder="VTSearch Auto-Detect Results {YYYYMMDD}",
            template_vars=("YYYYMMDD-HHMMSS", "YYYYMMDD", "YYYY", "MM", "DD", "detector_name", "username"),
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
                "When used with --stream-results, one email is sent per this many hits so a run "
                "larger than RAM never buffers the whole result set. Ignored for one-shot exports."
            ),
        ),
    ]

    @staticmethod
    def _addresses(field_values: dict[str, Any]) -> tuple[str, str]:
        """Return the validated ``(from, to)`` pair, or raise before anything is sent."""
        from_addr = field_values["from"]
        to_addr = field_values["to"]

        if not _is_valid_email(from_addr):
            raise ValueError("Sender email address is invalid.")
        if not _is_valid_email(to_addr):
            raise ValueError("Recipient email address is invalid.")
        return from_addr, to_addr

    @staticmethod
    def _deliver(from_addr: str, to_addr: str, subject: str, plain: str, html: str) -> str:
        """Resolve the recipient's MX host, deliver one message, and return the host.

        Each message resolves MX afresh and opens its own SMTP connection, so a
        slow scoring run between streamed parts can't leave a connection idle
        long enough to be dropped.
        """
        domain = to_addr.rsplit("@", 1)[1]
        mx_host = _resolve_mx(domain)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.attach(MIMEText(plain, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP(mx_host, 25, timeout=30) as server:
            server.ehlo()
            server.sendmail(from_addr, [to_addr], msg.as_string())
        return mx_host

    def _send(
        self,
        field_values: dict[str, Any],
        default_subject: str,
        plain: str,
        html: str,
        sent_detail: str,
    ) -> dict[str, Any]:
        """Validate the addresses, resolve MX, and deliver one message.

        Shared by both payload kinds: what differs between a scored run and a
        labelset is only the rendering and the fallback subject, never the
        delivery.
        """
        from_addr, to_addr = self._addresses(field_values)
        # The framework substitutes any {template} vars into ``subject`` at
        # ingress (see vtscore.plugins.normalize); a blank field falls back to
        # the auto-generated summary so existing behaviour is preserved.
        mx_host = self._deliver(from_addr, to_addr, field_values.get("subject") or default_subject, plain, html)

        return {
            "message": f"Email with {sent_detail} sent to {to_addr} via {mx_host}.",
            "to": to_addr,
        }

    def _send_batches(
        self,
        field_values: dict[str, Any],
        items: Iterator[Any],
        render: Callable[[list[Any], int], tuple[str, str, str]],
    ) -> tuple[int, int, str]:
        """Email *items* one ``batch_size`` part at a time; return ``(items sent, parts, to)``.

        *render* maps ``(batch, part)`` to that part's ``(default_subject, plain,
        html)``.  Shared by both streams: a part fills, is emailed, and is dropped
        before the next item arrives, so peak memory stays bounded by
        ``batch_size``.  One email is always sent, even for an empty run, so the
        recipient learns the run finished.
        """
        from_addr, to_addr = self._addresses(field_values)
        batch_size = resolve_stream_batch_size(field_values.get("batch_size"))
        # The framework substitutes any {template} vars into ``subject`` at
        # ingress; a blank field falls back to an auto-generated per-part line.
        subject_base = field_values.get("subject") or ""

        total = 0
        parts = 0
        for batch in iter_stream_batches(items, batch_size):
            part = parts + 1
            default_subject, plain, html = render(batch, part)
            subject = f"{subject_base} (part {part})" if subject_base else default_subject
            self._deliver(from_addr, to_addr, subject, plain, html)
            total += len(batch)
            parts += 1
        return total, parts, to_addr

    def export_find_results(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Email a per-detector summary of the scored run."""
        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        return self._send(
            field_values,
            _default_subject(results),
            _build_plain_text(results),
            _build_html(results),
            f"{total_hits} hit(s)",
        )

    def export_labelset(self, labelset: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Email the labelset as a good/bad listing with each item's origin.

        Before this existed the exporter understood only the scored-run shape,
        so a labelset export from the Export modal mailed an empty body under a
        "0 hit(s)" subject and still reported success.
        """
        labels = labelset.get("labels", []) or []
        return self._send(
            field_values,
            _default_labelset_subject(labelset),
            _build_labelset_plain(labelset),
            _build_labelset_html(labelset),
            f"{len(labels)} label(s)",
        )

    @property
    def supports_streaming(self) -> bool:
        return True

    def export_cli_streaming(
        self,
        header: dict[str, Any],
        records: Iterator[tuple[str, dict[str, Any]]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Send one email per fixed-size batch of hits as chunks stream in.

        A single email cannot hold a result set larger than RAM, so streaming
        delivery sends the hits in ``batch_size``-sized parts (see
        :meth:`_send_batches`).
        """
        media_type = header.get("media_type", "unknown")

        def _render(batch: list[tuple[str, dict[str, Any]]], part: int) -> tuple[str, str, str]:
            subject = f"VTSearch Auto-Detect: {len(batch)} hit(s) on {media_type} dataset (part {part})"
            return subject, _build_batch_plain(media_type, batch, part), _build_batch_html(media_type, batch, part)

        total_hits, parts, to_addr = self._send_batches(field_values, records, _render)
        return {
            "message": (f"Streamed {total_hits} hit(s) to {to_addr} in {parts} email(s)."),
            "to": to_addr,
        }

    def export_labelset_streaming(
        self,
        header: dict[str, Any],
        elements: Iterator[dict[str, Any]],
        field_values: dict[str, Any],
    ) -> dict[str, Any]:
        """Send one email per fixed-size batch of label entries, listed per detector.

        The labelset twin of :meth:`export_cli_streaming`, batched the same way
        by ``batch_size``.
        """
        media_type = header.get("media_type", "unknown")

        def _render(batch: list[dict[str, Any]], part: int) -> tuple[str, str, str]:
            return (
                _labelset_batch_subject(media_type, batch, part),
                _build_labelset_batch_plain(media_type, batch, part),
                _build_labelset_batch_html(media_type, batch, part),
            )

        total, parts, to_addr = self._send_batches(field_values, elements, _render)
        return {
            "message": f"Streamed {total} label(s) to {to_addr} in {parts} email(s).",
            "to": to_addr,
        }


EXPORTER = EmailResultsExporter()
