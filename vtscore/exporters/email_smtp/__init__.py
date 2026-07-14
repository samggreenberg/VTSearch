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
from typing import Any, Iterator

from vtscore.exporters.base import PluginField, LabelsetExporter, resolve_stream_batch_size

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


def _resolve_mx(domain: str) -> str:
    """Return the highest-priority MX host for *domain*."""
    import dns.resolver  # lazy import – dnspython is optional

    answers = dns.resolver.resolve(domain, "MX", lifetime=10)
    best = min(answers, key=lambda r: r.preference)
    return str(best.exchange).rstrip(".")


class EmailLabelsetExporter(LabelsetExporter):
    """Send auto-detect results by e-mail via direct MX delivery.

    Looks up the recipient domain's MX record and connects directly - no
    SMTP credentials or server configuration required.  The caller must
    supply a sender address on a domain they control; receiving MX hosts
    reject mail whose sender domain has no DNS records.
    """

    name = "email_smtp"
    display_name = "Send by Email"
    description = "Email the results summary to any address."
    icon = "📧"
    fields = [
        PluginField(
            key="from",
            label="Sender Email",
            field_type="email",
            description="The email address the results will be sent from.",
            hint=("Must be on a domain you control - most MX hosts reject mail from non-existent domains."),
            placeholder="vtsearch@your-domain.example",
        ),
        PluginField(
            key="to",
            label="Recipient Email",
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

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        from_addr = field_values["from"]
        to_addr = field_values["to"]

        if not _is_valid_email(from_addr):
            raise ValueError("Sender email address is invalid.")
        if not _is_valid_email(to_addr):
            raise ValueError("Recipient email address is invalid.")

        domain = to_addr.rsplit("@", 1)[1]
        mx_host = _resolve_mx(domain)

        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        # The framework substitutes any {template} vars into ``subject`` at
        # ingress (see vtscore.plugins.normalize); a blank field falls back to
        # the auto-generated summary so existing behaviour is preserved.
        subject = field_values.get("subject") or _default_subject(results)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr

        plain = _build_plain_text(results)
        html = _build_html(results)
        msg.attach(MIMEText(plain, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP(mx_host, 25, timeout=30) as server:
            server.ehlo()
            server.sendmail(from_addr, [to_addr], msg.as_string())

        return {
            "message": (f"Email with {total_hits} hit(s) sent to {to_addr} via {mx_host}."),
            "to": to_addr,
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
        """Send one email per fixed-size batch of hits as chunks stream in.

        A single email cannot hold a result set larger than RAM, so streaming
        delivery sends the hits in ``batch_size``-sized parts: hits accumulate
        until a batch fills, that batch is emailed, and the buffer is dropped
        before the next hit arrives. Peak memory stays bounded by
        ``batch_size``. Each message resolves the recipient's MX host afresh
        and opens its own SMTP connection, so a slow scoring run between
        batches can't leave a connection idle long enough to be dropped. One
        email is always sent even for an empty run, so the recipient learns the
        run finished.
        """
        from_addr = field_values["from"]
        to_addr = field_values["to"]

        if not _is_valid_email(from_addr):
            raise ValueError("Sender email address is invalid.")
        if not _is_valid_email(to_addr):
            raise ValueError("Recipient email address is invalid.")

        batch_size = resolve_stream_batch_size(field_values.get("batch_size"))
        media_type = header.get("media_type", "unknown")
        # The framework substitutes any {template} vars into ``subject`` at
        # ingress; a blank field falls back to an auto-generated per-part line.
        subject_base = field_values.get("subject") or ""

        total_hits = 0
        parts = 0

        def _send(batch: list[tuple[str, dict[str, Any]]]) -> None:
            nonlocal parts
            part = parts + 1
            if subject_base:
                subject = f"{subject_base} (part {part})"
            else:
                subject = f"VTSearch Auto-Detect: {len(batch)} hit(s) on {media_type} dataset (part {part})"

            domain = to_addr.rsplit("@", 1)[1]
            mx_host = _resolve_mx(domain)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = from_addr
            msg["To"] = to_addr
            msg.attach(MIMEText(_build_batch_plain(media_type, batch, part), "plain", "utf-8"))
            msg.attach(MIMEText(_build_batch_html(media_type, batch, part), "html", "utf-8"))

            with smtplib.SMTP(mx_host, 25, timeout=30) as server:
                server.ehlo()
                server.sendmail(from_addr, [to_addr], msg.as_string())
            parts += 1

        batch: list[tuple[str, dict[str, Any]]] = []
        for detector_name, hit in records:
            batch.append((detector_name, hit))
            total_hits += 1
            if len(batch) >= batch_size:
                _send(batch)
                batch = []
        # Flush the trailing partial batch; also fire once for an empty run so
        # the recipient always gets at least one email.
        if batch or parts == 0:
            _send(batch)

        return {
            "message": (f"Streamed {total_hits} hit(s) to {to_addr} in {parts} email(s)."),
            "to": to_addr,
        }


EXPORTER = EmailLabelsetExporter()
