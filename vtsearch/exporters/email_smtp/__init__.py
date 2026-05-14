"""Email exporter – sends auto-detect results directly via MX lookup.

Connects directly to the recipient's mail server (via DNS MX record lookup)
so no SMTP credentials or server configuration are needed.  The sender
address is supplied by the caller — a real domain you control is required,
because most MX hosts reject mail whose sender domain has no DNS records.

Requires the ``dnspython`` package for MX record resolution.
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from vtsearch.exporters.base import ExporterField, LabelsetExporter


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


def _resolve_mx(domain: str) -> str:
    """Return the highest-priority MX host for *domain*."""
    import dns.resolver  # lazy import – dnspython is optional

    answers = dns.resolver.resolve(domain, "MX", lifetime=10)
    best = min(answers, key=lambda r: r.preference)
    return str(best.exchange).rstrip(".")


class EmailLabelsetExporter(LabelsetExporter):
    """Send auto-detect results by e-mail via direct MX delivery.

    Looks up the recipient domain's MX record and connects directly — no
    SMTP credentials or server configuration required.  The caller must
    supply a sender address on a domain they control; receiving MX hosts
    reject mail whose sender domain has no DNS records.
    """

    name = "email_smtp"
    display_name = "Send by Email"
    description = "Email the results summary to any address."
    icon = "📧"
    fields = [
        ExporterField(
            key="from",
            label="Sender Email",
            field_type="email",
            description=(
                "The email address to send from. Must be on a domain you control — "
                "MX hosts reject mail from non-existent domains."
            ),
            placeholder="vtsearch@your-domain.example",
        ),
        ExporterField(
            key="to",
            label="Recipient Email",
            field_type="email",
            description="The email address to send the results to.",
            placeholder="recipient@example.com",
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        from_addr = field_values.get("from", "").strip()
        to_addr = field_values.get("to", "").strip()

        if not from_addr:
            raise ValueError("Sender email address is required.")
        if "@" not in from_addr:
            raise ValueError("Sender email address is invalid.")

        if not to_addr:
            raise ValueError("Recipient email address is required.")
        if "@" not in to_addr:
            raise ValueError("Recipient email address is invalid.")

        domain = to_addr.rsplit("@", 1)[1]
        mx_host = _resolve_mx(domain)

        media_type = results.get("media_type", "unknown")
        total_hits = sum(r.get("total_hits", 0) for r in results.get("results", {}).values())
        subject = f"VTSearch Auto-Detect: {total_hits} hit(s) on {media_type} dataset"

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


EXPORTER = EmailLabelsetExporter()
