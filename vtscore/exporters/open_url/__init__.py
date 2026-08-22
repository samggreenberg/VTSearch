"""Open-in-website exporter – formats the labelset into a URL for the browser to open.

The point of this exporter is the third-party site that has no ingest API: you
can't POST a labelset to it, but you *can* link into it, because its viewer
takes identifiers in the query string.  So instead of delivering the labelset
anywhere, this exporter formats it into that site's own URL and returns it as
``open_url``; the frontend opens the result in a new tab.

Nothing is sent from the server — no network call happens here at all.  The
only thing that leaves the machine is the URL the user's browser then requests,
which means whatever you encode into the template is visible to the target site
and lands in the user's browser history.  Keep it to identifiers.

Example template::

    https://example.com/review?ids={ids}&from=vtsearch

with ``id_field=md5`` and 3 good labels selected yields::

    https://example.com/review?ids=a1b2%2Cc3d4%2Ce5f6&from=vtsearch
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from vtscore.exporters.base import PluginField, ResultsExporter
from vtscore.security.url_validation import validate_browser_url

#: Practical ceiling on a URL the browser will actually follow.  The HTTP spec
#: sets no limit, but IE's historical 2083 became the de-facto floor and plenty
#: of servers/CDNs still refuse longer request lines.  Exceeding it is reported
#: as an error rather than silently truncated, because a half-encoded id list
#: would open the target site on the *wrong* selection.
MAX_URL_LENGTH = 2000

#: Fallback cap on how many identifiers are substituted into ``{ids}`` when the
#: user leaves ``max_items`` blank.
DEFAULT_MAX_ITEMS = 100

#: Every character RFC 3986 allows to appear literally in a URL, including
#: ``%`` so already-encoded sequences survive a second pass.  Anything outside
#: this set is percent-encoded — which matters because the framework
#: substitutes ``{detector_name}`` into the template verbatim, and a detector
#: named "Bird Calls" would otherwise put a raw space in the URL.
_URL_SAFE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"


def _encode_unsafe_chars(url: str) -> str:
    """Percent-encode characters that can't appear literally in a URL.

    Control characters are rejected rather than encoded: they have no
    legitimate place in a template, and silently turning one into ``%0A``
    would hide a malformed template instead of surfacing it.
    """
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):
        raise ValueError("URL Template must not contain control characters.")
    return quote(url, safe=_URL_SAFE_CHARS)


class OpenUrlResultsExporter(ResultsExporter):
    """Format the labelset into a URL and hand it to the browser to open.

    Fills ``{ids}`` (and ``{count}``) in a user-supplied URL template with the
    exported items' identifiers and returns the result as ``open_url``.  A
    template with no placeholder is fine too — it just opens the site.
    """

    name = "open_url"
    display_name = "Open in Website"
    description = "Format the labelset into a URL and open it in a new browser tab."
    icon = "\U0001f517"
    opens_url = True
    fields = [
        PluginField(
            key="url_template",
            label="URL Template",
            field_type="text",
            description=(
                "The URL to open. {ids} is replaced with the selected items' identifiers "
                "(URL-encoded, joined by the separator) and {count} with how many were included."
            ),
            placeholder="https://example.com/review?ids={ids}",
            hint=(
                "Only http:// and https:// URLs are allowed. Everything you put in the "
                "template is visible to the destination site and is recorded in your "
                "browser history, so keep it to identifiers."
            ),
            template_vars=("detector_name",),
        ),
        PluginField(
            key="id_field",
            label="Identifier",
            field_type="select",
            description="Which value from each labeled item to substitute into {ids}.",
            options=["md5", "filename", "origin_name", "category"],
            default="md5",
            required=False,
        ),
        PluginField(
            key="separator",
            label="Separator",
            field_type="text",
            description="Joins the identifiers in {ids}. URL-encoded along with them.",
            default=",",
            required=False,
        ),
        PluginField(
            key="max_items",
            label="Max items",
            field_type="number",
            description=(
                "How many identifiers to include before truncating. URLs have a practical "
                f"length limit of ~{MAX_URL_LENGTH} characters."
            ),
            default=str(DEFAULT_MAX_ITEMS),
            min="1",
            step="1",
            required=False,
        ),
    ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _items_from_find_results(results: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten a scored run's per-detector hit lists into one item list."""
        items: list[dict[str, Any]] = []
        for det_result in (results.get("results") or {}).values():
            if isinstance(det_result, dict):
                items.extend(h for h in det_result.get("hits") or [] if isinstance(h, dict))
        return items

    @staticmethod
    def _items_from_labelset(labelset: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a labelset's entries as the item list."""
        return [e for e in labelset.get("labels") or [] if isinstance(e, dict)]

    @staticmethod
    def _identifier(item: dict[str, Any], id_field: str) -> str:
        """Return *item*'s identifier under *id_field*, or ``""`` if absent."""
        value = item.get(id_field)
        if value is None:
            value = (item.get("custom_metadata") or {}).get(id_field)
        return "" if value is None else str(value)

    @staticmethod
    def _resolve_max_items(value: Any) -> int:
        """Coerce the ``max_items`` field value to a positive ``int``."""
        if value is None or value == "":
            return DEFAULT_MAX_ITEMS
        try:
            n = int(value)
        except (TypeError, ValueError):
            return DEFAULT_MAX_ITEMS
        return n if n > 0 else DEFAULT_MAX_ITEMS

    def _build_url(self, items: list[dict[str, Any]], field_values: dict[str, Any]) -> tuple[str, int, int]:
        """Return ``(url, included_count, total_count)`` for *items*.

        Both payload kinds reduce to a flat list of item dicts before they get
        here, so the URL is built the same way whether the identifiers came
        from a scored run's hits or a labelset's entries.

        Raises:
            ValueError: If the template is empty, the formatted URL fails
                :func:`validate_browser_url`, or it exceeds
                :data:`MAX_URL_LENGTH`.
        """
        template = str(field_values.get("url_template") or "").strip()
        if not template:
            raise ValueError("URL Template is required.")

        id_field = str(field_values.get("id_field") or "md5")
        separator = field_values.get("separator")
        separator = "," if separator is None or separator == "" else str(separator)
        max_items = self._resolve_max_items(field_values.get("max_items"))

        identifiers = [i for i in (self._identifier(item, id_field) for item in items) if i]
        total = len(identifiers)
        included = identifiers[:max_items]

        # Encode the joined string as a single query-parameter value: the
        # separator has to survive as a literal, so no character is `safe`.
        ids_param = quote(separator.join(included), safe="")
        url = template.replace("{ids}", ids_param).replace("{count}", str(len(included)))

        url = validate_browser_url(_encode_unsafe_chars(url))
        if len(url) > MAX_URL_LENGTH:
            raise ValueError(
                f"The formatted URL is {len(url)} characters, over the ~{MAX_URL_LENGTH} "
                f"practical limit. Lower 'Max items' (currently {max_items}) or export fewer labels."
            )
        return url, len(included), total

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _opened(self, items: list[dict[str, Any]], field_values: dict[str, Any]) -> dict[str, Any]:
        """Build the URL for *items* and describe it as a tab about to open."""
        url, included, total = self._build_url(items, field_values)
        detail = f"first {included} of {total} item(s)" if included < total else f"{included} item(s)"
        return {
            "message": f"Opening {detail} in a new tab.",
            "open_url": url,
            "included_count": included,
            "total_count": total,
        }

    def export_find_results(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Format the run's hits into the target site's URL."""
        return self._opened(self._items_from_find_results(results), field_values)

    def export_labelset(self, labelset: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Format the labelset's entries into the target site's URL."""
        return self._opened(self._items_from_labelset(labelset), field_values)

    def export_cli(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        """Return the formatted URL — there is no browser here to open it.

        The CLI prints the ``open_url`` of *any* exporter under the
        confirmation message, so this doesn't write to stdout itself: doing so
        would both duplicate that line and put prose in the middle of the
        NDJSON stream under ``--progress-format json``.
        """
        url, included, total = self._build_url(self._items_from_find_results(results), field_values)
        detail = f"first {included} of {total} item(s)" if included < total else f"{included} item(s)"
        return {
            "message": f"Formatted a URL covering {detail}.",
            "open_url": url,
            "included_count": included,
            "total_count": total,
        }


EXPORTER = OpenUrlResultsExporter()
