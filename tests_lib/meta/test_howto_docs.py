"""The how-to pages under ``docs/user/howto/`` work in both places they are read.

Each how-to is one task, click by click, on the user guide's Smiley example.
They are read on GitHub *and* inside the app: the Help panel's User guide tab
opens ``USER_GUIDE.md`` and follows a link to another doc under ``docs/user/``
in place (``keyboard-help-modal.component.ts``, ``resolveDocPath``). That only
works for what these tests pin:

- every how-to is linked from the guide, since the guide is the only way into
  one in-app (the Help panel has no file browser);
- every screenshot is written ``../assets/…``, the one spelling that resolves
  both on GitHub (relative to the page) and in the Help panel (relative to the
  page, then to the served docs folder);
- every relative link to another doc stays inside ``docs/user/``: the Help
  panel hands anything else to the browser, which has no route for it.

Links, anchors and embedded screenshot ids are checked repo-wide by
``scripts/check-docs.py`` and ``scripts/screenshots/wiring-check.py``; this
file covers only what is particular to the how-to folder.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
USER_DOCS = REPO / "docs" / "user"
HOWTO = USER_DOCS / "howto"
GUIDE = USER_DOCS / "USER_GUIDE.md"

#: A fenced code block, stripped before scanning so samples aren't parsed.
_FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
#: Inline code, likewise.
_CODE_RE = re.compile(r"`[^`\n]*`")
#: A markdown link target: ``[text](target)``.
_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
#: An HTML image or ``<source>`` path.
_SRC_RE = re.compile(r"""\b(?:src|srcset)="([^"]+)\"""")


def _howtos() -> list[Path]:
    return sorted(HOWTO.glob("*.md"))


def _prose(path: Path) -> str:
    return _CODE_RE.sub("", _FENCE_RE.sub("", path.read_text(encoding="utf-8")))


def _resolve(from_rel: str, href: str) -> str | None:
    """Python twin of ``resolveDocPath``: *href* in the doc at *from_rel*
    (both relative to ``docs/user/``), or ``None`` if it leaves the folder."""
    parts = from_rel.split("/")[:-1]
    for part in href.split("#", 1)[0].split("?", 1)[0].split("/"):
        if part == "..":
            if not parts:
                return None
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def test_there_are_howtos():
    assert _howtos(), f"no how-to pages under {HOWTO.relative_to(REPO)}"


def test_every_howto_is_linked_from_the_guide():
    guide = GUIDE.read_text(encoding="utf-8")
    missing = [p.name for p in _howtos() if f"](howto/{p.name}" not in guide]
    assert not missing, (
        "how-to pages not linked from docs/user/USER_GUIDE.md (the only way into them in the app's Help panel): "
        + ", ".join(missing)
    )


def test_screenshots_are_written_relative_to_the_page():
    bad = []
    for page in _howtos():
        for src in _SRC_RE.findall(_prose(page)):
            if not src.startswith("../assets/"):
                bad.append(f"{page.name}: {src}")
    assert not bad, "how-to images must be written ../assets/<file>: " + "; ".join(bad)


def test_doc_links_stay_inside_docs_user():
    bad = []
    for page in _howtos():
        rel = page.relative_to(USER_DOCS).as_posix()
        for href in _LINK_RE.findall(_prose(page)):
            if re.match(r"^[a-z]+:", href) or href.startswith(("#", "/")):
                continue
            if ".md" not in href.split("#", 1)[0]:
                continue
            if _resolve(rel, href) is None:
                bad.append(f"{page.name}: {href}")
    assert not bad, (
        "how-to links that leave docs/user/ are dead in the app's Help panel; "
        "name the file in backticks instead: " + "; ".join(bad)
    )
