"""Every in-document link in the user guide must resolve to a real heading.

The guide carries a hand-written table of contents plus a dozen body
cross-references, and nothing used to check them.  They rotted exactly the
way that invites: the headings read ``## Autopilot - the guided workflow``,
which GitHub slugs to ``autopilot---the-guided-workflow`` (the spaces *around*
the hyphen each become a hyphen too), while every link was written with two.
Sixteen dead TOC entries, and no reader complaint loud enough to notice.

So this test re-derives each heading's anchor with GitHub's slug rule and
fails on any ``](#...)`` that points nowhere.  It is deliberately
dependency-free (stdlib + the file itself) so it runs in the library tier.

The in-app copy of the guide needs the same slugs — ``marked`` emits no
heading ids, so the Help panel adds them itself.  ``headingSlug`` in
``frontend/src/app/components/modals/keyboard-help-modal/keyboard-help-modal.component.ts``
is the TypeScript twin of :func:`github_slug`; the two must agree, and its
own spec pins the shared cases.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_GUIDE = REPO_ROOT / "docs" / "user" / "USER_GUIDE.md"

#: An ATX heading: capture level and text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
#: An inline link to a same-document anchor, e.g. ``[Find](#find-scoring)``.
_ANCHOR_LINK_RE = re.compile(r"\]\(#([^)]+)\)")
#: A fenced code block, stripped before scanning so samples aren't parsed.
_FENCE_RE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)


def github_slug(text: str) -> str:
    """Return GitHub's heading anchor for *text*.

    Lowercase, drop everything that isn't a word character, hyphen, or space,
    then replace each remaining space with a hyphen.  Space-by-space, not
    run-collapsing: ``"a - b"`` yields ``a---b``, which is the whole reason
    this test exists.
    """
    slug = text.strip().lower()
    slug = re.sub(r"[^\w\- ]+", "", slug, flags=re.UNICODE)
    return slug.replace(" ", "-")


def heading_anchors(markdown: str) -> list[str]:
    """Return every heading's anchor, in document order, with GitHub's dedupe.

    A slug already used gets ``-1``, ``-2``, … appended, matching GitHub.
    """
    anchors: list[str] = []
    seen: dict[str, int] = {}
    for _level, raw in _HEADING_RE.findall(_FENCE_RE.sub("", markdown)):
        # Strip inline markdown emphasis/code so the slug matches the rendered
        # heading text rather than its source.
        text = re.sub(r"[*_`]", "", raw)
        base = github_slug(text)
        if not base:
            continue
        count = seen.get(base, 0)
        seen[base] = count + 1
        anchors.append(base if count == 0 else f"{base}-{count}")
    return anchors


class TestUserGuideAnchors:
    def test_every_anchor_link_resolves_to_a_heading(self):
        markdown = USER_GUIDE.read_text(encoding="utf-8")
        anchors = set(heading_anchors(markdown))
        assert anchors, "no headings found in USER_GUIDE.md"

        dead = sorted({a for a in _ANCHOR_LINK_RE.findall(markdown) if a not in anchors})
        assert not dead, "USER_GUIDE.md links to anchors with no matching heading: " + ", ".join(f"#{a}" for a in dead)

    def test_table_of_contents_covers_every_top_level_section(self):
        """The TOC must list every ``##`` section, so none goes unreachable."""
        markdown = USER_GUIDE.read_text(encoding="utf-8")
        body = _FENCE_RE.sub("", markdown)
        sections = [
            slug
            for level, raw in _HEADING_RE.findall(body)
            if len(level) == 2
            # "Contents" is the TOC itself; it does not list itself.
            and (slug := github_slug(re.sub(r"[*_`]", "", raw))) != "contents"
        ]
        linked = set(_ANCHOR_LINK_RE.findall(body))
        missing = [s for s in sections if s not in linked]
        assert not missing, "USER_GUIDE.md sections missing from the table of contents: " + ", ".join(
            f"#{s}" for s in missing
        )


class TestGithubSlug:
    def test_spaced_hyphen_yields_three_hyphens(self):
        """The exact trap this file guards: ``a - b`` is *not* ``a-b``."""
        assert github_slug("Autopilot - the guided workflow") == "autopilot---the-guided-workflow"

    def test_colon_and_parens_are_dropped(self):
        assert github_slug("Autopilot: the guided workflow") == "autopilot-the-guided-workflow"
        assert github_slug("Pre-computed embeddings (.npz)") == "pre-computed-embeddings-npz"
