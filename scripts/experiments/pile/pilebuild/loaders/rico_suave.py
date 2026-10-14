"""``rico_suave``: Android app screens by what the screen is FOR (#4356), the genre track.

Built from Rico (Deka et al., UIST 2017; :data:`pile_config.RICO_ROOT`), 66,261
screens each with its view hierarchy. No source labels screen function at scale,
so a label is the agreement of two signals nobody fitted to images:

* **proposal** -- the developer-written Activity name (``LoginActivity``,
  ``SettingsActivity``), matched against :data:`CLASSES`' first regex. The first
  class to match wins, so the specific classes come before the broad ones
  (sign-up before login, cart before search).
* **referee** -- Screen2Words' ~5 crowd summaries of the screen (Wang et al.,
  UIST 2021), written by different people for a different purpose. A summary
  hits a class on its second regex; :data:`AGREE_MIN` hits make the captions
  "say" the class.

A screen is a positive for a :data:`ROSTER` class when both say it. Nothing
else is in the pool: a screen with a generic (``MainActivity``) or app-specific
activity name has no accepted label, and "no keyword matched" is never a
negative. A screen's negatives are therefore the other classes' agreed screens.

**Disagreements are owed a human, not a default.** Where the activity says a
class and the captions do not (or, for cart/browser/splash, the captions say it
and the activity does not), the screen went to a VTSearch Good/Bad review queue,
one per class. This build is **v0**: the agreed set only. A class's reviewed
screens join when its queue is complete, never part-way, because the owner votes
a queue in the app's own ranked order and a part-reviewed queue is the
easy-to-find end of it.

**Multi-label screens are not negatives.** A login screen whose captions also say
"sign up" may well be what someone searching for a sign-up screen wants back, so
an agreed screen is not evaluable for any other class its captions say
(``evaluable_categories``). The ambiguity is concentrated where expected:
login/sign-up, map/search, login/onboarding.

**Splits are by app, through ``split_group``.** Rico records two packages per
screen: the app being explored (``ui_details.csv``) and the app in the
foreground (the activity's package -- the system browser or gallery opened from
another app). Screens sharing either are one group (a union-find component), so
the harness never trains on one of an app's screens and scores its sibling.

**Byte-identical screens are one media.** Rico repeats some screenshots under
different ids (67697 == 67701); the lowest id is kept.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import pile_config as pc
from pilebuild.env import log

#: Of a screen's ~5 summaries, how many must hit a class for the captions to say
#: it. Keyword matching understates true agreement, so the bar is two, not most.
AGREE_MIN = 2

#: ``(class, activity-simple-name regex, caption regex)``. Order matters: the
#: first activity regex to match names the proposal. ``ad``, ``chooser`` and
#: ``camera`` are matched so that their screens are claimed (an ``AdActivity`` is
#: not a candidate for anything else) but are not on :data:`ROSTER`: ads barely
#: reach the captions, and the share sheet and viewfinder are the same system
#: screen from 12 and 19 apps. These rules are the ones the review queues were
#: drawn with (#4356); a change here moves screens between the agreed set and the
#: queues, so it needs new queues too.
CLASSES: tuple[tuple[str, str, str], ...] = (
    (
        "signup",
        r"signup|register|registration|createaccount|join",
        r"sign ?up|regist|create (an |new )?account",  # codespell:ignore regist
    ),
    ("login", r"login|signin|logon|auth(?!or)", r"log ?in|sign ?in|login"),
    ("settings", r"setting|preference|prefs?activity", r"setting|preference"),
    ("browser", r"webview|browser|webactivity|inappweb|customtab", r"browser|web ?page|website|web site"),
    (
        "onboarding",
        r"tutorial|onboard|intro|walkthrough|welcome|guide|tour",
        r"tutorial|welcome|intro|onboard|walkthrough|guide",
    ),
    (
        "ad",
        r"adactivity|mraid|interstitial|admob|adview|^ads?activity",
        r"\bads?\b|advertis",  # codespell:ignore advertis
    ),
    (
        "player",
        r"player|playback|nowplaying|videoactivity|video",
        r"\bplayer\b|\bnow playing\b|\bplaying\b|\bvideos?\b",
    ),
    ("cart", r"cart|checkout|basket|payment", r"\bcart\b|checkout|basket|payment"),
    ("search", r"search", r"search"),
    ("map", r"map(?!per)|location|nearby", r"\bmaps?\b|location"),
    ("gallery", r"gallery|album|photo|imageview|imagegrid", r"gallery|album|photos|images"),
    ("chooser", r"chooseractivity|resolveractivity|shareactivity", r"share|open with|complete action"),
    ("chat", r"chat|conversation|messenger|message|inbox", r"\bchats?\b|\bmessag|\bconversation"),
    ("camera", r"camera|capture|scanner|barcode|qr", r"\bcamera\b|\bscan"),
    ("splash", r"splash|launcher(?!icon)|loading", r"\bsplash\b|\bloading\b"),
)
_ACT_RE = tuple((c, re.compile(a)) for c, a, _ in CLASSES)
_CAP_RE = {c: re.compile(k) for c, _, k in CLASSES}

#: The owner's roster (2026-09-30) and the text each class is searched with --
#: the review queues asked "would you want this back searching for <text>?".
ROSTER: dict[str, str] = {
    "settings": "a settings screen",
    "login": "a login screen",
    "signup": "a sign-up screen",
    "onboarding": "an onboarding or tutorial screen",
    "map": "a map screen",
    "search": "a search screen",
    "chat": "a chat or messaging screen",
    "gallery": "a photo gallery screen",
    "player": "a media player screen",
    "cart": "a shopping cart or checkout screen",
    "browser": "an in-app web browser screen",
    "splash": "a splash or loading screen",
}


class Screen(NamedTuple):
    rico_id: int
    cls: str
    also: tuple[str, ...]  # other roster classes the captions say: not evaluable for those
    group: str  # split group: the app component


def activity_class(activity_name: str) -> str | None:
    """The class the developer's Activity name proposes, or ``None``."""
    simple = activity_name.rsplit("/", 1)[-1].rsplit(".", 1)[-1].lower().split("$", 1)[0]
    for cls, rx in _ACT_RE:
        if rx.search(simple):
            return cls
    return None


def caption_classes(summaries: list[str]) -> set[str]:
    """Every class at least :data:`AGREE_MIN` of the summaries hit."""
    low = [s.lower() for s in summaries]
    return {c for c, rx in _CAP_RE.items() if sum(bool(rx.search(s)) for s in low) >= AGREE_MIN}


def read_captions(path: Path) -> dict[str, list[str]]:
    caps: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            caps[row["screenId"]].append(row["summary"])
    return caps


def read_packages(path: Path) -> dict[str, str]:
    """Rico id -> the package of the app being explored (``ui_details.csv``)."""
    with open(path, newline="") as fh:
        return {row["UI Number"]: row["App Package Name"] for row in csv.DictReader(fh)}


def app_groups(pairs: dict[int, tuple[str, str]]) -> dict[int, str]:
    """Union-find over both packages: ``{rico_id: group}``, named by the group's least package."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs.values():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {rid: find(a) for rid, (a, _) in pairs.items()}


def select(root: Path) -> list[Screen]:
    """The agreed screens, deduplicated by bytes, with their split groups. Reads no pixels but the hashes."""
    caps = read_captions(root / "screen2words" / "screen_summaries.csv")
    traced = read_packages(root / "ui_details.csv")
    combined = root / "combined"
    agreed: dict[int, tuple[str, tuple[str, ...]]] = {}
    pairs: dict[int, tuple[str, str]] = {}
    for rid in sorted(caps, key=int):
        hier = combined / f"{rid}.json"
        if not hier.exists():
            raise SystemExit(f"rico_suave: Screen2Words names screen {rid}, which {combined} lacks")
        act = json.loads(hier.read_text()).get("activity_name") or ""
        cls = activity_class(act)
        if cls not in ROSTER:
            continue
        said = caption_classes(caps[rid])
        if cls not in said:
            continue
        agreed[int(rid)] = (cls, tuple(sorted(c for c in said if c != cls and c in ROSTER)))
        pairs[int(rid)] = (act.split("/", 1)[0], traced[rid])
    groups = app_groups(pairs)

    by_md5: dict[str, list[int]] = defaultdict(list)
    for rid in agreed:
        by_md5[hashlib.md5((combined / f"{rid}.jpg").read_bytes()).hexdigest()].append(rid)
    keep: list[Screen] = []
    for ids in by_md5.values():
        if len({agreed[r][0] for r in ids}) > 1:
            continue  # the same pixels agreed as two classes: no accepted label
        rid = min(ids)
        keep.append(Screen(rid, agreed[rid][0], agreed[rid][1], groups[rid]))
    return sorted(keep)


def screen_media(s: Screen, data: bytes, size: tuple[int, int], embedder_name: str) -> dict:
    return {
        "id": s.rico_id,
        "media_type": "image",
        "embedder": embedder_name,
        "duration": 0,
        "file_size": len(data),
        "md5": "",
        "embeddings": {},
        "media_bytes": data,
        "media_string": None,
        "filename": f"{s.rico_id}.jpg",
        "width": size[0],
        "height": size[1],
        "category": s.cls,
        "categories": [s.cls],
        "evaluable_categories": [c for c in ROSTER if c not in s.also],
        "split_group": s.group,
        "origin": {"importer": "rico_suave", "params": {"embedder": embedder_name, "version": "v0"}},
        "origin_name": f"rico:{s.rico_id}",
    }


def load(dataset: str, medias: dict[int, dict], embedder_name: str) -> None:
    """Populate *medias* with every agreed screen, labelled with its one class."""
    from PIL import Image  # noqa: PLC0415

    root = pc.RICO_ROOT
    screens = select(root)
    for s in screens:
        data = (root / "combined" / f"{s.rico_id}.jpg").read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            size = im.size
        medias[s.rico_id] = screen_media(s, data, size, embedder_name)
    per = Counter(s.cls for s in screens)
    log(
        f"  rico_suave: {len(screens):,} agreed screens, {len({s.group for s in screens}):,} app groups; "
        + ", ".join(f"{c} {per[c]}" for c in ROSTER)
    )


def check(dataset: str) -> str:
    """What a rebuild reads: the hierarchies, ``ui_details.csv``, the summaries and every agreed screen's JPEG."""
    root = pc.RICO_ROOT
    for need in (root / "combined", root / "ui_details.csv", root / "screen2words" / "screen_summaries.csv"):
        if not need.exists():
            raise SystemExit(f"{dataset}: {need} is missing (see {root}/README.md)")
    screens = select(root)
    return f"Rico at {root}: {len(screens):,} agreed screens over {len(ROSTER)} classes"
