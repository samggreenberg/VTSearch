"""Completeness proposers that share nothing with SIFT: OCR text and template NCC.

The completeness pass (#3927) found missing members with SIFT, so the members
it still lacks are exactly the ones SIFT cannot see -- faint, smeared, rotated,
or too small to hold keypoints -- and any SIFT-vs-X comparison on the result is
tilted toward SIFT.  These two proposers fail in different places:

* ``ocr_text`` reads every anchor page with EasyOCR and looks for the words a
  class's mark carries (``class_text.json``: "DY.Secretary 1111222",
  "OUTWARD").  It shares no pixel features with any matcher, so a smeared stamp
  whose letters survive is still found.  It only covers classes with legible
  text, and a class whose words also occur in ordinary letters floods with
  false positives -- matches must be *spatially clustered* (every class token
  near one anchor token), and single-token classes are flagged.
* ``template_ncc`` slides the query crop over each page at the class's own box
  scales and a few rotations and scores zero-mean normalised cross-correlation.
  No keypoints and no learning: it is strongest exactly where SIFT starves (a
  211-keypoint crop) and weakest on warped or heavily re-inked copies.

Both write the shared interface the multi-method slate reads::

    /expscratch/sgreenberg/fullmarks/completeness2/proposals-<method>.json
    {"method": str, "classes": {class_id: [[page_id, score, [x, y, w, h] | null], ...]}}

ranked best first.  The list keeps members too (``K = n_instances + TOP``) so
recall of known members -- how well a method finds what is already labelled --
can be read off the same file; the slate drops members itself.

Subcommands (GRID; heavy outputs on /expscratch)::

    proposers_ocr_ncc.py ocr-pages --shard I --shards N     # GPU, caches raw OCR
    proposers_ocr_ncc.py ocr-queries                         # OCR the query crops
    proposers_ocr_ncc.py score-ocr                           # -> proposals-ocr_text.json
    proposers_ocr_ncc.py ncc --shard I --shards N            # GPU, per-shard scores
    proposers_ocr_ncc.py merge-ncc                           # -> proposals-template_ncc.json
    proposers_ocr_ncc.py recall                              # known-member recall table

EasyOCR is not in the shared venv: it lives in a private ``--no-deps --target``
directory (``OCR_PYDEPS``), put on ``sys.path`` only by the OCR subcommands.
"""

from __future__ import annotations

import functools
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

HERE = Path(__file__).resolve().parent
CLASS_TEXT = HERE / "class_text.json"
OUT = Path("/expscratch/sgreenberg/fullmarks/completeness2")
OCR_CACHE = Path("/expscratch/sgreenberg/fullmarks/ocr/easyocr")
OCR_PYDEPS = Path("/expscratch/sgreenberg/ocr-pydeps")
OCR_MODELS = Path("/expscratch/sgreenberg/easyocr-models")
ANCHOR_SOURCES = ("spods", "staver", "tobacco800")

#: Proposals kept per class beyond its own members.
TOP = 100
#: A page token counts as a class token at this normalised similarity.
MIN_TOKEN_SIM = 0.7
#: Class tokens must sit within this many median member-box long sides of the anchor token.
RADIUS_BOXES = 1.5
#: NCC canvas: pages are downscaled so the long side is this many pixels.
CANVAS = 1024
#: Template rotations, degrees counter-clockwise.
ROTATIONS = (0.0, 15.0, -15.0, 90.0)
#: Windows flatter than this (std of intensity in [0, 1]) never score: blank paper.
MIN_WINDOW_STD = 0.04
#: Smallest template side worth correlating, canvas pixels.
MIN_TEMPLATE_SIDE = 12


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Casefold, strip accents, keep only letters and digits."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def tokens(text: str) -> list[str]:
    """Normalised words of *text*, split at anything not a letter or digit.

    Splitting at punctuation too means "E-Mail" and "E Mail" agree however
    the OCR spaced them.  Single characters are dropped: they match anything.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return [tok for tok in re.split(r"[^a-z0-9]+", text) if len(tok) >= 2]


@functools.lru_cache(maxsize=1 << 20)
def similarity(a: str, b: str) -> float:
    """1 - Levenshtein(a, b) / max(len): 1.0 identical, 0.0 nothing shared."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


@dataclass
class Word:
    text: str  # normalised token
    box: tuple[int, int, int, int]  # x, y, w, h in page pixels

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2


def words_from_ocr(items: Iterable[Sequence[Any]]) -> list[Word]:
    """EasyOCR ``[quad, text, conf]`` items -> one :class:`Word` per token, boxed by its item.

    The detector often cuts a stamped word in two ("Secre" | "tary"), so when two
    consecutive items sit side by side on one line the last token of the first
    joined to the first token of the second is added as well, boxed by both.
    """
    out: list[Word] = []
    prev: Optional[tuple[tuple[int, int, int, int], list[str]]] = None
    for quad, text, _conf in items:
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        box = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        toks = tokens(text)
        out.extend(Word(tok, box) for tok in toks)
        if prev is not None and prev[1] and toks and _side_by_side(prev[0], box):
            out.append(Word(prev[1][-1] + toks[0], union([prev[0], box])))
        prev = (box, toks)
    return out


def _side_by_side(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """*b* starts just right of *a* (gap under one line height) and overlaps it vertically."""
    height = max(a[3], b[3], 1)
    gap = b[0] - (a[0] + a[2])
    overlap = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return -height <= gap <= height and overlap >= 0.5 * min(a[3], b[3])


def union(boxes: Sequence[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def score_page_text(
    class_tokens: Sequence[str],
    words: Sequence[Word],
    radius: float,
    *,
    min_sim: float = MIN_TOKEN_SIM,
) -> tuple[float, Optional[tuple[int, int, int, int]]]:
    """Best spatially-clustered match of *class_tokens* among a page's *words*.

    Every page word that matches some class token is tried as an anchor; each
    class token then takes its best match within *radius* page pixels of that
    anchor.  The score is the mean similarity over class tokens (an unmatched
    token contributes 0), so "PHILIP" alone in body text scores half of
    "PHILIP MORRIS" together.  The box is the union of the matched words.
    """
    if not class_tokens or not words:
        return 0.0, None
    # Only words that pass min_sim for some token can take part; usually a handful.
    hits = [[(i, s) for i, w in enumerate(words) if (s := similarity(ct, w.text)) >= min_sim] for ct in class_tokens]
    anchors = sorted({i for per in hits for i, _ in per})
    best, best_box = 0.0, None
    for a in anchors:
        ax, ay = words[a].centre
        total, boxes = 0.0, []
        for per in hits:
            top, top_i = 0.0, -1
            for i, s in per:
                if s <= top:
                    continue
                cx, cy = words[i].centre
                if math.hypot(cx - ax, cy - ay) <= radius:
                    top, top_i = s, i
            if top_i >= 0:
                total += top
                boxes.append(words[top_i].box)
        score = total / len(class_tokens)
        if score > best:
            best, best_box = score, union(boxes)
    return best, best_box


# ---------------------------------------------------------------------------
# Template NCC (pure geometry; the correlation itself runs on the GPU)
# ---------------------------------------------------------------------------


def template_scales(fractions: Sequence[float], *, pad: float = 0.15, step: float = 1.15, most: int = 8) -> list[float]:
    """Geometric ladder of template long sides, as fractions of the page long side.

    *fractions* are the class members' box long sides over their page long
    sides; the ladder spans them with *pad* either way, one rung per *step*.
    """
    lo, hi = min(fractions) * (1 - pad), max(fractions) * (1 + pad)
    n = max(2, min(most, math.ceil(math.log(hi / lo) / math.log(step)) + 1))
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


def peak_to_box(x: int, y: int, w: int, h: int, factor: float) -> list[int]:
    """A template placed at canvas (x, y) with size (w, h) -> page-pixel ``[x, y, w, h]``.

    *factor* is canvas pixels per page pixel.
    """
    return [round(x / factor), round(y / factor), round(w / factor), round(h / factor)]


# ---------------------------------------------------------------------------
# Shared corpus plumbing
# ---------------------------------------------------------------------------


def _corpus(corpus: Path) -> tuple[dict[str, Any], list[Any]]:
    sys.path.insert(0, str(HERE))
    from sources._common import read_manifest  # noqa: PLC0415

    classes = json.loads((corpus / "classes.json").read_text(encoding="utf-8"))
    roster = {k: v for k, v in classes.items() if v.get("on_roster")}
    pages = [p for p in read_manifest(corpus / "corpus.jsonl") if p.source in ANCHOR_SOURCES]
    return roster, pages


def member_boxes(
    class_id: str, pages_by_id: dict[str, Any], meta: dict[str, Any]
) -> list[tuple[Any, tuple[int, int, int, int]]]:
    out = []
    for pid in meta.get("page_ids", []):
        page = pages_by_id.get(pid)
        if page is None:
            continue
        for mark in page.marks:
            if mark.class_id == class_id:
                out.append((page, tuple(mark.box)))
                break
    return out


def write_proposals(method: str, ranked: dict[str, list[list[Any]]], out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"proposals-{method}.json"
    path.write_text(json.dumps({"method": method, "classes": ranked}) + "\n", encoding="utf-8")
    return path


def _shard(items: list[Any], shard: int, shards: int) -> list[Any]:
    return [x for i, x in enumerate(items) if i % shards == shard]


def _cache_path(page_id: str) -> Path:
    return OCR_CACHE / (page_id.replace("/", "__") + ".json")


def _reader() -> Any:
    sys.path.insert(0, str(OCR_PYDEPS))
    import easyocr  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    import torch  # noqa: PLC0415

    return easyocr.Reader(
        ["en", "de"],
        gpu=torch.cuda.is_available(),
        model_storage_directory=str(OCR_MODELS),
        download_enabled=False,
        verbose=False,
    )


def _ocr(reader: Any, image: Any) -> list[list[Any]]:
    import numpy as np  # noqa: PLC0415

    res = reader.readtext(np.asarray(image.convert("RGB")), paragraph=False)
    return [[[[float(v) for v in pt] for pt in quad], text, float(conf)] for quad, text, conf in res]


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_ocr_pages(args: Any) -> int:
    import time  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    _, pages = _corpus(args.corpus)
    todo = [
        p
        for p in _shard(sorted(pages, key=lambda p: p.page_id), args.shard, args.shards)
        if not _cache_path(p.page_id).exists()
    ]
    print(f"shard {args.shard}/{args.shards}: {len(todo)} page(s) to read", flush=True)
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    reader, t0 = _reader(), time.time()
    for i, page in enumerate(todo):
        with Image.open(page.path) as im:
            items = _ocr(reader, im)
        tmp = _cache_path(page.page_id).with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"page_id": page.page_id, "width": page.width, "height": page.height, "items": items}),
            encoding="utf-8",
        )
        tmp.replace(_cache_path(page.page_id))
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)
    return 0


def cmd_ocr_queries(args: Any) -> int:
    from PIL import Image  # noqa: PLC0415

    roster, _ = _corpus(args.corpus)
    reader = _reader()
    out = {}
    for class_id, meta in sorted(roster.items()):
        crops = meta.get("query_crops") or [meta["query_crop"]]
        texts = []
        for path in crops:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if max(im.size) < 600:  # small crops read better upscaled
                    k = 600 / max(im.size)
                    im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
                texts.append(" | ".join(f"{t} ({c:.2f})" for _, t, c in _ocr(reader, im)))
        out[class_id] = texts
        print(f"{class_id}: {texts}", flush=True)
    (OUT / "query_ocr.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_score_ocr(args: Any) -> int:
    import statistics  # noqa: PLC0415

    roster, pages = _corpus(args.corpus)
    by_id = {p.page_id: p for p in pages}
    spec = json.loads(CLASS_TEXT.read_text(encoding="utf-8"))["classes"]
    page_words = {}
    for page in pages:
        cache = _cache_path(page.page_id)
        if cache.exists():
            page_words[page.page_id] = words_from_ocr(json.loads(cache.read_text(encoding="utf-8"))["items"])
    print(f"{len(page_words)}/{len(pages)} anchor pages have OCR", flush=True)
    ranked = {}
    for class_id, meta in sorted(roster.items()):
        texts = (spec.get(class_id) or {}).get("text")
        if not texts:
            continue
        variants = [tokens(t) for t in ([texts] if isinstance(texts, str) else texts)]
        sides = [max(b[2], b[3]) for _, b in member_boxes(class_id, by_id, meta)]
        radius = RADIUS_BOXES * statistics.median(sides)
        scored = []
        for pid, words in page_words.items():
            s, box = max((score_page_text(v, words, radius) for v in variants), key=lambda r: r[0])
            if s > 0:
                scored.append([pid, round(s, 4), list(box) if box else None])
        scored.sort(key=lambda r: (-r[1], r[0]))
        ranked[class_id] = scored[: TOP + len(meta.get("page_ids", []))]
        print(f"  {class_id}: {variants} radius {radius:.0f}px, {len(scored)} page(s) scored", flush=True)
    print(write_proposals("ocr_text", ranked))
    return 0


def _gray_canvas(image: Any, canvas: int = CANVAS) -> tuple[Any, float]:
    """Page -> float32 [0, 1] grayscale array with long side *canvas*, and canvas px per page px."""
    import numpy as np  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    g = image.convert("L")
    factor = canvas / max(g.size)
    g = g.resize((max(1, round(g.width * factor)), max(1, round(g.height * factor))), Image.BILINEAR)
    return np.asarray(g, dtype=np.float32) / 255.0, factor


def _templates(roster: dict[str, Any], by_id: dict[str, Any]) -> list[dict[str, Any]]:
    """Every (class, scale, rotation) template as a canvas-scale float32 array."""
    import numpy as np  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    out = []
    for class_id, meta in sorted(roster.items()):
        fractions = [max(b[2], b[3]) / max(p.width, p.height) for p, b in member_boxes(class_id, by_id, meta)]
        with Image.open(meta["query_crop"]) as im:
            crop = im.convert("L")
        for frac in template_scales(fractions):
            long_side = frac * CANVAS
            k = long_side / max(crop.size)
            w, h = max(1, round(crop.width * k)), max(1, round(crop.height * k))
            if min(w, h) < MIN_TEMPLATE_SIDE or max(w, h) >= CANVAS * 0.9:
                continue
            base = crop.resize((w, h), Image.LANCZOS)
            for rot in ROTATIONS:
                t = base.rotate(rot, resample=Image.BILINEAR, expand=True, fillcolor=255) if rot else base
                out.append(
                    {"class_id": class_id, "frac": frac, "rot": rot, "array": np.asarray(t, dtype=np.float32) / 255.0}
                )
    return out


def cmd_ncc(args: Any) -> int:
    import time  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    roster, pages = _corpus(args.corpus)
    by_id = {p.page_id: p for p in pages}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temps = _templates(roster, by_id)
    print(f"{len(temps)} templates over {len(roster)} classes on {dev}", flush=True)
    # Zero-mean templates padded into the canvas, conj-FFT'd once; norms for the denominator.
    tf, tnorm, tsize = [], [], []
    for t in temps:
        a = t["array"] - t["array"].mean()
        pad = np.zeros((CANVAS, CANVAS), dtype=np.float32)
        pad[: a.shape[0], : a.shape[1]] = a
        tf.append(torch.conj(torch.fft.rfft2(torch.from_numpy(pad).to(dev))))
        tnorm.append(float(np.sqrt((a * a).sum())) + 1e-6)
        tsize.append(a.shape)
    class_idx: dict[str, list[int]] = {}
    for i, t in enumerate(temps):
        class_idx.setdefault(t["class_id"], []).append(i)
    mine = _shard(sorted(pages, key=lambda p: p.page_id), args.shard, args.shards)
    rows, t0 = [], time.time()
    for n, page in enumerate(mine):
        with Image.open(page.path) as im:
            arr, factor = _gray_canvas(im)
        hp, wp = arr.shape
        P = torch.zeros((CANVAS, CANVAS), device=dev)
        P[:hp, :wp] = torch.from_numpy(arr).to(dev)
        fP = torch.fft.rfft2(P)
        # Integral images (one row/column of zeros in front) for window sums of P and P^2.
        I1 = torch.nn.functional.pad(P.cumsum(0).cumsum(1), (1, 0, 1, 0))
        I2 = torch.nn.functional.pad((P * P).cumsum(0).cumsum(1), (1, 0, 1, 0))
        best: dict[str, tuple[float, list[int]]] = {}
        for i, t in enumerate(temps):
            h, w = tsize[i]
            vh, vw = hp - h + 1, wp - w + 1
            if vh <= 0 or vw <= 0:
                continue
            num = torch.fft.irfft2(fP * tf[i], s=(CANVAS, CANVAS))[:vh, :vw]
            s1 = I1[h : h + vh, w : w + vw] - I1[:vh, w : w + vw] - I1[h : h + vh, :vw] + I1[:vh, :vw]
            s2 = I2[h : h + vh, w : w + vw] - I2[:vh, w : w + vw] - I2[h : h + vh, :vw] + I2[:vh, :vw]
            var = (s2 - s1 * s1 / (h * w)).clamp_min(0)
            ncc = num / (torch.sqrt(var) * tnorm[i] + 1e-6)
            ncc = torch.where(var / (h * w) >= MIN_WINDOW_STD**2, ncc, torch.zeros_like(ncc))
            v, flat = torch.max(ncc.reshape(-1), 0)
            v = float(v)
            cid = t["class_id"]
            if cid not in best or v > best[cid][0]:
                y, x = divmod(int(flat), vw)
                best[cid] = (v, peak_to_box(x, y, w, h, factor) + [t["rot"]])
        for cid, (v, box) in best.items():
            rows.append([cid, page.page_id, round(v, 4), box[:4], box[4]])
        if n % 100 == 0:
            print(f"  {n}/{len(mine)} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"ncc-shard{args.shard:02d}of{args.shards:02d}.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print("done", flush=True)
    return 0


def cmd_merge_ncc(args: Any) -> int:
    roster, _ = _corpus(args.corpus)
    per: dict[str, list[list[Any]]] = {}
    shards = sorted(OUT.glob("ncc-shard*.json"))
    for path in shards:
        for cid, pid, v, box, _rot in json.loads(path.read_text(encoding="utf-8")):
            per.setdefault(cid, []).append([pid, v, box])
    ranked = {}
    for cid, rows in sorted(per.items()):
        rows.sort(key=lambda r: (-r[1], r[0]))
        ranked[cid] = rows[: TOP + len(roster[cid].get("page_ids", []))]
    print(f"{len(shards)} shard(s), {sum(len(r) for r in per.values())} rows")
    print(write_proposals("template_ncc", ranked))
    return 0


def recall_at(ranked: Sequence[Sequence[Any]], members: set[str], k: int) -> float:
    """Share of *members* among the first *k* ranked page ids."""
    if not members:
        return 0.0
    return len({r[0] for r in ranked[:k]} & members) / len(members)


def cmd_recall(args: Any) -> int:
    roster, _ = _corpus(args.corpus)
    for method in args.methods:
        path = OUT / f"proposals-{method}.json"
        if not path.exists():
            print(f"{method}: missing")
            continue
        classes = json.loads(path.read_text(encoding="utf-8"))["classes"]
        print(f"\n{method}  (recall of known members in the top n_members / top n_members+{TOP})")
        for cid, meta in sorted(roster.items()):
            members = set(meta.get("page_ids", []))
            ranked = classes.get(cid)
            if ranked is None:
                print(f"  {cid:40s} n={len(members):3d}  (no proposals)")
                continue
            print(
                f"  {cid:40s} n={len(members):3d}  R@n {recall_at(ranked, members, len(members)):.2f}  R@n+{TOP} {recall_at(ranked, members, len(members) + TOP):.2f}"
            )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("/expscratch/sgreenberg/fullmarks/corpus"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ocr-pages", "ncc"):
        p = sub.add_parser(name)
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--shards", type=int, default=1)
    sub.add_parser("ocr-queries")
    sub.add_parser("score-ocr")
    sub.add_parser("merge-ncc")
    p = sub.add_parser("recall")
    p.add_argument("--methods", nargs="+", default=["ocr_text", "template_ncc"])
    args = ap.parse_args(argv)
    return {
        "ocr-pages": cmd_ocr_pages,
        "ocr-queries": cmd_ocr_queries,
        "score-ocr": cmd_score_ocr,
        "ncc": cmd_ncc,
        "merge-ncc": cmd_merge_ncc,
        "recall": cmd_recall,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
