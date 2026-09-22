"""Sample the unchecked half of a class's pool and ask a human what is in it.

97.6% of every page DocMarks scores is a ``presumed_negative`` -- a page from a
contamination-safe source that nobody looked at.  ``CONTAMINATES`` removes the
*systematic* risk by construction (Tobacco800 is never scored against UCSF's
Tobacco industry, same IIT-CDIP archive), but the residual has never been
measured, and an unlabelled positive does not make a benchmark noisy: it makes a
correct retrieval score as a false positive, so the metric punishes a model for
being right.

**This estimates a rate; it does not clear anything.**  Nothing here may mint a
negative -- a detector's misses sit exactly on the faint, small marks an
advanced method should win on, so clearing a pool with one puts the benchmark's
ceiling at the annotation tool's recall.  A sampled page a human finds the mark
on goes to ``excluded_page_ids`` (seen to carry the mark, not made an instance),
which removes it from every pool.

**Every queue carries planted known positives.** A reviewer answering fifty
pages in a row, all of them negatives, stops looking and starts pressing -- and
an inattentive sweep of "no" is indistinguishable from an attentive one in the
result. So each queue hides one instance the corpus already knows carries the
mark, and a third of them hide two. Missing one does not invalidate the pass by
itself, but it is the only evidence available that the zeros were looked at, and
the control accuracy is reported beside the rate.

Controls are drawn from the class's own adjudicated instances, never from a page
used as a reference on the sheet, and they are shuffled in with a fixed seed.
**Nothing in the filename or the footer says which arm a question came from** --
not the controls, and not the uniform/ranked split either, because knowing a page
was ranked highly by SigLIP is a reason to look harder at it.

Two samples per class, reported separately rather than pooled:

* ``uniform``  -- a detector-free random draw, so the overall bound owes nothing
  to any ranker.  With zero hits in *n*, the 95% upper bound is 3/n.
* ``ranked``   -- the top of a SigLIP ranking over the same set.  This is where
  AP damage lives: an unlabelled positive at rank 40,000 cannot move a number,
  one at rank 5 moves it a lot.  Biased by construction and never pooled with
  the uniform draw.

Risk is concentrated by source: a SPODS mark is a logo invented for the dataset
and a StaVer mark is on a German invoice, so neither can plausibly appear on a
real UCSF scan.  The Tobacco800 classes are the live case -- same era, same
American corporate letterhead, and the Kraft/Philip Morris path is already
documented in ``CONTAMINATES``.  The spot checks exist to confirm the near-zero
prior on the other two, not to spend review on it.

    python contamination_sample.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import docmarks_config as cfg  # noqa: E402
import roster  # noqa: E402
from binary_review import Question, class_refs, emit  # noqa: E402
from sources._common import read_manifest  # noqa: E402

#: Tobacco800 carries the live risk; the other two are spot checks on a prior.
TOBACCO_CLASSES = (
    "tobacco800/logo_ajj10e00_1",
    "tobacco800/logo_aah97e00-page02_1_0",
    "tobacco800/logo_ald41a00-ernest_1",
    "tobacco800/logo_cgr96c00_1",
)
SPOT_CLASSES = ("spods/logo_00003_0", "staver/stamp_stampds-00213_1")
N_UNIFORM = 50
N_RANKED = 50
N_SPOT = 20
#: Planted known positives per queue, and the chance of a second one.  Kept
#: unpredictable on purpose: a reviewer who knows there is exactly one can relax
#: after finding it.
N_CONTROL = 1
P_SECOND_CONTROL = 1 / 3
#: Fixed so the draw is reproducible and cannot be re-rolled to a nicer answer.
SEED = 20260920


def pools_for(classes: dict[str, Any], pages_by_source, industry_of) -> dict[str, list[str]]:
    """Each roster class -> the UCSF pages in its headline pool nobody checked.

    Anchor pages are ``presumed`` too, but they are exhaustively decomposed into
    marks and clustered, so their absence of a mark is near-verified.  The
    untouched half is UCSF, and that is what this samples.
    """
    out: dict[str, list[str]] = {}
    for cid, meta in sorted(classes.items()):
        if not meta.get("on_roster"):
            continue
        own = roster.eligible_pages(
            meta, pages_by_source, verified_negative_sources=[meta.get("source")], industry_of=industry_of
        )
        out[cid] = sorted(p for p in own["presumed_negative"] if p.startswith("ucsf/"))
    return out


def siglip_rank(class_id: str, candidates: Sequence[str], classes, tier: str) -> list[str]:
    """*candidates* ordered by SigLIP cosine to the class's query crop, best first.

    Mirrors ``eval_retrieval``'s scoring exactly -- same cell, same reader, same
    embedder -- so the ranked arm is the ordering a study would actually see.
    """
    import numpy as np  # noqa: PLC0415

    import embed_corpus as E  # noqa: PLC0415
    from eval_retrieval import embed_query, read_vectors  # noqa: PLC0415
    from vtscore.media import get_embedder  # noqa: PLC0415

    ids, matrix = read_vectors(E.cell_path(tier, "siglip"), "siglip")
    emb = get_embedder("siglip")
    emb.load_models()
    vec, _feats = embed_query(emb, Path(classes[class_id]["query_crop"]))
    if vec is None:
        raise SystemExit(f"{class_id}: no vector for its query crop")
    sims = matrix @ np.asarray(vec, dtype=np.float32)
    score = dict(zip(ids, sims.tolist()))
    return sorted((p for p in candidates if p in score), key=lambda p: -score[p])


def controls_for(class_id: str, classes, pages, refs, rng, n: int, prefer_source: str = "ucsf") -> list[str]:
    """*n* pages the corpus says carry this mark, excluding anything on the sheet.

    Drawn from the middle of the size distribution rather than the largest
    instances: the biggest are the ones already shown as references, and a
    control that is twice the size of anything else on the page is a giveaway
    rather than a check.
    """
    meta = classes[class_id]
    shown = {r.page_id for r in refs if r.page_id} | {meta.get("query_page_id")}
    sized = []
    for pid in meta.get("page_ids", []):
        if pid in shown or pid not in pages:
            continue
        for m in pages[pid].marks:
            if m.class_id == class_id:
                sized.append((m.box[2] * m.box[3], pid))
                break
    if not sized:
        return []
    # A control has to look like the rest of the queue.  Every candidate here is
    # a UCSF page, so a control on a SPODS page is spotted from its style alone
    # and the check degrades into "find the odd one out".  Prefer an instance on
    # the same source; fall back only when the class has none there.
    same = [x for x in sized if x[1].startswith(prefer_source + "/")]
    if len(same) >= n:
        sized = same
    sized.sort()
    lo, hi = len(sized) // 4, max(len(sized) // 4 + 1, (3 * len(sized)) // 4)
    middle = [pid for _a, pid in sized[lo:hi]] or [pid for _a, pid in sized]
    return rng.sample(middle, min(n, len(middle)))


def questions_for(
    class_id: str,
    items: Sequence[tuple[str, str]],
    classes: dict[str, Any],
    pages: dict[str, Any],
    refs,
) -> list[Question]:
    """One Question per (page_id, arm), numbered in the order given.

    The arm lives in ``key`` and nowhere a reviewer can see it: not the
    filename, not the footer.
    """
    slug = class_id.replace("/", "_").replace("-", "_")
    out = []
    for i, (pid, arm) in enumerate(items):
        page = pages.get(pid)
        if page is None:
            continue
        out.append(
            Question(
                filename=f"contam__{slug}__{i:03d}.jpg",
                task="contamination",
                question="Is the mark on the left anywhere on this page?",
                refs=refs,
                page_id=pid,
                # The whole page: there is no proposed box, the question is
                # whether the mark is present at all.
                box=[0, 0, page.width, page.height],
                outline=False,
                key={"class_id": class_id, "page_id": pid, "arm": arm},
                # Portrait so the page is legible, but greyscale at q80 and a
                # notch smaller than the first cut: the reviewer pays for every
                # byte of this sheet, twice (centre panel, then its thumbnail in
                # the label list), and 1700x1500 RGB q90 cost six seconds a page.
                canvas=[1530, 1350],
                greyscale=True,
                quality=80,
                # Border trim is deliberately OFF here, measured 2026-09-20:
                # 73% of these pages trim at all, and only to 1.14x linear
                # magnification (1.21x best) -- UCSF scans are already tightly
                # cropped.  It also costs ~4% MORE bytes, because white paper is
                # nearly free in JPEG and the content scaled up to fill the panel
                # is not.  Worth revisiting for a source with real letterboxing.
                trim_border=False,
                anonymous=True,
                detail="",
            )
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=cfg.OUT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tier", default="m")
    ap.add_argument("--no-ranked", action="store_true", help="uniform arm only")
    args = ap.parse_args(argv)

    order = list(cfg.TIER_ORDER)
    want = set(order[: order.index(args.tier) + 1])
    classes = json.loads((args.corpus / "classes.json").read_text(encoding="utf-8"))

    pages: dict[str, Any] = {}
    pages_by_source: dict[str, list[str]] = collections.defaultdict(list)
    industry_of: dict[str, Optional[str]] = {}
    for page in read_manifest(args.corpus / "corpus.jsonl"):
        meta = page.meta or {}
        if meta.get("tier") not in want:
            continue
        pages[page.page_id] = page
        pages_by_source[page.source].append(page.page_id)
        industry_of[page.page_id] = meta.get("industry")

    unchecked = pools_for(classes, pages_by_source, industry_of)
    rng = random.Random(SEED)
    plan = [(c, N_UNIFORM, N_RANKED) for c in TOBACCO_CLASSES] + [(c, N_SPOT, 0) for c in SPOT_CLASSES]

    written = []
    for class_id, n_uniform, n_ranked in plan:
        frame = unchecked.get(class_id) or []
        if not frame:
            print(f"  {class_id}: no unchecked UCSF pages in pool, skipped")
            continue
        refs = class_refs(class_id, classes, pages)
        uniform = rng.sample(frame, min(n_uniform, len(frame)))
        items = [(p, "uniform") for p in uniform]
        if n_ranked and not args.no_ranked:
            taken = set(uniform)
            ranked = [p for p in siglip_rank(class_id, frame, classes, args.tier) if p not in taken][:n_ranked]
            items += [(p, "ranked") for p in ranked]
        n_ctrl = N_CONTROL + (1 if rng.random() < P_SECOND_CONTROL else 0)
        controls = controls_for(class_id, classes, pages, refs, rng, n_ctrl)
        off = [p for p in controls if not p.startswith("ucsf/")]
        if off:
            print(f"     note: {class_id} has no UCSF instance to plant; using {off}")
        if len(controls) < N_CONTROL:
            print(f"  !! {class_id}: only {len(controls)} control(s) available")
        items += [(p, "control") for p in controls]
        # Shuffled before numbering, so a control is not the first or last thing
        # answered and the arms are interleaved.
        rng.shuffle(items)
        qs = questions_for(class_id, items, classes, pages, refs)
        name = f"docmarks {class_id.split('/')[-1]} -- is this mark on the page?"
        qdir = args.out / (class_id.replace("/", "_").replace("-", "_") + "__contam")
        emit(qs, qdir, name, pages=pages, corpus=args.corpus)
        written.append((name, qdir, len(qs), len(frame), len(controls)))
        print(f"  {class_id}: {len(qs)} question(s) ({len(controls)} planted) from a frame of {len(frame)} -> {qdir}")

    total = sum(w[2] for w in written)
    ctrl = sum(w[4] for w in written)
    print(f"\n{len(written)} queue(s), {total} question(s) including {ctrl} planted; seed {SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
