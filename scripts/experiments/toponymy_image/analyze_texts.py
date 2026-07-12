"""Text-variant diagnostics: distinctiveness + vocabulary-mismatch numbers.

Answers the questions the cluster metrics can't:

- **Concentration**: how much of a dataset's text output is the same few
  terms? (A subset where every image tags as "dog, dog collar, …" cannot
  yield distinguishing signs.) Reported as top-1-term share and the number
  of distinct terms covering 80% of items.
- **Vocabulary mismatch**: on documents/screenshots, what fraction of
  zero-shot tags are photo-object terms that would be distractor signs?
  Approximated by SigLIP text-space similarity of each tag to domain probes
  ("a photograph of an object/animal" vs "a document/screenshot").
- **Term diversity within a category** (fine-grain potential): for each
  ground-truth category, the number of distinct top-1 terms — a captioner
  that says "a dog" for every breed scores 1; one that names breeds scores
  high.

Writes ``RESULTS/<ds>/textstats_<variant>.json`` and prints a table.

Usage::

    python analyze_texts.py caltech101 stanford_dogs enrico rvl_cdip mixed
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

import common

common.setup_env()

import numpy as np  # noqa: E402


def first_term(text: str) -> str:
    """The lead term of a text: first comma field (tags) or first 4 words (captions)."""
    t = str(text).strip().lower()
    if "," in t and len(t.split(",")[0]) < 40:
        return t.split(",")[0].strip()
    words = re.findall(r"[a-z0-9']+", t)
    return " ".join(words[:4])


def all_terms(text: str) -> list[str]:
    t = str(text).strip().lower()
    if "," in t and len(t.split(",")[0]) < 40:
        return [x.strip() for x in t.split(",") if x.strip()]
    return [" ".join(re.findall(r"[a-z0-9']+", t))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--embedder", default="siglip")
    args = ap.parse_args()

    from run_toponymy import SiglipTextEncoder

    enc = SiglipTextEncoder(args.embedder)
    probes = [
        "a photograph of an object, animal, or scene",
        "a scanned document, form, or page of text",
        "a screenshot of an app or website",
    ]
    pv = enc.encode(probes)
    pv = pv / (np.linalg.norm(pv, axis=1, keepdims=True) + 1e-9)

    rows = []
    for ds in args.datasets:
        out = common.ds_dir(ds)
        if not (out / "meta.json").exists():
            continue
        meta = common.load_json(out / "meta.json")
        cats = [m["category"] for m in meta]
        for tf in sorted(out.glob("texts_*.json")):
            if tf.name.endswith("_info.json") or "_probe" in tf.name:
                continue
            variant = tf.stem.replace("texts_", "")
            texts = common.load_json(tf)
            n = min(len(texts), len(meta))
            leads = [first_term(t) for t in texts[:n]]
            lead_counts = Counter(leads)
            top1_share = lead_counts.most_common(1)[0][1] / n if n else 0
            # distinct lead terms covering 80% of items
            need, cum = 0, 0
            for _, c in lead_counts.most_common():
                cum += c
                need += 1
                if cum >= 0.8 * n:
                    break

            # per-category lead-term diversity (fine-grain potential)
            by_cat = defaultdict(set)
            for c, ld in zip(cats[:n], leads):
                by_cat[c].add(ld)
            per_cat_div = float(np.mean([len(v) for v in by_cat.values()]))

            # domain alignment of the tag/caption vocabulary
            uniq_terms = list({t for txt in texts[:n] for t in all_terms(txt)})[:4000]
            tv = enc.encode(uniq_terms) if uniq_terms else np.zeros((0, 3))
            if len(uniq_terms):
                tv = tv / (np.linalg.norm(tv, axis=1, keepdims=True) + 1e-9)
                nearest = np.argmax(tv @ pv.T, axis=1)
                photo_frac = float((nearest == 0).mean())
                doc_frac = float((nearest == 1).mean())
                screen_frac = float((nearest == 2).mean())
            else:
                photo_frac = doc_frac = screen_frac = 0.0

            stat = {
                "dataset": ds,
                "variant": variant,
                "n": n,
                "top1_term_share": round(top1_share, 4),
                "n_lead_terms_80pct": need,
                "distinct_lead_terms": len(lead_counts),
                "per_category_lead_diversity": round(per_cat_div, 2),
                "term_domain_photo_frac": round(photo_frac, 4),
                "term_domain_document_frac": round(doc_frac, 4),
                "term_domain_screenshot_frac": round(screen_frac, 4),
                "top_leads": lead_counts.most_common(8),
            }
            common.save_json(out / f"textstats_{variant}.json", stat)
            rows.append(stat)

    print(
        f"\n{'dataset':14s} {'variant':22s} {'top1%':6s} {'n80':5s} {'perCat':7s} {'photo':6s} {'doc':5s} {'scrn':5s}"
    )
    for r in rows:
        print(
            f"{r['dataset']:14s} {r['variant']:22s} {r['top1_term_share'] * 100:5.1f}  "
            f"{r['n_lead_terms_80pct']:5d} {r['per_category_lead_diversity']:7.2f} "
            f"{r['term_domain_photo_frac'] * 100:5.1f}  {r['term_domain_document_frac'] * 100:4.1f}  "
            f"{r['term_domain_screenshot_frac'] * 100:4.1f}"
        )


if __name__ == "__main__":
    main()
