"""Find non-Tobacco UCSF pages whose metadata names a Tobacco800 roster company (#3914).

    python meta_scan.py measurements/meta_hits.json

Reads ``corpus.jsonl`` from ``VTS_FULLMARKS_OUT`` (tiers s and m only) and writes
the matching page ids per company.  Independent of any ranker: it reaches the
pages that say who they are from, whether or not SigLIP ranks them.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

COMPANIES = {
    "philip morris": r"philip\s*morris|\bkraft|general foods",
    "rjr": r"\brjr\b|reynolds|nabisco",
    "lorillard": r"lorillard|loews",
    "b&w": r"brown\s*&\s*williamson|\bb\s*&\s*w\b",
    "american tobacco": r"american tobacco|american brands",
}
TIERS = {"s", "m"}


def main(out: Path) -> int:
    corpus = Path(os.environ.get("VTS_FULLMARKS_OUT", "/expscratch/sgreenberg/fullmarks/corpus"))
    hits: dict[str, list[tuple]] = collections.defaultdict(list)
    for line in (corpus / "corpus.jsonl").open(encoding="utf-8"):
        page = json.loads(line)
        meta = page.get("meta", {})
        if page["source"] != "ucsf" or meta.get("industry") in (None, "Tobacco") or meta.get("tier") not in TIERS:
            continue
        text = " ".join(str(meta.get(k) or "") for k in ("author", "collection", "title")).lower()
        for company, pattern in COMPANIES.items():
            if re.search(pattern, text):
                hits[company].append(
                    (page["page_id"], meta.get("industry"), meta.get("collection"), meta.get("author"))
                )
    for company, rows in hits.items():
        by_collection = collections.Counter((r[1], r[2]) for r in rows)
        print(f"== {company}: {len(rows)} pages;", by_collection.most_common(6))
    out.write_text(json.dumps({c: [r[0] for r in rows] for c, rows in hits.items()}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
