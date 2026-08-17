"""Assert the two bench arms are actually paired, before either array is launched.

    python verify_pairing.py --arms fp32_l40s,fp16_l40s --root <bench root>

The precision contrast is only readable as a *paired* difference if both arms
select the same categories, in the same order, with the same exemplar
candidates.  By construction they should: category selection reads boxes and
counts (no vectors), and exemplar candidates are drawn from a per-category RNG
seed.  But "by construction" with no check is the shape of every mis-specified
arm in this repo's history (#2877, #2897, #2905), and here the failure is
invisible in the output — two unpaired arms still produce a table, just one whose
error bars are a fiction and whose difference includes the category draw.

Exit 1 on any disagreement, so the launcher can gate on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", required=True, help="comma-separated arm names")
    ap.add_argument("--root", required=True, help="bench root holding <arm>/results")
    args = ap.parse_args(argv)

    arms = [a for a in args.arms.split(",") if a]
    root = Path(args.root)
    problems: list[str] = []
    infos: dict[str, dict] = {}

    for arm in arms:
        path = root / arm / "results" / "prepare_info.json"
        if not path.exists():
            problems.append(f"{arm}: no prepare_info.json at {path} — prepare did not finish")
            continue
        infos[arm] = json.loads(path.read_text())
        failed = infos[arm].get("failed") or []
        if failed:
            problems.append(f"{arm}: prepare reported failures: {failed}")

    if len(infos) < 2:
        for p in problems:
            log(f"PROBLEM: {p}")
        log("\nfewer than two arms prepared; nothing to pair")
        return 1

    base_arm = arms[0]
    base = infos[base_arm]
    for arm in arms[1:]:
        other = infos[arm]
        for ds, per_emb in base.get("datasets", {}).items():
            for emb, entry in per_emb.items():
                o_entry = (other.get("datasets", {}).get(ds) or {}).get(emb)
                if o_entry is None:
                    problems.append(f"{arm}: missing {ds} x {emb}, which {base_arm} has")
                    continue
                # Categories: same set AND same order (the array index is
                # positional, so a reordering silently re-pairs every cell).
                if entry.get("selected_categories") != o_entry.get("selected_categories"):
                    problems.append(
                        f"{ds} x {emb}: category selection differs.\n"
                        f"    {base_arm}: {entry.get('selected_categories')}\n"
                        f"    {arm}: {o_entry.get('selected_categories')}"
                    )
                if entry.get("n_medias") != o_entry.get("n_medias"):
                    problems.append(
                        f"{ds} x {emb}: media count differs ({entry.get('n_medias')} vs {o_entry.get('n_medias')})"
                    )
                if entry.get("dim") != o_entry.get("dim"):
                    problems.append(f"{ds} x {emb}: embedding dim differs ({entry.get('dim')} vs {o_entry.get('dim')})")

        # Exemplar candidate ids, per category, from the crops sidecar.
        for ds, per_emb in base.get("datasets", {}).items():
            for emb in per_emb:
                name = f"{ds}_{emb}.json"
                b_path = root / base_arm / "results" / "crops" / name
                o_path = root / arm / "results" / "crops" / name
                if not (b_path.exists() and o_path.exists()):
                    # The basename is built by cfg.crops_basename; if the guess
                    # is wrong say so rather than silently declaring a pass.
                    matches = sorted((root / base_arm / "results" / "crops").glob(f"*{emb}*.json"))
                    if not matches:
                        problems.append(
                            f"{ds} x {emb}: no exemplar-candidate JSON found; cannot verify the seed pairing"
                        )
                        continue
                    b_path = matches[0]
                    o_path = root / arm / "results" / "crops" / b_path.name
                    if not o_path.exists():
                        problems.append(f"{arm}: missing {o_path.name}")
                        continue
                b_cand, o_cand = json.loads(b_path.read_text()), json.loads(o_path.read_text())
                if b_cand != o_cand:
                    diff = [c for c in set(b_cand) | set(o_cand) if b_cand.get(c) != o_cand.get(c)]
                    problems.append(
                        f"{ds} x {emb}: exemplar candidates differ for {diff[:5]}"
                        f"{' (and more)' if len(diff) > 5 else ''} — the arms would seed from different images"
                    )

    log(f"checked {len(infos)} arms: {', '.join(infos)}")
    for arm, info in infos.items():
        for ds, per_emb in info.get("datasets", {}).items():
            for emb, entry in per_emb.items():
                log(
                    f"  {arm:14s} {ds:18s} {emb:12s} {entry.get('n_medias')} medias, "
                    f"dim {entry.get('dim')}, {len(entry.get('selected_categories') or [])} categories"
                )

    if problems:
        log("")
        for p in problems:
            log(f"PROBLEM: {p}")
        log(f"\n{len(problems)} problem(s) — the arms are NOT paired; do not launch the arrays.")
        return 1
    log("\npairing verified: identical categories, order, populations and exemplar candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
