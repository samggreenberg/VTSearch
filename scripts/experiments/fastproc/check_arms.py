"""Structural gate on the #3146 arms — run before reading a single number.

    python check_arms.py

Every check here corresponds to a way a previous study reached a wrong verdict
without anything looking wrong at the time.  They are cheap, and they run
against files on disk rather than against what a launcher believed it did.

1. **Provenance exists and matches the arm table.**  A cell with no provenance
   cannot be attributed to a treatment at all.
2. **The processor class the arm actually loaded is the one it asked for.**
   transformers *warns and falls back* when a backend is unavailable, so the
   default outcome of an impossible request is a mislabelled arm, not an error.
   This is the check that would have caught #3146's own premise.
3. **Every arm ran on the same node.**  #3160 established that a device swap
   moves siglip2_l fp32 by 1.5e-4 — larger than some of the effects here — so an
   arm that landed elsewhere is confounded and must be rebuilt, not adjusted for.
4. **Every arm covers the same medias.**  A short cell makes a paired
   comparison silently unpaired.
5. **No zero-byte cells.**  They count as "done" to the resume path.
6. **The reference rebuild is compared to the published pile cell**, and the
   result is *reported* rather than asserted: whichever way it comes out is a
   finding, and a hard failure here would only hide it.

Exit 1 if any of 1-5 fails.  6 never fails the gate; it prints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

import fastproc_config as fcfg  # noqa: E402

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"ok:   {msg}")


def backend_of(class_name: str, transformers_major: int) -> str:
    """Delegates to the app — see ``build_arm._backend_of`` for why."""
    from vtscore.config import processor_backend_from_class_name

    return processor_backend_from_class_name(class_name, transformers_major) or "torchvision"


def main() -> int:
    provs: dict[str, dict] = {}
    for arm in fcfg.ARMS:
        p = fcfg.provenance_path(arm)
        if not p.exists():
            fail(f"{arm}: no provenance.json at {p}")
            continue
        provs[arm] = json.loads(p.read_text())
    if not provs:
        print("\nno arms have provenance; nothing to check")
        return 1

    # 1-2. the arm did what the table says
    for arm, prov in provs.items():
        want_b = fcfg.ARMS[arm]["backend"]
        want_d = fcfg.ARMS[arm]["device"]
        if prov.get("backend") != want_b or prov.get("processor_device") != want_d:
            fail(
                f"{arm}: provenance says {prov.get('backend')}/{prov.get('processor_device')}, table says {want_b}/{want_d}"
            )
            continue
        probe = prov.get("probe", {})
        try:
            major = int(str(probe.get("transformers", "5")).split(".")[0])
        except ValueError:
            major = 5
        for emb, per in probe.get("embedders", {}).items():
            got_b = backend_of(per["processor_class"], major)
            got_d = per["pixel_device"].split(":")[0]
            if got_b != want_b:
                fail(f"{arm} x {emb}: loaded {per['processor_class']} (= {got_b}), asked for {want_b}")
            elif got_d != want_d:
                fail(f"{arm} x {emb}: pixels on {per['pixel_device']}, asked for {want_d}")
            else:
                ok(f"{arm} x {emb}: {per['processor_class']} on {per['pixel_device']}")

    # 3. one node for every arm
    nodes = {arm: prov.get("hostname") for arm, prov in provs.items()}
    distinct = sorted({n for n in nodes.values() if n})
    if len(distinct) > 1:
        fail(
            f"arms ran on {len(distinct)} different nodes {distinct}: {nodes} — a device swap alone moves siglip2_l by 1.5e-4 (#3160)"
        )
    else:
        ok(f"every arm ran on {distinct[0] if distinct else '?'}")

    # also: the CPU allocation, since one arm's treatment is where work runs
    threads = {arm: (prov.get("cpus"), prov.get("torch_threads")) for arm, prov in provs.items()}
    if len({t for t in threads.values()}) > 1:
        fail(
            f"arms had different CPU/thread allocations {threads} — a timing comparison across them is not a treatment effect"
        )
    else:
        ok(f"identical CPU allocation across arms: {next(iter(threads.values()), '?')}")

    # 4-5. cells present, non-empty, same medias
    from _cells_io import load_medias  # noqa: PLC0415

    ids_by_arm: dict[tuple[str, str], set] = {}
    for arm in provs:
        for emb in fcfg.EMBEDDERS:
            cell = fcfg.arm_cell(arm, emb)
            if not cell.exists():
                fail(f"{arm} x {emb}: cell missing at {cell}")
                continue
            if cell.stat().st_size == 0:
                fail(f"{arm} x {emb}: ZERO-BYTE cell — resume would skip it")
                continue
            ids_by_arm[(arm, emb)] = set(load_medias(cell))
    for emb in fcfg.EMBEDDERS:
        sets = {arm: s for (arm, e), s in ids_by_arm.items() if e == emb}
        if len(sets) < 2:
            continue
        sizes = {arm: len(s) for arm, s in sets.items()}
        if len(set(sizes.values())) > 1:
            fail(f"{emb}: arms cover different media counts {sizes} — the comparison is not paired")
        else:
            ok(f"{emb}: all {len(sets)} arms cover {next(iter(sizes.values()))} medias")

    # 6. the adjudication — reported, never fatal
    print("\n--- reference rebuild vs the published pile cell (reported, not gated) ---")
    import numpy as np

    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    def mat(path: Path):
        medias = load_medias(path)
        ids, rows = [], []
        for mid in sorted(medias):
            v = media_embedding(medias[mid])
            if v is None:
                continue
            ids.append(mid)
            rows.append(np.asarray(v, dtype=np.float64))
        m = np.vstack(rows)
        return ids, m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    for emb in fcfg.EMBEDDERS:
        pub = fcfg.shared_cell(emb)
        if not pub.exists():
            print(f"  {emb}: no published cell at {pub}")
            continue
        pub_ids, pub_m = mat(pub)
        for arm in provs:
            cell = fcfg.arm_cell(arm, emb)
            if not cell.exists():
                continue
            ids, m = mat(cell)
            if ids != pub_ids:
                print(f"  {emb} {arm:14s} id set differs ({len(ids)} vs {len(pub_ids)})")
                continue
            d = 1.0 - np.clip((pub_m * m).sum(axis=1), -1.0, 1.0)
            print(
                f"  {emb:12s} {arm:14s} median 1-cos {np.median(d):.3e}  max {d.max():.3e}  "
                f"{(d == 0).mean() * 100:5.1f}% identical"
            )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED — do not read the numbers yet")
        return 1
    print("all structural checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
