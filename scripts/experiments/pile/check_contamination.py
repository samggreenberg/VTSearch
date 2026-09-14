"""Did any verdict land on an image outside its own dataset?

A detector trained while a DIFFERENT dataset is loaded will happily accept
labels against that dataset's images, and nothing in the stored verdict says
which set it came from. So the check is membership: every labelled filename must
exist in the folder the detector's own slate was built from.
"""

import json
import pathlib

HR = pathlib.Path("/exp/sgreenberg/projects/vts-annq-3720/scripts/experiments/pile/human_record")
SL = pathlib.Path("/expscratch/sgreenberg/vlm-3720/slates")
BC = pathlib.Path("/expscratch/sgreenberg/vlm-3720/belowcut")

bad_total = 0
for f in sorted(HR.glob("LABELSETS__*.json")):
    doc = json.loads(f.read_text())
    if not isinstance(doc, dict):
        continue  # older WORK__ manifests are plain lists
    cls, kind = doc.get("class"), doc.get("kind")
    if not cls or kind not in ("slate", "belowcut"):
        continue
    root = (SL if kind == "slate" else BC) / cls.replace(" ", "_") / "images"
    if not root.exists():
        print(f"  {f.name}: folder missing ({root})")
        continue
    have = {p.name for p in root.glob("*.jpg")}
    labelled = [r["filename"] for side in ("good", "bad") for r in doc.get(side, [])]
    stray = [n for n in labelled if n not in have]
    flag = "" if not stray else f"   <-- {len(stray)} NOT IN THIS DATASET"
    bad_total += len(stray)
    print(f"  {kind:<9}{cls:<14}{len(labelled):>5} labelled, {len(have):>5} in folder{flag}")
    for n in stray[:5]:
        print(f"        stray: {n}")
print(f"\ntotal verdicts on images outside their own dataset: {bad_total}")
