"""How would the dropped planter screen have done against the real verdicts?"""

import json
import pathlib

HR = pathlib.Path("/exp/sgreenberg/projects/vts-annq-3720/scripts/experiments/pile/human_record")
flag = {
    (f["class"], int(f["image_id"]))
    for f in json.loads(pathlib.Path("/expscratch/sgreenberg/vlm-3720/old_pots.json").read_text())["flagged"]
}
for cls in ("bowl", "vase"):
    doc = json.loads((HR / f"LABELSETS__recheck__{cls}.json").read_text())
    # Bad = the photo does NOT contain one = the old positive is retired
    retired, kept = set(), set()
    for side, tgt in (("bad", retired), ("good", kept)):
        for r in doc.get(side, []):
            try:
                tgt.add(int(str(r["filename"]).split(".")[0]))
            except (ValueError, KeyError):
                continue
    fl = {i for c, i in flag if c == cls}
    tp = len(fl & retired)
    fp = len(fl & kept)
    fn = len(retired - fl)
    print(f"{cls}: {len(retired)} retired of {len(retired) + len(kept)}")
    print(f"   screen flagged {len(fl)}: {tp} correctly, {fp} wrongly")
    print(f"   screen MISSED {fn} of the {len(retired)} that had to go")
    if len(retired):
        print(f"   screen recall {tp / len(retired):.2f}, precision {tp / len(fl) if fl else float('nan'):.2f}")
