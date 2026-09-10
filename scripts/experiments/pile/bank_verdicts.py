"""Export every detector's verdicts. Kind is decided by the QUESTION, not the name.

The previous version keyed the filename off the substring "below-cut", which the
rename removed from every detector name -- so both kinds collapsed onto one
filename and the later write clobbered the earlier. The question a detector asks
is the thing that actually distinguishes them, so it is what the filename
follows, and a collision is now an error rather than a silent overwrite.
"""

import json
import pathlib
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

_SQUEUE = shutil.which("squeue") or "/usr/bin/squeue"
_ARGS = [_SQUEUE, "-u", "sgreenberg", "-h", "-n", "vtsearch", "-o", "%N"]
node = (
    subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        _ARGS, capture_output=True, text=True, check=False
    )
    .stdout.strip()
    .split()[0]
)
BASE = f"http://{node}:11850"
REPO = pathlib.Path("/exp/sgreenberg/projects/vts-annq-3720/scripts/experiments/pile/human_record")


def get(p):
    return json.load(urllib.request.urlopen(BASE + p, timeout=120))  # noqa: S310 - our own app


QUESTION = {
    "slate": "is this box one?",
    "belowcut": "is there one anywhere in this image?",
    "recheck": "does this photo contain one under the revised rule?",
}

written: dict[pathlib.Path, str] = {}
fail = False
for d in sorted(get("/api/detectors/registry")["detectors"], key=lambda x: x["name"]):
    if not (d.get("num_training") or 0):
        continue
    name = d["name"]
    # Three questions live in this dashboard and they must never share a file.
    # "(any in image, no box)" and the older "[below-cut: ...]" ask whether the
    # class is anywhere in the image; "-- recheck:" re-asks the class question of
    # an OLD positive after a definition change; everything else is the slate's
    # "is this box one?". The first version of this script keyed off a substring
    # the dashboard rename had already removed, and silently overwrote one file
    # with another -- so the mapping is explicit and a collision is fatal.
    if "-- recheck" in name:
        kind = "recheck"
    elif ("any in image" in name) or ("below-cut" in name):
        kind = "belowcut"
    else:
        kind = "slate"
    rule = name.split(" [")[0].split(" (")[0].split(" -- ")[0]
    cls = d.get("text_query") or rule.split()[0]
    live = get(f"/api/detectors/{urllib.parse.quote(name)}/labels-detail")
    good, bad = live.get("good", []), live.get("bad", [])
    p = REPO / f"LABELSETS__{kind}__{cls.replace(' ', '_')}.json"
    if p in written:
        print(f"  COLLISION {p.name}: {written[p]!r} and {name!r} -- refusing")
        fail = True
        continue
    written[p] = name
    p.write_text(
        json.dumps(
            {
                "detector_name": name,
                "rule": rule,
                "class": cls,
                "kind": kind,
                "question": QUESTION[kind],
                "source": "OWLv2-screened per-class review, exported from the live app",
                "exported": "2026-09-09",
                "good": good,
                "bad": bad,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"  {p.name:<40} {len(good):>4}g {len(bad):>4}b   <- {name[:44]}")
print(f"\n{len(written)} files written, {sum(1 for _ in written)} distinct detectors")
sys.exit(1 if fail else 0)
