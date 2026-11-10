"""Fit a face space's ball and write its entry to ``face_ball.json`` (#5497).

The face crops' line (#5478) is a cosine floor per balance, and the floors belong to one face
embedder's space: a server whose face provider is another embedder (the ``ext`` provider's, or a
different ``oss`` pack) fits its own here, from a labelled face set that provider embedded.  The
fit, its go/no-go check and the entry's format are :mod:`vtscore.media.face.face_ball_fit` and
:mod:`vtscore.media.face.face_ball`; this script loads the set, names the space, records how the
entry was made, and writes it.

Input: one or more cell pickles (``{id: media}``, every media with the face vector and the person
it shows in ``categories``; FHIBE's are ``<data>/embeddings/fhibe_faces_ik_{1024,640}__face.pkl``),
and the people to fit on, one per line (``#`` comments skipped).  The space is the medias' own
``Face Compat`` tag; ``--compat`` names it for a set that carries none, and must agree with one
that does.

The entry is a calculated asset: it goes beside the face pipeline config in use (default
``$VTSEARCH_MODELS_DIR/imagekit/face_ball.json``, beside the face pack; beside the config
``IMAGEKIT_FACE_CONFIG`` names when it is set), never into the repo.  Other spaces' entries in the
file are kept.

Usage::

    python scripts/data/fit_face_ball.py \\
        --medias <data>/embeddings/fhibe_faces_ik_1024__face.pkl <data>/embeddings/fhibe_faces_ik_640__face.pkl \\
        --people <runs>/2026-10-09-k4/identities.txt --corpus FHIBE [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess  # noqa: S404 - fixed argv, no shell
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "calibration"))


def _commit() -> dict[str, object]:
    def git(*args: str) -> str:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(REPO), *args],  # noqa: S607 - git resolves on PATH, as everywhere in this repo
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def read_people(path: Path) -> list[str]:
    """The people in *path*, one per line, ``#`` comments and blank lines skipped."""
    return [s for line in path.read_text().splitlines() if (s := line.strip()) and not s.startswith("#")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--medias", type=Path, nargs="+", required=True, help="cell pickles of one face space")
    ap.add_argument("--people", type=Path, required=True, help="the people to fit on, one per line")
    ap.add_argument("--compat", default="", help="the space's Face Compat tag, for medias that carry none")
    ap.add_argument("--corpus", default="", help="a name for the labelled set, for the provenance")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="split seeds (default: 0, the review's)")
    ap.add_argument("--out", type=Path, default=None, help="the file to write (default: the models folder's)")
    ap.add_argument("--dry-run", action="store_true", help="print the entry and write nothing")
    args = ap.parse_args(argv)

    from _cells_io import load_medias  # noqa: PLC0415

    from vtscore.media.face.face_ball import face_ball_path, face_compat_of, write_entry  # noqa: PLC0415
    from vtscore.media.face.face_ball_fit import fit_face_ball  # noqa: PLC0415

    corpora = {p.name.removesuffix(".pkl"): load_medias(p, repair=True) for p in args.medias}
    tags = {face_compat_of(m) for m in corpora.values()}
    if len(tags) != 1:
        ap.error(f"the medias carry more than one face space: {sorted(map(str, tags))}")
    (tag,) = tags
    if tag and args.compat and tag != args.compat:
        ap.error(f"--compat {args.compat!r} disagrees with the medias' own tag {tag!r}")
    compat = tag or args.compat
    if not compat:
        ap.error("the medias carry no Face Compat tag: name their space with --compat")
    people = read_people(args.people)
    ball = fit_face_ball(
        compat,
        corpora,
        people,
        args.seeds,
        provenance={
            "corpus": args.corpus or ", ".join(corpora),
            "medias": [str(p) for p in args.medias],
            "people_file": str(args.people),
            "people_sha256": hashlib.sha256(args.people.read_bytes()).hexdigest(),
            **_commit(),
            "fitted_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "tool": "scripts/data/fit_face_ball.py",
        },
    )
    print(json.dumps({compat: ball.to_json()}, indent=1))
    if not args.dry_run:
        path = write_entry(ball, args.out or face_ball_path())
        print(f"wrote {compat} ({'go' if ball.use else 'no-go'}) to {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
