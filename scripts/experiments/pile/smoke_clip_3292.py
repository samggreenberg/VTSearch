#!/usr/bin/env python3
"""Gate the #3292 pile build: do both CLIP arms actually work end to end?

Runs *before* 4200 images are embedded, because the two ways this arm can fail
are both silent:

* **No text tower.**  ``CALIB_REQUIRE_OPENING=text`` cells open on a typed
  query.  When ``embed_text_query`` yields nothing the harness falls back to the
  known-good opening, the arm stops being comparable to the SigLIP runs it is
  meant to be a third row beside, and *nothing raises* (#3278).
* **A wrong dimension.**  ``clip_l`` is in this study specifically because it is
  768-d like ``siglip``; if the checkpoint silently gave 512-d, the one confound
  the arm was added to remove would be back, and only this assert would say so.

Also checks that the text and image spaces are actually aligned -- a text vector
of the right shape that ranks nothing is still a dead opening -- by asking each
encoder to prefer the matching caption over a mismatched one.
"""

from __future__ import annotations

import sys

import numpy as np
import pile_config as pc

pc.setup_env()

#: (name, expected dim).  Pinned, not read from the embedder: the point is to
#: catch a checkpoint that quietly changed, and a self-reported dim cannot.
ARMS = [("clip", 512), ("clip_l", 768)]


def main() -> int:
    from PIL import Image

    from vtscore.media import get_embedder

    failures: list[str] = []

    # Two flat colour fields: crude, but they are unambiguously different and
    # need no dataset staged to run.
    red = Image.new("RGB", (256, 256), (200, 30, 30))
    blue = Image.new("RGB", (256, 256), (30, 30, 200))

    for name, want_dim in ARMS:
        emb = get_embedder(name)
        print(f"[smoke] {name}: {emb.display_name}  model={emb.model_id}", flush=True)

        if emb.embedding_dim != want_dim:
            failures.append(f"{name}: declares embedding_dim={emb.embedding_dim}, expected {want_dim}")

        iv = emb.embed_pil_image(red)
        if iv is None:
            failures.append(f"{name}: embed_pil_image returned None")
            continue
        if iv.shape != (want_dim,):
            failures.append(f"{name}: image vector is {iv.shape}, expected ({want_dim},)")

        tv = emb.embed_text("a photo of something red")
        if tv is None:
            failures.append(f"{name}: NO TEXT TOWER (embed_text returned None) -- cells would open on known_good")
            continue
        if tv.shape != (want_dim,):
            failures.append(f"{name}: text vector is {tv.shape}, expected ({want_dim},)")
            continue

        # Alignment: the red field must rank "red" above "blue" and vice versa.
        # A text vector of the right shape in an unaligned space is still a dead
        # opening, and shape alone cannot see that.
        def cos(a: np.ndarray, b: np.ndarray) -> float:
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        t_red = emb.embed_text("a photo of something red")
        t_blue = emb.embed_text("a photo of something blue")
        v_red, v_blue = iv, emb.embed_pil_image(blue)
        if v_blue is None or t_red is None or t_blue is None:
            failures.append(f"{name}: could not build the alignment probe")
            continue
        margin_r = cos(v_red, t_red) - cos(v_red, t_blue)
        margin_b = cos(v_blue, t_blue) - cos(v_blue, t_red)
        print(f"[smoke] {name}: dim={iv.shape[0]} margin(red)={margin_r:+.3f} margin(blue)={margin_b:+.3f}", flush=True)
        if margin_r <= 0 or margin_b <= 0:
            failures.append(
                f"{name}: text and image spaces are not aligned "
                f"(red margin {margin_r:+.3f}, blue margin {margin_b:+.3f})"
            )

    if failures:
        print("\n[smoke] FAILED:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    print("\n[smoke] both CLIP arms embed images and text, at the right dims, in aligned spaces.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
