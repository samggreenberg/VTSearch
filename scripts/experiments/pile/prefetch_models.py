"""Download the pile's embedder weights into the pile's models dir.

Run this once, on CPU, *before* the GPU build jobs. Two reasons it is its own
stage rather than a side effect of the first build:

* Parallel build jobs would otherwise race to populate the same shared HF cache.
* A download stall (the cluster's shared egress NAT is rate-limited for
  anonymous traffic) is much cheaper to hit on a CPU node than while holding a
  GPU allocation.
"""

from __future__ import annotations

import sys

import pile_config as pc

pc.setup_env()


def main() -> int:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    from vtscore import config as vtc  # noqa: PLC0415

    # HF-hub repo ids, resolvable by snapshot_download.  ``siglip_l`` is absent
    # on purpose: it loads through open_clip's pretrained registry, whose key
    # (``ViT-SO400M-14-SigLIP-384``) is an arch name, not a repo id -- passing it
    # to snapshot_download raises. open_clip fetches it from ``timm/<arch>`` on
    # first use and caches it under the same HF_HOME as the rest.
    wanted = {
        "siglip": vtc.SIGLIP_MODEL_ID,
        "siglip2": vtc.SIGLIP2_MODEL_ID,
        "siglip2_l": vtc.SIGLIP2_L_MODEL_ID,
        "dinov3_patch": vtc.DINOV3_MODEL_ID,
    }
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(wanted)

    failed = []
    for name in names:
        model_id = wanted.get(name)
        if model_id is None:
            print(f"[prefetch] unknown embedder {name!r}", flush=True)
            failed.append(name)
            continue
        print(f"[prefetch] {name}: {model_id}", flush=True)
        try:
            # cache_dir=MODELS, not the default HF_HOME/hub: the embedders load
            # with ``cache_dir=<VTSEARCH_MODELS_DIR>`` (see embedder_siglip2.py),
            # which puts ``models--*`` at the top of that dir. Prefetching to
            # HF_HOME/hub instead leaves weights the jobs cannot see, so each one
            # re-downloads -- and three parallel jobs race on the same dir.
            path = snapshot_download(model_id, cache_dir=str(pc.MODELS))
            print(f"[prefetch]   -> {path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next
            print(f"[prefetch]   FAILED {name}: {exc}", flush=True)
            failed.append(name)

    if failed:
        print(f"[prefetch] FAILED: {failed}", flush=True)
        return 1
    print("[prefetch] all weights present", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
