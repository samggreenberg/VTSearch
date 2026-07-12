"""Fit Toponymy on an audio dataset and dump the signpost tree.

Mirrors the architecture in ``docs/plans/vtsbrowse-toponymy.md``:

- ``embedding_vectors`` = the CLAP matrix (what VTSearch holds in memory),
- ``clusterable_vectors`` = a dedicated 5-D UMAP of that matrix
  (``metric="cosine"``, per Toponymy's examples),
- ``text_embedding_model`` = an adapter over the CLAP text branch
  (``MediaEmbedder.embed_text``) so keyphrases live in the same space,
- ``objects`` = the per-clip texts from ``make_texts.py`` (the
  ``object_to_text`` output, precomputed),
- namer = ``keyphrase`` (no-LLM fallback: top contrastive keyphrase) or
  ``hf`` (local HuggingFace LLM, default Qwen/Qwen2.5-7B-Instruct).

Outputs ``RESULTS/<dataset>/topo_<variant>_<namer>.json`` with per-layer
topic names, per-clip cluster labels, the cluster tree, 2-D anchors for
each topic (medoid of members in a 2-D UMAP), timings, and LLM call stats.

Usage::

    python run_toponymy.py esc50 clap_audioset keyphrase
    python run_toponymy.py esc50 clap_audioset hf --hf-model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import re

import common

common.setup_env()

import numpy as np  # noqa: E402


class ClapTextEncoder:
    """TextEmbedderProtocol adapter over the active embedder's text branch."""

    def __init__(self, embedder_name: str):
        from vtscore import media as media_registry

        self.embedder = media_registry.get_embedder(embedder_name)
        self.embedder.load_models()
        self._dim = None

    def encode(self, texts, show_progress_bar=False, **kwargs):
        out = []
        for t in texts:
            v = self.embedder.embed_text(str(t)) if str(t).strip() else None
            if v is None:
                v = np.zeros(self._dim or 512, dtype=np.float32)
            self._dim = len(v)
            out.append(np.asarray(v, dtype=np.float32))
        return np.stack(out)


class CallStats:
    def __init__(self):
        self.calls = 0
        self.samples = []

    def record(self, kind, prompt, response):
        self.calls += 1
        if len(self.samples) < 15:
            self.samples.append({"kind": kind, "prompt": str(prompt)[-1800:], "response": str(response)[:600]})


def make_keyphrase_namer(stats: CallStats):
    """No-LLM fallback namer: answers every naming prompt with the cluster's
    top contrastive keyphrase (parsed from the prompt Toponymy built)."""
    from toponymy.llm_wrappers import LLMWrapper

    class KeyphraseNamer(LLMWrapper):
        @property
        def supports_system_prompts(self) -> bool:
            return False

        def _respond(self, prompt: str) -> str:
            if "new_topic_name_mapping" in prompt:
                # Duplicate-name disambiguation pass: echo old names (an
                # honest no-LLM fallback cannot invent new distinctions).
                names = re.findall(r"^\s*(\d+)\.\s*(.+?)\s*$", prompt, re.MULTILINE)
                mapping = {f"{i}. {n}": n for i, n in names}
                resp = json.dumps({"new_topic_name_mapping": mapping, "topic_specificities": [0.5] * len(mapping)})
            else:
                kw = re.search(r"Keywords for this group include:\s*([^\n]+)", prompt)
                name = kw.group(1).split(",")[0].strip() if kw else "unnamed"
                resp = json.dumps({"topic_name": name, "topic_specificity": 0.5})
            stats.record("keyphrase", prompt, resp)
            return resp

        def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
            return self._respond(prompt)

        def _call_llm_with_system_prompt(self, system_prompt, user_prompt, temperature, max_tokens):
            return self._respond(user_prompt)

    return KeyphraseNamer()


def make_hf_namer(model_id: str, stats: CallStats):
    import torch
    from toponymy.llm_wrappers import HuggingFaceNamer

    class CountingHFNamer(HuggingFaceNamer):
        def _call_llm(self, prompt, temperature, max_tokens):
            r = super()._call_llm(prompt, temperature, max_tokens)
            stats.record("hf", prompt, r)
            return r

        def _call_llm_with_system_prompt(self, system_prompt, user_prompt, temperature, max_tokens):
            r = super()._call_llm_with_system_prompt(system_prompt, user_prompt, temperature, max_tokens)
            stats.record("hf_sys", {"system": system_prompt, "user": user_prompt}, r)
            return r

    return CountingHFNamer(
        model=model_id,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("variant", help="texts_<variant>.json to use as objects")
    ap.add_argument("namer", choices=["keyphrase", "hf"])
    ap.add_argument("--embedder", default="clap")
    ap.add_argument("--hf-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--umap-dim", type=int, default=5)
    ap.add_argument("--min-clusters", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus-desc", default=None)
    args = ap.parse_args()

    out = common.ds_dir(args.dataset)
    meta = common.load_json(out / "meta.json")
    emb = np.load(out / f"embeddings_{args.embedder}.npy")
    texts = common.load_json(out / f"texts_{args.variant}.json")
    n = min(len(texts), len(meta))
    meta, emb, texts = meta[:n], emb[:n], [str(t) for t in texts[:n]]
    print(f"{args.dataset}/{args.variant}/{args.namer}: {n} clips, emb dim {emb.shape[1]}")

    timings: dict = {}
    import umap

    with common.timed("umap_5d_clusterable", timings):
        umap5 = umap.UMAP(n_components=args.umap_dim, metric="cosine", random_state=args.seed).fit_transform(emb)
    with common.timed("umap_2d_layout", timings):
        umap2 = umap.UMAP(n_components=2, metric="cosine", random_state=args.seed).fit_transform(emb)

    from toponymy import Toponymy
    from toponymy.clustering import ToponymyClusterer

    stats = CallStats()
    with common.timed("namer_init", timings):
        if args.namer == "keyphrase":
            namer = make_keyphrase_namer(stats)
        else:
            namer = make_hf_namer(args.hf_model, stats)

    with common.timed("text_encoder_init", timings):
        text_encoder = ClapTextEncoder(args.embedder)

    corpus_desc = args.corpus_desc or f"a collection of short audio clips ({args.dataset})"
    model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=text_encoder,
        clusterer=ToponymyClusterer(min_clusters=args.min_clusters, verbose=True),
        object_description="audio clips",
        corpus_description=corpus_desc,
        verbose=True,
    )
    run_id = f"topo_{args.variant}_{args.namer}"
    try:
        with common.timed("toponymy_fit", timings):
            model.fit(texts, emb.astype(np.float32), umap5.astype(np.float32))
    except Exception as e:  # a variant that breaks the pipeline IS a result
        import traceback

        common.save_json(
            out / f"{run_id}.json",
            {
                "dataset": args.dataset,
                "variant": args.variant,
                "namer": args.namer,
                "n_clips": n,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-3000:],
                "timings_s": timings,
            },
        )
        raise

    layers = []
    for i, layer in enumerate(model.cluster_layers_):
        labels = np.asarray(layer.cluster_labels)
        names = model.topic_names_[i]
        anchors = []
        for c in range(len(names)):
            members = np.where(labels == c)[0]
            if len(members):
                mid = np.median(umap2[members], axis=0)
                anchors.append([round(float(mid[0]), 3), round(float(mid[1]), 3)])
            else:
                anchors.append(None)
        layers.append(
            {
                "layer": i,
                "n_topics": len(names),
                "topic_names": list(names),
                "cluster_labels": labels.tolist(),
                "noise_frac": round(float((labels < 0).mean()), 4),
                "anchors": anchors,
            }
        )

    tree = {f"{k[0]}:{k[1]}": [f"{c[0]}:{c[1]}" for c in v] for k, v in model.cluster_tree_.items()}
    common.save_json(
        out / f"{run_id}.json",
        {
            "dataset": args.dataset,
            "variant": args.variant,
            "namer": args.namer,
            "hf_model": args.hf_model if args.namer == "hf" else None,
            "n_clips": n,
            "layers": layers,
            "cluster_tree": tree,
            "umap2d": np.round(umap2, 3).tolist(),
            "timings_s": timings,
            "llm_calls": stats.calls,
            "llm_samples": stats.samples,
        },
    )


if __name__ == "__main__":
    main()
