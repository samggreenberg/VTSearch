"""Pluggable region sources for the small-object-detection sweep.

A *region source* turns one image into a bag of ``(box, vector)`` candidates,
all in a single embedding space, so the scoring heads
(:mod:`vtscore.eval.scoring_heads`) and the K-curve
(:mod:`vtscore.eval.region_curve`) can treat "whole / sliding-window /
DINO-proposed / HAC" uniformly. Everything downstream (MLP, cosine,
cross-calibration, weighted FPR+FNR) is identical regardless of where the
candidates came from — the only thing that varies is the source.

Two structurally different families (see docs/plans/... small-object sweep):

* **Crop-and-re-embed** (``CropReembedSource``): a proposer yields boxes, the
  box pixels are cropped and re-embedded with a text/image embedder
  (SigLIP/CLIP/SigLIP2). Small objects get "zoomed" to model resolution.
* **Patch-grid** (``PatchTreeSource`` / ``PatchBoxPoolSource``): one patch
  forward (DINOv2/v3) gives a patch grid; regions are HAC-tree nodes or boxes
  pooled from the grid via ``box_to_vote_vector`` — cheap (no re-embed) but
  floored by the patch-grid resolution.

Every source keeps its exemplar (GT-box) and whole-image vectors in the *same*
space as its region vectors, so a head never mixes embedding spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

Box = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Proposers (extracted / adapted from scripts/vg/eval_crop_clip.py)
# ---------------------------------------------------------------------------


def sliding_boxes_by_scale(
    w: int, h: int, scales: list[float], overlap: float, min_window: int
) -> dict[float, list[Box]]:
    """Multiscale square windows in PIXEL coords, grouped by scale."""
    short = min(w, h)
    out: dict[float, list[Box]] = {}
    for f in scales:
        side = min(int(round(f * short)), short)
        if side < min_window:
            continue
        stride = max(1, int(round(side * (1.0 - overlap))))
        xs = list(range(0, max(1, w - side + 1), stride))
        ys = list(range(0, max(1, h - side + 1), stride))
        if w - side > 0 and xs[-1] != w - side:
            xs.append(w - side)
        if h - side > 0 and ys[-1] != h - side:
            ys.append(h - side)
        boxes = [(float(x), float(y), float(x + side), float(y + side)) for x in xs for y in ys]
        if boxes:
            out[f] = list(dict.fromkeys(boxes))
    return out


def crops_from_boxes(img: "Image.Image", boxes_px: list[Box]) -> list["Image.Image"]:
    """Crop each PIXEL box out of *img* (corners clamped to >= 1px inside bounds)."""
    crops = []
    for b in boxes_px:
        x0 = max(0, min(int(b[0]), img.width - 1))
        y0 = max(0, min(int(b[1]), img.height - 1))
        x1 = max(x0 + 1, min(int(b[2]), img.width))
        y1 = max(y0 + 1, min(int(b[3]), img.height))
        crops.append(img.crop((x0, y0, x1, y1)))
    return crops


def _norm_boxes(boxes_px: list[Box], w: int, h: int) -> np.ndarray:
    """(N,4) pixel boxes -> normalized [0,1] (x0,y0,x1,y1)."""
    if not boxes_px:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.asarray(boxes_px, dtype=np.float32)
    arr[:, [0, 2]] /= float(w)
    arr[:, [1, 3]] /= float(h)
    return np.clip(arr, 0.0, 1.0)


def _as_box(b) -> Box:
    """Coerce a length-4 array/sequence into a typed (x0,y0,x1,y1) tuple."""
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def _covering_box(boxes: list[Box]) -> Box:
    """Minimal axis-aligned box covering every instance box (matches
    :func:`vtscore.eval.labels.region_box_for_category`): the box a Good vote on
    an image with several instances of the class designates."""
    xs0 = [float(b[0]) for b in boxes]
    ys0 = [float(b[1]) for b in boxes]
    xs1 = [float(b[2]) for b in boxes]
    ys1 = [float(b[3]) for b in boxes]
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _node_cell_masks(tree, grid_shape: tuple[int, int]) -> np.ndarray:
    """Per-node patch-cell union masks for a region tree (shape ``(N, H, W)`` bool).

    Leaf = its own ``cell_mask``; CLS (index 0, no mask, no children) = all-True (the whole
    image); internal = union of its children's masks. Computed bottom-up over the flat list
    (build order guarantees children precede their parent), so one pass suffices."""
    n = len(tree)
    h, w = grid_shape
    masks = np.zeros((n, h, w), dtype=bool)
    for i, r in enumerate(tree):
        if r.cell_mask is not None:
            masks[i] = np.asarray(r.cell_mask, dtype=bool)
        elif r.children is None:  # CLS full-image node
            masks[i] = True
        else:
            a, b = r.children
            masks[i] = masks[a] | masks[b]
    return masks


def _denorm_boxes(boxes: np.ndarray, w: int, h: int) -> list[Box]:
    """(N,4) normalized boxes -> pixel-coordinate box tuples."""
    out: list[Box] = []
    for x0, y0, x1, y1 in boxes:
        out.append((float(x0 * w), float(y0 * h), float(x1 * w), float(y1 * h)))
    return out


class Proposer(Protocol):
    """image -> list of NORMALIZED [0,1] candidate boxes."""

    name: str

    def __call__(self, image: "Image.Image") -> np.ndarray: ...


class WholeProposer:
    name = "whole"

    def __call__(self, image: "Image.Image") -> np.ndarray:  # noqa: ARG002
        return np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)


@dataclass
class SlidingWindowProposer:
    scales: tuple[float, ...] = (1.0, 0.5, 0.25)
    overlap: float = 0.5
    min_window: int = 64
    name: str = "sliding"

    def __call__(self, image: "Image.Image") -> np.ndarray:
        by_scale = sliding_boxes_by_scale(image.width, image.height, list(self.scales), self.overlap, self.min_window)
        flat: list[Box] = [b for boxes in by_scale.values() for b in boxes]
        boxes = _norm_boxes(flat, image.width, image.height)
        if boxes.shape[0] == 0:  # image smaller than min_window at every scale
            return np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        return boxes


class DinoRegionProposer:
    """DINOv2/v3 patch forward -> HAC region boxes (normalized). Proposal only.

    The proposed crops are re-embedded by whatever embedder the enclosing
    ``CropReembedSource`` holds (as in ``eval_crop_clip``'s dino_v2/v3 methods).
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        num_register_tokens: int = 0,
        k: int = 12,
        alpha: float = 0.5,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        from vtscore.media.embedder import hf_token

        self.name = "dino"
        self.k = k
        self.alpha = alpha
        self.device = device
        self.num_register_tokens = num_register_tokens
        tok = hf_token() or None
        self.model = AutoModel.from_pretrained(model_id, attn_implementation="eager", token=tok).eval().to(device)
        self.proc = AutoImageProcessor.from_pretrained(model_id, token=tok)

    def __call__(self, image: "Image.Image") -> np.ndarray:
        import torch

        from vtscore.media.patch_embed import build_region_tree, hf_vit_to_patch_output

        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        pe = hf_vit_to_patch_output(outputs, num_register_tokens=self.num_register_tokens)
        if pe is None:
            return np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        boxes = [r.box for r in build_region_tree(pe, k=self.k, alpha=self.alpha)]
        return np.asarray(boxes, dtype=np.float32) if boxes else np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Prepared-image + region-source interfaces
# ---------------------------------------------------------------------------


@dataclass
class PreparedImage:
    """One image's region candidates + whole vector, all L2-normalized (N,D)/(D,)."""

    boxes: np.ndarray  # (N, 4) normalized
    vecs: np.ndarray  # (N, D)
    whole_vec: np.ndarray  # (D,)
    exemplars: np.ndarray  # (M, D) — GT-box exemplar vectors (empty when none requested)
    # (N,) bool, aligned with ``boxes``/``vecs``: True for the "childless" nodes a
    # Bad vote floods as negatives — the CLS full-image node + HAC leaves — and
    # False for internal HAC merge nodes (correlated duplicates, dropped in the
    # region-voting negative set). ``None`` when the source has no tree structure
    # (crop/sliding/whole), where every candidate is a leaf by construction.
    leaf_mask: np.ndarray | None = None
    # (N, 2) int, aligned with ``boxes``/``vecs``: the two child indices (into this
    # same flat node list) of each HAC merge node, or ``(-1, -1)`` for a childless
    # node (CLS + leaves). Lets the labeling-trace visualizer redraw the HAC
    # dendrogram. ``None`` when the source has no tree structure (crop/sliding/whole).
    children: np.ndarray | None = None
    # (N, H, W) bool, aligned with ``boxes``: each node's patch-cell union footprint
    # (leaf = its own cell mask, CLS = all-True, internal = union of children). Lets the
    # labeling-trace visualizer draw masked-cell pixel thumbnails (the true footprint, not
    # the loose bbox). ``None`` for tree-less sources.
    cell_masks: np.ndarray | None = None
    # (H, W) float, the per-patch CLS→patch attention/saliency grid, for the trace's
    # inferno attention overlay. ``None`` for tree-less sources.
    saliency: np.ndarray | None = None


class RegionSource(Protocol):
    name: str
    input_dim: int
    supports_text: bool

    def prepare(self, image: "Image.Image", gt_boxes: list[Box] | None = None) -> PreparedImage:
        """Run the (expensive) forward once and return all vectors for the image.

        ``gt_boxes`` (normalized) are pooled/cropped into positive exemplar
        vectors in the same space as the regions; pass ``None`` for test images.
        """
        ...

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Query vector for the cosine head (``None`` when not text-capable)."""
        ...


# ---------------------------------------------------------------------------
# Concrete sources
# ---------------------------------------------------------------------------


class _CropReembedSource:
    """Proposer boxes -> crop pixels -> re-embed with a text/image embedder."""

    def __init__(self, embedder, proposer: Proposer, name: str) -> None:
        self._emb = embedder
        self._proposer = proposer
        self.name = name
        self.supports_text = bool(embedder.supports_text)
        self.input_dim = _embedder_dim(embedder)

    def _forward(self, images: list["Image.Image"]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.input_dim), dtype=np.float32)
        return np.asarray(self._emb._forward_pil_batch(images), dtype=np.float32)

    def prepare(self, image: "Image.Image", gt_boxes: list[Box] | None = None) -> PreparedImage:
        norm_boxes = self._proposer(image)
        px = _denorm_boxes(norm_boxes, image.width, image.height)
        vecs = self._forward(crops_from_boxes(image, px)) if len(px) else np.zeros((0, self.input_dim), np.float32)
        whole_vec = self._forward([image])[0]
        if vecs.shape[0] == 0:  # degenerate proposer -> fall back to whole
            norm_boxes = np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
            vecs = whole_vec[None, :]
        exemplars = np.zeros((0, self.input_dim), np.float32)
        if gt_boxes:
            gx = _denorm_boxes(np.asarray(gt_boxes, dtype=np.float32), image.width, image.height)
            exemplars = self._forward(crops_from_boxes(image, gx))
        return PreparedImage(boxes=norm_boxes, vecs=vecs, whole_vec=whole_vec, exemplars=exemplars)

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if not self.supports_text:
            return None
        v = self._emb.embed_text(text)
        return None if v is None else np.asarray(v, dtype=np.float32)


class _WholeImageSource:
    """Single region = the full frame; positive exemplar = whole-image vector too."""

    def __init__(self, embedder, name: str = "whole") -> None:
        self._emb = embedder
        self.name = name
        self.supports_text = bool(embedder.supports_text)
        self.input_dim = _embedder_dim(embedder)

    def prepare(self, image: "Image.Image", gt_boxes: list[Box] | None = None) -> PreparedImage:
        whole_vec = np.asarray(self._emb._forward_pil_batch([image]), dtype=np.float32)[0]
        boxes = np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        # The "whole" method is image-level: a positive exemplar is the whole
        # positive image (boxes ignored), matching eval_crop_clip's `whole`.
        n = len(gt_boxes) if gt_boxes else 0
        exemplars = np.tile(whole_vec, (n, 1)) if n else np.zeros((0, self.input_dim), np.float32)
        return PreparedImage(boxes=boxes, vecs=whole_vec[None, :], whole_vec=whole_vec, exemplars=exemplars)

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if not self.supports_text:
            return None
        v = self._emb.embed_text(text)
        return None if v is None else np.asarray(v, dtype=np.float32)


class _PatchSource:
    """Patch-grid source: regions from the HAC tree or from pooled boxes.

    One patch forward per image; region/whole/exemplar vectors are all pooled
    from that grid, so they share the DINO patch space. ``proposer`` is ``None``
    for the HAC-tree variant (regions are tree nodes) or a box proposer for the
    box-pool variant (regions are those boxes pooled via ``box_to_vote_vector``).
    """

    def __init__(
        self,
        embedder,
        name: str,
        *,
        k: int = 12,
        alpha: float = 0.5,
        pca_dims: int | None = None,
        proposer: Proposer | None = None,
        region_voting: bool = False,
    ) -> None:
        self._emb = embedder
        self.name = name
        self._k = k
        self._alpha = alpha
        # Per-image PCA dims for the HAC merge order (tree topology only; stored
        # vecs stay full-dim). None = off. Inert for the box-pool variant, which
        # never builds a tree.
        self._pca_dims = pca_dims
        self._proposer = proposer
        # Region-voting label construction (matches the app detector's DINO-patch
        # path): a Good vote's box snaps to the nearest tree node, and negatives
        # flood only the childless nodes. Only meaningful for the HAC-tree variant
        # (``proposer is None``); ignored for the box-pool variant.
        self._region_voting = bool(region_voting) and proposer is None
        self.supports_text = False  # DINO patch embedders have no text encoder
        self.input_dim = _embedder_dim(embedder)

    def _patch_forward(self, image: "Image.Image"):
        return self._emb._patch_forward_pil_batch([image])[0]

    def prepare(self, image: "Image.Image", gt_boxes: list[Box] | None = None) -> PreparedImage:
        from vtscore.media.patch_embed import box_to_vote_vector, build_region_tree, snap_box_to_region

        pe = self._patch_forward(image)
        if pe is None:  # forward failed — degrade to a zero whole vector
            z = np.zeros(self.input_dim, np.float32)
            return PreparedImage(
                np.asarray([[0.0, 0.0, 1.0, 1.0]], np.float32),
                z[None, :],
                z,
                np.zeros((0, self.input_dim), np.float32),
                leaf_mask=np.ones(1, dtype=bool),
                children=np.full((1, 2), -1, dtype=int),
            )
        whole_vec = np.asarray(pe.cls_vec, dtype=np.float32)
        tree = None
        cell_masks: np.ndarray | None = None
        saliency: np.ndarray | None = None
        if self._proposer is None:
            tree = build_region_tree(pe, k=self._k, alpha=self._alpha, pca_dims=self._pca_dims)
            boxes = np.asarray([r.box for r in tree], dtype=np.float32)
            vecs = np.asarray([r.vec for r in tree], dtype=np.float32)
            # Childless nodes (CLS + HAC leaves) — the set a Bad vote floods.
            leaf_mask = np.asarray([r.children is None for r in tree], dtype=bool)
            # Child indices (into this same flat list) for redrawing the dendrogram;
            # (-1, -1) for childless nodes. Indices already align with ``boxes``.
            children = np.asarray([r.children if r.children is not None else (-1, -1) for r in tree], dtype=int)
            # Per-node patch-cell union masks + the attention grid, for the trace viz.
            cell_masks = _node_cell_masks(tree, np.asarray(pe.patch_saliency).shape)
            saliency = np.asarray(pe.patch_saliency, dtype=np.float32)
        else:
            boxes = self._proposer(image)
            vecs = np.asarray([box_to_vote_vector(pe.patch_grid, _as_box(b)) for b in boxes], dtype=np.float32)
            leaf_mask = np.ones(boxes.shape[0], dtype=bool)
            children = np.full((boxes.shape[0], 2), -1, dtype=int)  # flat pool: no hierarchy
        exemplars = np.zeros((0, self.input_dim), np.float32)
        if gt_boxes:
            if self._region_voting and tree is not None:
                # One positive per image: snap the covering box (union of all
                # instance boxes) to its best-IoU tree node — the exact candidate
                # the detector max-pools over — instead of a uniform grid pool.
                snapped = snap_box_to_region(tree, _covering_box(gt_boxes))
                if snapped is None:  # empty tree (shouldn't happen) — grid fallback
                    snapped = box_to_vote_vector(pe.patch_grid, _covering_box(gt_boxes))
                exemplars = np.asarray(snapped, dtype=np.float32)[None, :]
            else:
                exemplars = np.asarray(
                    [box_to_vote_vector(pe.patch_grid, _as_box(b)) for b in gt_boxes], dtype=np.float32
                )
        return PreparedImage(
            boxes=boxes,
            vecs=vecs,
            whole_vec=whole_vec,
            exemplars=exemplars,
            leaf_mask=leaf_mask,
            children=children,
            cell_masks=cell_masks,
            saliency=saliency,
        )

    def embed_text(self, text: str) -> Optional[np.ndarray]:  # noqa: ARG002
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _embedder_dim(embedder) -> int:
    """Embedding dimensionality, determined by a one-shot probe forward.

    Config attributes are unreliable across embedders (SigLIP nests hidden sizes
    under text/vision sub-configs), so we embed a tiny dummy image and read the
    vector length — this matches exactly the space the region vectors live in.
    """
    from PIL import Image  # noqa: PLC0415

    embedder.load_models()
    vec = np.asarray(embedder._forward_pil_batch([Image.new("RGB", (32, 32))]), dtype=np.float32)
    if vec.ndim == 2 and vec.shape[1] > 0:
        return int(vec.shape[1])
    raise ValueError(f"cannot determine embedding dim for {getattr(embedder, 'name', embedder)!r}")


def build_region_source(
    proposal: str,
    embedder,
    *,
    scales: tuple[float, ...] = (1.0, 0.5, 0.25),
    overlap: float = 0.5,
    min_window: int = 64,
    hac_k: int = 12,
    hac_alpha: float = 0.5,
    hac_pca_dims: int | None = None,
    dino_model_id: str | None = None,
    dino_device: str = "cpu",
    dino_register_tokens: int = 0,
    region_voting: bool = False,
) -> RegionSource:
    """Build a region source for ``proposal`` bound to ``embedder``.

    Enforces embedding-space consistency: every source embeds/pools its
    regions, whole-image, and exemplars with the *same* ``embedder`` (patch
    sources additionally require ``supports_patch_regions``). ``dino`` uses a
    separate DINO model only to *propose* boxes; the boxes are re-embedded by
    ``embedder``, so the vectors stay in ``embedder``'s space.
    """
    embedder.load_models()
    if proposal == "whole":
        return _WholeImageSource(embedder)
    if proposal == "sliding":
        prop = SlidingWindowProposer(scales=scales, overlap=overlap, min_window=min_window)
        return _CropReembedSource(embedder, prop, name="sliding")
    if proposal == "dino":
        if dino_model_id is None:
            raise ValueError("proposal='dino' requires dino_model_id")
        prop = DinoRegionProposer(dino_model_id, dino_device, dino_register_tokens, k=hac_k, alpha=hac_alpha)
        return _CropReembedSource(embedder, prop, name="dino")
    if proposal == "hac":
        if not embedder.supports_patch_regions:
            raise ValueError(f"proposal='hac' needs a patch embedder; {embedder.name!r} has no patch grid")
        return _PatchSource(
            embedder,
            name="hac",
            k=hac_k,
            alpha=hac_alpha,
            pca_dims=hac_pca_dims,
            proposer=None,
            region_voting=region_voting,
        )
    if proposal == "hac_boxpool":
        if not embedder.supports_patch_regions:
            raise ValueError(f"proposal='hac_boxpool' needs a patch embedder; {embedder.name!r} has no patch grid")
        prop = SlidingWindowProposer(scales=scales, overlap=overlap, min_window=min_window)
        return _PatchSource(embedder, name="hac_boxpool", k=hac_k, alpha=hac_alpha, proposer=prop)
    raise ValueError(f"unknown proposal {proposal!r}")
