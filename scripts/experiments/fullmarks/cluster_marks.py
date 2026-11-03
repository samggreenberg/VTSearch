"""Turn located marks into identity classes — and be honest that it's a guess.

SPODS and StaVer both ship *where* every logo and stamp is, and neither ships
*which* one it is.  A previous VTSearch study reported "64 logo/stamp classes"
for SPODS with names like ``logo_14``; those identities were derived, not read
off the dataset, and nothing verified them.  Since class identity is the entire
ground truth of an instance-retrieval benchmark, a derived clustering that
nobody checked is not a benchmark — it is a hypothesis with error bars nobody
measured.

So this module does three things in order:

1. crop every mark and describe it (``phash`` by default, ``siglip`` when the
   pile's models are available);
2. single-linkage agglomerate under a distance threshold into candidate classes;
3. emit the material a human needs to confirm or correct the result, and mark
   every derived ``class_id`` with ``provenance="clustered"`` so downstream code
   can tell a verified identity from a guessed one.

Single linkage is chosen deliberately over a centroid method: instances of one
stamp vary continuously with ink coverage and scan quality, so a chain of near
neighbours is the right shape, and the failure mode it does have — two classes
bridged by one ambiguous crop — is exactly what a human reviewing the largest
clusters will spot immediately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional, Sequence, Union

import numpy as np

from sources._common import Mark, Page

#: Side length the crop is normalised to before hashing.
PHASH_RESIZE = 64

#: Low-frequency DCT block kept.  16x16 coefficients -> a 256-bit hash.
#:
#: This was 8x8 (64 bits), and 64 bits could not tell two round stamps apart.
#: Measured on the real corpus: a red book stamp and a blue elephant stamp,
#: both circular with a heavy ring border, landed in one class of 32 and no
#: threshold separated them — a single pair at Hamming 2/64 bridged the set, so
#: it was one group at 0.04 and 21 fragments at 0.03.
#:
#: The mechanism is a frequency argument.  A stamp's border ring is a big,
#: smooth, *low*-frequency structure, and its interior — the part that says
#: which stamp this is — is higher-frequency detail.  An 8x8 block keeps almost
#: nothing but the ring, so the descriptor encodes "is a round stamp" rather
#: than "which round stamp".  Keeping 16x16 pushes the balance the other way.
PHASH_BLOCK = 16

#: Fraction of the crop's half-diagonal over which the radial taper acts.  The
#: border is also the part of a mark that varies least between *different*
#: marks, so damping it costs little discrimination and removes a lot of shared
#: signal.  Applied on top of the bigger DCT block, not instead of it.
PHASH_TAPER = 0.22


@dataclass(frozen=True)
class MarkRef:
    """A pointer to one mark inside the page list, plus its crop descriptor."""

    page_index: int
    mark_index: int
    page_id: str
    kind: str
    box: tuple[int, int, int, int]


#: One side of an adjudicated pair: a ``(page_id, mark_index)`` mark, or a bare
#: page id for the case — now only a legacy one — where the page carries a
#: single clustered mark and the index adds nothing.  See ``resolve_pairs``.
Endpoint = Union[str, tuple[str, int]]


class Side(NamedTuple):
    """One side of a stored separation that also records its class's members.

    *anchor* is the mark the row names, the same endpoint a bare side would
    be.  *members* are every ``(page_id, mark_index)`` the class held when the
    verdict was given.  ``apply_confusable`` records them for a class that has
    not been through the membership pass.  Such a class is not one must-linked
    group yet, so the anchor alone would hold apart one mark and not the class
    (#3837).  ``load_adjudications`` returns a bare endpoint for a side without
    members, so a store with none of these reads exactly as it did before.
    """

    anchor: Endpoint
    members: tuple[tuple[str, int], ...]


#: One side of a loaded separation: a bare endpoint, or a `Side` carrying the
#: class it stood for.
SeparationSide = Union[Endpoint, Side]


@dataclass(frozen=True)
class Separation:
    """Two sets of rows ruled to be different marks, resolved for one clustering.

    ``left`` and ``right`` are the rows the verdict names, and they are
    **hard**.  If a must-link puts both in one group, the store contradicts
    itself and the clustering refuses, the same as it does for a single pair.

    ``left_members`` and ``right_members`` are the rest of each class as it
    stood when the verdict was given, and they are **soft**.  A membership
    snapshot is a presumption about marks nobody has looked at one by one, and
    later passes do look.  A membership rejection or a completeness accept can
    move a mark out of the class it was presumed to be in.  So where a human
    must-link puts a soft member in a group with the other side, the human
    verdict wins and the member's tag is dropped for that group.  Against a
    *distance* the soft members hold, which is the point: the threshold is the
    thing the verdict exists to overrule.
    """

    left: tuple[int, ...]
    right: tuple[int, ...]
    left_members: tuple[int, ...] = ()
    right_members: tuple[int, ...] = ()


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------


def _dct2(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II via matrix multiply.

    Written out rather than pulled from scipy so the clustering pass has the
    same dependency footprint as the rest of the builder (numpy + Pillow) and
    runs identically on a login node and a compute node.
    """
    n = block.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    basis[0, :] = basis[0, :] / np.sqrt(2)
    return basis @ block @ basis.T


def _radial_taper(arr: np.ndarray, width: float = PHASH_TAPER) -> np.ndarray:
    """Fade the crop's outer annulus toward its own mean.

    A stamp's border ring is the single strongest thing in the image and the
    *least* informative: near enough every round stamp has one, so it is shared
    signal that crowds out the interior.  Fading toward the mean rather than to
    white avoids replacing one strong edge with another.

    The taper is radial because the marks that need separating are round, and
    it is soft because a hard mask would itself be a circular edge — the exact
    artefact being removed.
    """
    n = arr.shape[0]
    axis = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    radius = np.hypot(*np.meshgrid(axis, axis, indexing="ij"))
    # 1 in the middle, falling smoothly to 0 at the corners of the inscribed
    # circle and beyond.
    weight = np.clip((1.0 - radius) / max(width, 1e-6), 0.0, 1.0)
    weight = weight * weight * (3 - 2 * weight)  # smoothstep
    return arr.mean() + (arr - arr.mean()) * weight


def phash(image: Any) -> np.ndarray:
    """A perceptual hash of *image* as a boolean vector, `PHASH_BLOCK`² bits.

    Scale-invariant by construction (everything is resized to a fixed square),
    which is what we want: the same stamp appears at whatever size the scanner
    and the page layout produced.  Aspect ratio is deliberately discarded here
    and reintroduced as a separate gate in :func:`distance_matrix`, because two
    marks with very different aspect ratios are not the same mark however
    similar their normalised pixels look.

    Greyscale on purpose, so ink colour never splits a class.  Confirmed on the
    real corpus, where the same elephant stamp appears in blue on 26 pages and
    red on one, and lands in a single group.
    """
    grey = image.convert("L").resize((PHASH_RESIZE, PHASH_RESIZE))
    arr = _radial_taper(np.asarray(grey, dtype=np.float64))
    coeffs = _dct2(arr)[:PHASH_BLOCK, :PHASH_BLOCK]
    flat = coeffs.flatten()
    # Drop the DC term before taking the median: it encodes overall brightness,
    # which on a scan is the paper, not the mark.
    median = np.median(flat[1:])
    return flat > median


def crop_mark(page_image: Any, box: tuple[int, int, int, int], *, pad_frac: float = 0.04) -> Any:
    """Crop *box* out of *page_image* with a little context padding."""
    x, y, w, h = box
    pad = int(round(max(w, h) * pad_frac))
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(page_image.width, x + w + pad)
    bottom = min(page_image.height, y + h + pad)
    return page_image.crop((left, top, right, bottom))


def describe_marks(
    pages: Sequence[Page],
    refs: Sequence[MarkRef],
    *,
    backend: str = "phash",
) -> np.ndarray:
    """Descriptor matrix for *refs*, one row per mark.

    ``phash`` returns a boolean matrix compared by normalised Hamming distance;
    ``siglip`` returns L2-normalised float vectors compared by cosine distance.
    """
    from PIL import Image

    if backend == "phash":
        rows = []
        cache: dict[str, Any] = {}
        for ref in refs:
            page = pages[ref.page_index]
            if page.path not in cache:
                cache.clear()  # one page open at a time; pages are large
                cache[page.path] = Image.open(page.path).convert("L")
            rows.append(phash(crop_mark(cache[page.path], ref.box)))
        return np.array(rows, dtype=bool)

    if backend == "siglip":
        from vtscore.media.image.embedders.embedder_siglip import ImageSiglipEmbedder  # noqa: PLC0415

        embedder = ImageSiglipEmbedder()
        crops = []
        cache_path: Optional[str] = None
        cache_img: Any = None
        for ref in refs:
            page = pages[ref.page_index]
            if page.path != cache_path:
                cache_path, cache_img = page.path, Image.open(page.path).convert("RGB")
            crops.append(crop_mark(cache_img, ref.box))
        # `embed_pil_image` is the embedder's only in-memory entry point, and it
        # returns None for a crop the model declines.  A zero row keeps the
        # matrix aligned with *refs* -- dropping the row would silently shift
        # every later label onto the wrong mark -- and normalises to a vector at
        # distance 1.0 from everything, so that mark simply clusters alone.
        rows = [embedder.embed_pil_image(crop) for crop in crops]
        dim = embedder.embedding_dim
        vecs = np.asarray([row if row is not None else np.zeros(dim) for row in rows], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-8, None)

    raise ValueError(f"unknown cluster backend {backend!r} (expected 'phash' or 'siglip')")


def distance_matrix(
    desc: np.ndarray,
    refs: Sequence[MarkRef],
    *,
    backend: str = "phash",
    max_aspect_ratio: float = 2.0,
) -> np.ndarray:
    """Pairwise distances, with mismatched aspect ratios forced apart.

    The aspect gate is what stops a round rubber stamp and a wide letterhead
    banner from merging just because both are dark ink on white paper at 32x32.
    """
    if backend == "phash":
        bits = desc.astype(np.uint8)
        # Hamming distance via matrix algebra: |a XOR b| = a.(1-b) + (1-a).b
        inv = 1 - bits
        dist = (bits @ inv.T + inv @ bits.T).astype(np.float64) / desc.shape[1]
    else:
        dist = 1.0 - (desc @ desc.T)

    aspects = np.array([max(b.box[2], 1) / max(b.box[3], 1) for b in refs], dtype=np.float64)
    ratio = np.maximum(aspects[:, None] / aspects[None, :], aspects[None, :] / aspects[:, None])
    dist = np.where(ratio > max_aspect_ratio, 1.0, dist)

    np.fill_diagonal(dist, 0.0)
    return dist


# --------------------------------------------------------------------------
# Representative instances
# --------------------------------------------------------------------------


def medoid_core(dist: np.ndarray, *, spread: float) -> tuple[int, list[int]]:
    """The medoid row of a group, and the rows close enough to speak for it.

    ``dist`` is the group's own square distance matrix — one class's instances,
    not the whole corpus.  Returns ``(medoid, core)`` where *core* always holds
    the medoid and is sorted ascending.

    The cut is **relative to the group's own spread**, and that is the entire
    point rather than an implementation detail.  Comparing a descriptor distance
    against a fixed bar does not work on document marks and has been measured
    not working: on the live corpus the one confirmed-wrong exemplar sat at
    0.172 from its own class where the 60-class median was 0.28 — second
    *lowest*, i.e. the healthiest-looking distance of nearly all of them (#3599).
    A perceptual hash of blue ink on white paper tracks ink layout, so its
    absolute value says how busy a stamp is and not which stamp it is.  What the
    same number still supports is a comparison *within* one group, where paper,
    scanner, ink and layout are held fixed: an instance that sits three of its
    own class's median-absolute-deviations past its own class's median distance
    is an outlier on a scale the class itself set.

    The scale is a median-absolute-deviation rather than a standard deviation
    because the thing being screened for — a handful of instances of some other
    mark — is exactly what drags a mean and a variance out until they cover it.
    MAD tolerates contamination up to half the group; past that there is no
    majority mark to find, and the caller is expected to say so rather than pick
    one.  It is rescaled by the usual 1.4826 so that *spread* is read in
    standard deviations of the group's own distances, robustly estimated.
    """
    n = int(dist.shape[0])
    if n == 0:
        raise ValueError("medoid_core needs at least one row")
    if n == 1:
        return 0, [0]

    # Mean distance to every *other* row; the smallest is the medoid.  argmin
    # takes the lowest index on a tie, so the result is a pure function of the
    # matrix and not of iteration order.
    medoid = int(np.argmin(dist.sum(axis=1) / (n - 1)))
    others = np.delete(dist[medoid], medoid)
    median = float(np.median(others))
    mad = float(np.median(np.abs(others - median)))
    cutoff = median + spread * 1.4826 * mad
    return medoid, [i for i in range(n) if i == medoid or dist[medoid, i] <= cutoff]


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def single_linkage(
    dist: np.ndarray,
    threshold: float,
    *,
    cannot_link: Optional[Sequence[tuple[int, int]]] = None,
    must_link: Optional[Sequence[tuple[int, int]]] = None,
    separations: Optional[Sequence[Separation]] = None,
) -> list[int]:
    """Union-find single-linkage clustering.  Returns a label per row.

    Two kinds of human decision override the distances, and both are honoured
    before the threshold gets a say:

    * *must_link* — pairs adjudicated as the **same mark**.  Joined whatever
      their distance, because a clustering that splits one mark in two is a
      thing a person can see and the descriptor cannot.
    * *cannot_link* and *separations* — marks adjudicated as **different**.
      Never joined, however similar they look, and the constraint propagates:
      once ``a`` and ``b`` are apart, nothing may reunite their groups through a
      third crop.  A *cannot_link* pair is the one-row-a-side case of a
      `Separation`; a separation between two *classes* names every row of each,
      because naming one per side holds apart those two rows and no more.
      Merges run in increasing distance order, so a cheaper crossing edge
      elsewhere fuses the classes first and leaves the two named rows as the
      only ones kept apart (#3837).

    Each group carries, per separation, which side of it the group is on (a
    bit for left, a bit for right), and a union is refused when the two groups
    together would hold both sides.  That is the same rule as before for a pair.
    It is also exactly the rule for a class, and it costs a dict merge per
    union, not a rewrite of every constraint.

    The asymmetry between the two kinds is the whole operating strategy.  Run the
    threshold **strict**, so the partition over-splits: a split is one visible
    mistake that a person fixes with one merge, while a bad merge is invisible
    contamination that quietly makes a class mean two things.  These lists
    are how that repair survives — they are replayed on every re-cluster, so
    the work is done once rather than re-done whenever a number moves.

    Threshold merges are applied in increasing distance order, so the result
    does not depend on iteration order once constraints start blocking merges.
    """
    n = dist.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Human merges first, so a separation is checked against the groups the
    # person actually meant rather than against raw rows.
    for a, b in must_link or ():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    rules = [Separation((a,), (b,)) for a, b in cannot_link or ()] + list(separations or ())
    left_bit, right_bit = 1, 2
    both = left_bit | right_bit

    for rule in rules:
        clash = [(a, b) for a in rule.left for b in rule.right if find(a) == find(b)]
        if clash:
            a, b = clash[0]
            raise ValueError(
                f"rows {a} and {b} are adjudicated both same and different — "
                "one of the two verdicts is wrong, and guessing which would "
                "silently pick a side"
            )

    # Per group, per rule: which side(s) the group is on.  Named rows first;
    # a member only tags a group the named rows have not already placed, and a
    # group holding members of both sides — two presumptions a human must-link
    # has just contradicted — keeps neither.
    hard: dict[int, dict[int, int]] = {}
    soft: dict[int, dict[int, int]] = {}
    for index, rule in enumerate(rules):
        for rows, bit, store in (
            (rule.left, left_bit, hard),
            (rule.right, right_bit, hard),
            (rule.left_members, left_bit, soft),
            (rule.right_members, right_bit, soft),
        ):
            for row in rows:
                tags = store.setdefault(find(row), {})
                tags[index] = tags.get(index, 0) | bit
    sides: dict[int, dict[int, int]] = {}
    for root in sorted(set(hard) | set(soft)):
        named_on, presumed_on = hard.get(root, {}), soft.get(root, {})
        tags = {}
        for index in set(named_on) | set(presumed_on):
            bits = named_on.get(index) or presumed_on.get(index, 0)
            if bits != both:
                tags[index] = bits
        if tags:
            sides[root] = tags

    def blocked(a: dict[int, int], b: dict[int, int]) -> bool:
        if len(a) > len(b):
            a, b = b, a
        return any(bits | b.get(index, 0) == both for index, bits in a.items())

    candidates = sorted(
        ((dist[i, j], i, j) for i in range(n) for j in range(i + 1, n) if dist[i, j] <= threshold),
        key=lambda t: (t[0], t[1], t[2]),
    )
    for _d, i, j in candidates:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        ti, tj = sides.get(ri), sides.get(rj)
        if ti and tj and blocked(ti, tj):
            continue
        keep, gone = min(ri, rj), max(ri, rj)
        parent[gone] = keep
        # Fold the absorbed group's sides into the survivor's, smaller into
        # larger, so a long chain of unions does not copy one big dict each time.
        absorbed = sides.pop(gone, None)
        if absorbed:
            kept = sides.get(keep)
            if kept is None:
                sides[keep] = absorbed
            else:
                if len(kept) < len(absorbed):
                    kept, absorbed = absorbed, kept
                for index, bits in absorbed.items():
                    kept[index] = kept.get(index, 0) | bits
                sides[keep] = kept

    # Relabel roots to dense 0..k-1 in first-appearance order, so the labelling
    # is a pure function of the distance matrix and not of dict iteration.
    remap: dict[int, int] = {}
    labels = []
    for i in range(n):
        root = find(i)
        if root not in remap:
            remap[root] = len(remap)
        labels.append(remap[root])
    return labels


def assign_class_ids(
    pages: list[Page],
    refs: Sequence[MarkRef],
    labels: Sequence[int],
    *,
    source: str,
    provenance: str = "clustered",
) -> dict[str, list[MarkRef]]:
    """Write clustered ``class_id``\\ s back onto the pages' marks.

    Class ids are derived from the cluster's *smallest page id*, not from its
    index, so adding pages to the corpus cannot silently renumber existing
    classes and invalidate a previous run's audit verdicts.
    """
    members: dict[int, list[MarkRef]] = {}
    for ref, label in zip(refs, labels):
        members.setdefault(label, []).append(ref)

    out: dict[str, list[MarkRef]] = {}
    for label, group in members.items():
        anchor = min(group, key=lambda r: (r.page_id, r.mark_index))
        kind = group[0].kind
        class_id = f"{source}/{kind}_{anchor.page_id.split('/')[-1]}_{anchor.mark_index}"
        out[class_id] = group
        for ref in group:
            mark = pages[ref.page_index].marks[ref.mark_index]
            pages[ref.page_index].marks[ref.mark_index] = Mark(
                kind=mark.kind,
                box=mark.box,
                class_id=class_id,
                provenance=provenance,
            )
    return out


def resolve_pairs(
    refs: Sequence[MarkRef],
    pairs: Sequence[tuple[Endpoint, Endpoint]],
) -> list[tuple[int, int]]:
    """Turn adjudicated endpoint pairs into row-index pairs.

    Adjudications are stored against **page ids**, not row indices or class
    ids, because those are the only identifiers that survive a re-cluster.  A
    row index changes whenever the corpus grows; a class id changes whenever
    the threshold moves — and a human decision about two marks must outlive
    both, or every re-run quietly discards the annotation it was supposed to be
    built on.

    A page id alone is only an identifier for the *mark* when the page carries
    one, and on every source here that is false: a SPODS page carries a logo, a
    stamp and a signature, a StaVer page carries two stamps.  A bare page id
    then resolves to *every* pair of marks across the two pages, which is not
    what anybody adjudicated — and in the ``same`` direction it is actively
    destructive.  The real corpus carried one such pair, the ``DY.Secretary``
    merge (``spods/00546`` ↔ ``spods/00551``, two 30-instance logo classes and
    one stamp class between them): replaying it would have must-linked all four
    crossings and fused three classes into one 76-instance blob, silently, on
    the next build (#3343).

    So an endpoint is either a page id — accepted only while it is unambiguous
    on this source's marks — or an explicit ``(page_id, mark_index)``, which is
    what everything writes now.  ``mark_index`` survives a re-cluster for the
    same reason a page id does: it indexes the page's own decomposition and
    owes nothing to the clustering.  An ambiguous bare endpoint is **refused**
    rather than guessed at, because guessing silently picks a side.

    A pair naming a page that is no longer in the corpus is skipped rather than
    raising: pages come and go with tier budgets, and a dropped page is not a
    reason to refuse to build.
    """
    endpoints = _EndpointRows(refs)
    out: list[tuple[int, int]] = []
    for left, right in pairs:
        for a in endpoints.rows(left):
            for b in endpoints.rows(right):
                if a != b:
                    out.append((a, b))
    endpoints.refuse_ambiguous()
    return out


def resolve_separations(
    refs: Sequence[MarkRef],
    pairs: Sequence[tuple[SeparationSide, SeparationSide]],
) -> list[Separation]:
    """Turn stored separations into `Separation`\\ s over row indices.

    A bare side resolves exactly as in :func:`resolve_pairs`, to the one row it
    names.  A `Side` resolves its anchor the same way, plus its recorded
    members as the soft rows of that side.  A member that is not a row here
    (another source's page, a page outside this tier, a mark already carrying
    an identity) is skipped, as a dropped page is.  So is a whole separation one
    of whose sides has no row here at all: there is nothing on that side for
    the constraint to keep anything away from.
    """
    endpoints = _EndpointRows(refs)
    out: list[Separation] = []
    for left, right in pairs:
        resolved = []
        for side in (left, right):
            if isinstance(side, Side):
                named = tuple(endpoints.rows(side.anchor))
                members = {row for member in side.members for row in endpoints.rows(member)}
                resolved.append((named, tuple(sorted(members - set(named)))))
            else:
                resolved.append((tuple(endpoints.rows(side)), ()))
        (left_named, left_members), (right_named, right_members) = resolved
        if not (left_named or left_members) or not (right_named or right_members):
            continue
        if left_named and left_named == right_named:
            continue  # one mark on both sides; `resolve_pairs` drops this too
        out.append(Separation(left_named, right_named, left_members, right_members))
    endpoints.refuse_ambiguous()
    return out


class _EndpointRows:
    """Rows of *refs* by endpoint, collecting every ambiguous bare page id."""

    def __init__(self, refs: Sequence[MarkRef]) -> None:
        self.refs = refs
        self.by_page: dict[str, list[int]] = {}
        self.by_mark: dict[tuple[str, int], int] = {}
        for index, ref in enumerate(refs):
            self.by_page.setdefault(ref.page_id, []).append(index)
            self.by_mark[(ref.page_id, ref.mark_index)] = index
        self.ambiguous: list[str] = []

    def rows(self, endpoint: Endpoint) -> list[int]:
        if isinstance(endpoint, Side):
            # Reading only the anchor would silently turn a class-level
            # separation back into the two-mark kind (#3837).
            raise TypeError("a separation side that carries members resolves through resolve_separations")
        if isinstance(endpoint, str):
            rows = self.by_page.get(endpoint, [])
            if len(rows) > 1:
                self.ambiguous.append(
                    f"{endpoint} carries {len(rows)} clustered marks "
                    f"({', '.join(str(self.refs[r].mark_index) for r in rows)})"
                )
                return []
            return rows
        row = self.by_mark.get((endpoint[0], endpoint[1]))
        return [row] if row is not None else []

    def refuse_ambiguous(self) -> None:
        if self.ambiguous:
            raise ValueError(
                f"{len(self.ambiguous)} adjudication endpoint(s) name a page carrying more than one "
                "clustered mark, so which mark was ruled on is not recorded: "
                f"{'; '.join(sorted(set(self.ambiguous))[:5])}. "
                "Re-record them with a mark index — `audit_to_corrections.py --migrate-adjudications "
                "--corpus <dir>` does it from the class ids the rows already carry."
            )


def collect_refs(pages: Sequence[Page], *, kinds: Iterable[str], source: str) -> list[MarkRef]:
    """Every unlabelled mark of the given *kinds* from *source*'s pages."""
    wanted = set(kinds)
    refs = []
    for pi, page in enumerate(pages):
        if page.source != source:
            continue
        for mi, mark in enumerate(page.marks):
            if mark.kind in wanted and mark.class_id is None:
                refs.append(MarkRef(pi, mi, page.page_id, mark.kind, mark.box))
    return refs


def cluster_source(
    pages: list[Page],
    source: str,
    *,
    kinds: Iterable[str] = ("logo", "stamp"),
    backend: str = "phash",
    threshold: float = 0.18,
    same: Optional[Sequence[tuple[Endpoint, Endpoint]]] = None,
    different: Optional[Sequence[tuple[SeparationSide, SeparationSide]]] = None,
    provenance: str = "clustered",
) -> dict[str, Any]:
    """Cluster one source's marks in place.  Returns a summary for the report.

    *same* and *different* are adjudicated pairs, as `load_adjudications`
    returns them.  They survive re-clustering: a human decision about two marks
    must not be undone by someone later nudging the threshold.

    ``separations_on_members`` counts the separations that carry a recorded
    class membership (#3837), of ``separations_applied`` in all.
    """
    refs = collect_refs(pages, kinds=kinds, source=source)
    if not refs:
        return {"source": source, "marks": 0, "classes": 0, "backend": backend, "threshold": threshold}

    desc = describe_marks(pages, refs, backend=backend)
    dist = distance_matrix(desc, refs, backend=backend)
    must_link = resolve_pairs(refs, same or ())
    separations = resolve_separations(refs, different or ())
    labels = single_linkage(dist, threshold, must_link=must_link, separations=separations)
    classes = assign_class_ids(pages, refs, labels, source=source, provenance=provenance)

    sizes = sorted((len(v) for v in classes.values()), reverse=True)
    return {
        "source": source,
        "marks": len(refs),
        "classes": len(classes),
        "backend": backend,
        "threshold": threshold,
        "merges_applied": len(must_link),
        "separations_applied": len(separations),
        "separations_on_members": sum(1 for s in separations if s.left_members or s.right_members),
        "largest_clusters": sizes[:10],
        "singletons": sum(1 for s in sizes if s == 1),
    }


def write_cluster_report(summaries: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Adjudicated same/different decisions
# --------------------------------------------------------------------------
#
# The corpus stores *both* directions of the ground truth, because a benchmark
# needs both: same-mark pairs say what the detector must find, different-mark
# pairs say what it must keep apart.  Clustering can only ever propose the
# first; the second has to be adjudicated and then enforced, which is what
# the separations in single_linkage do with these.


def load_adjudications(
    path: Path,
) -> tuple[list[tuple[Endpoint, Endpoint]], list[tuple[SeparationSide, SeparationSide]]]:
    """``(same, different)`` mark pairs a human has ruled on.

    Both directions live in one file because they are one decision procedure
    seen from two sides, and because a pair appearing in both is a conflict
    that has to be catchable — which it is not if they sit in separate stores
    that nothing reads together.

    An endpoint carrying ``left_mark_index`` / ``right_mark_index`` comes back
    as ``(page_id, mark_index)``; one without comes back as the bare page id it
    was written as, and ``resolve_pairs`` decides whether that is still
    unambiguous.  Reading a legacy row as a mark would be inventing the index.

    A ``different`` side that records its class's members
    (``left_members`` / ``right_members``) comes back as a `Side`, and has to go
    through `resolve_separations`; reading only its anchor would turn a
    separation between two classes back into one between two marks (#3837).
    """
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))

    def endpoint(row: dict[str, Any], side: str) -> Endpoint:
        index = row.get(f"{side}_mark_index")
        page_id = row[f"{side}_page_id"]
        return page_id if index is None else (page_id, int(index))

    def separation_side(row: dict[str, Any], side: str) -> SeparationSide:
        members = row.get(f"{side}_members")
        if not members:
            return endpoint(row, side)
        return Side(endpoint(row, side), tuple((str(page_id), int(index)) for page_id, index in members))

    same = [(endpoint(r, "left"), endpoint(r, "right")) for r in payload.get("same", [])]
    different = [(separation_side(r, "left"), separation_side(r, "right")) for r in payload.get("different", [])]
    return same, different


def load_adjudication_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The ``(same, different)`` rows verbatim, for anyone about to rewrite them.

    ``load_adjudications`` narrows each row to the pair the clusterer needs;
    reading that and writing it back is lossy — it drops the class ids and the
    reviewer's note, which are the only record of *why* a pair was ruled on.
    """
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("same", [])), list(payload.get("different", []))


#: The two marks a stored row rules on, each ``(page_id, mark_index)``, sorted.
#: A bare page id keys as mark ``-1``.
PairKey = tuple[tuple[str, int], tuple[str, int]]


def coinciding(left: Iterable[PairKey], right: Iterable[PairKey]) -> list[tuple[PairKey, PairKey]]:
    """Every ``(a, b)``, *a* from *left* and *b* from *right*, that may name the same two marks.

    Equal keys do.  So do keys that differ only where one of them is a bare
    page id: a bare endpoint names whichever mark `resolve_pairs` finds on its
    page, and that can be the very mark the other key names.  Comparing the
    ``-1`` as if it were a mark of its own is how the store came to hold the
    owner's 2026-09-14 merge next to the separation it overturned, after a
    ``--supersede`` that should have dropped it (#4866).  Neither check could
    see that ``(bad45f00, -1)`` and ``(bad45f00, 1)`` were one mark, and every
    Tobacco800 rebuild refused.
    """

    def fits(x: tuple[str, int], y: tuple[str, int]) -> bool:
        return x[0] == y[0] and (x[1] == y[1] or -1 in (x[1], y[1]))

    def pages(key: PairKey) -> tuple[str, str]:
        return key[0][0], key[1][0]

    by_pages: dict[tuple[str, str], list[PairKey]] = {}
    for key in right:
        by_pages.setdefault(pages(key), []).append(key)
    out: list[tuple[PairKey, PairKey]] = []
    for a in left:
        for b in by_pages.get(pages(a), ()):
            if (fits(a[0], b[0]) and fits(a[1], b[1])) or (fits(a[0], b[1]) and fits(a[1], b[0])):
                out.append((a, b))
    return sorted(out)


def save_adjudications(
    same: Sequence[dict[str, Any]],
    different: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    """Write both directions, deduplicated and ordered, refusing a conflict.

    Pairs are stored unordered-within-pair (sorted) so that ruling on
    ``(a, b)`` and later ``(b, a)`` records one decision rather than two.

    The key is the pair of *marks*, not of pages: one page can carry a logo
    that is the same as another page's and a stamp that is not, so keying on
    pages would let the second verdict overwrite the first (#3343).  A row with
    no mark index keys on ``-1``, which keeps a legacy file idempotent.  That
    key is not a mark, though, so the clash check compares it with
    `coinciding`: a bare endpoint ruled one way and a real mark on the same
    page ruled the other may be one pair ruled both ways, and the clusterer
    would refuse the store (#4866).  That is refused too, until
    ``--migrate-adjudications`` gives the bare endpoint its index.
    """

    def key_of(row: dict[str, Any], side: str) -> tuple[str, int]:
        index = row.get(f"{side}_mark_index")
        return (row[f"{side}_page_id"], -1 if index is None else int(index))

    def flip(key: str) -> str:
        """``left_class_id`` <-> ``right_class_id``, and so on."""
        if key.startswith("left_"):
            return "right_" + key[len("left_") :]
        if key.startswith("right_"):
            return "left_" + key[len("right_") :]
        return key

    def canon(rows: Sequence[dict[str, Any]]) -> dict[tuple[tuple[str, int], tuple[str, int]], dict[str, Any]]:
        out: dict[tuple[tuple[str, int], tuple[str, int]], dict[str, Any]] = {}
        for row in rows:
            left, right = key_of(row, "left"), key_of(row, "right")
            merged = dict(row)
            if right < left:
                # Every ``left_*`` field moves with its side, not just the page
                # id: swapping the ids alone would leave ``left_class_id``
                # naming the page now on the right.
                left, right = right, left
                merged = {flip(k): v for k, v in merged.items()}
            out[(left, right)] = merged
        return out

    same_map, diff_map = canon(same), canon(different)
    clash = sorted(set(same_map) & set(diff_map))
    if clash:
        raise ValueError(
            f"{len(clash)} pair(s) adjudicated both same and different, e.g. {clash[0]} — "
            "resolve the verdicts before writing; storing both would let whichever "
            "is applied last silently win"
        )
    maybe = coinciding(same_map, diff_map)
    if maybe:
        a, b = maybe[0]
        raise ValueError(
            f"{len(maybe)} pair(s) may be adjudicated both same and different, e.g. same {a} and different {b}: "
            "a bare page id (mark -1) names whichever mark the clusterer finds on that page, which can be the "
            "mark the other ruling names. Give it its index first — `audit_to_corrections.py "
            "--migrate-adjudications --corpus <dir>`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "same": [same_map[k] for k in sorted(same_map)],
        "different": [diff_map[k] for k in sorted(diff_map)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
