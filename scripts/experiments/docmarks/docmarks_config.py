"""Configuration for the DocMarks corpus — stamps and logos in scanned documents.

DocMarks is one corpus assembled from several sources, in three strata:

* **anchor** — real ground truth from SPODS, StaVer and Tobacco800.  These carry
  per-mark boxes; SPODS and StaVer need an identity-clustering pass on top (see
  :mod:`cluster_marks`) because neither ships instance labels.
* **haystack** — real scanned pages with no marks of interest, pulled from the
  UCSF Industry Documents Library.  Distractors, plus (optionally) weakly
  labelled letterhead classes keyed on the document's ``author`` field.
* **synth** — real mark artwork pasted onto held-out real scans at known
  ``(x, y, scale, rotation)``.  Instance ground truth by construction; used for
  statistical power, never quoted on its own.

The corpus is emitted as one manifest with **nested tiers**, so a 5k experiment
and a 200k experiment read the same file and the same class ids.

Every knob here is overridable by environment variable so a GRID job can be
re-pointed without editing the tree.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

#: Where source archives are downloaded and unpacked.  Big; keep it off the
#: 50G mount (see scripts/experiments/GRID-PLAYBOOK.md).
RAW = Path(os.environ.get("VTS_DOCMARKS_RAW", "/expscratch/{u}/docmarks/raw".format(u=os.environ.get("USER", "user"))))

#: Where the assembled corpus lands: ``images/``, ``queries/``, ``corpus.jsonl``,
#: ``classes.json``.  This is what a study reads and what the pile embeds.
OUT = Path(
    os.environ.get("VTS_DOCMARKS_OUT", "/expscratch/{u}/docmarks/corpus".format(u=os.environ.get("USER", "user")))
)

# --------------------------------------------------------------------------
# Tiers — nested by construction
# --------------------------------------------------------------------------

#: ``tier -> total page budget``.  Tiers are *nested*: every page in ``s`` is in
#: ``m``, every page in ``m`` is in ``l``.  Positives are in every tier (a class
#: with 30 instances is useless if a tier keeps 3 of them); only distractors are
#: subsampled, by a stable seeded rank on the page id, so growing a tier never
#: reshuffles the smaller one.
TIERS: dict[str, int] = {
    "s": int(os.environ.get("VTS_DOCMARKS_TIER_S", "5000")),
    "m": int(os.environ.get("VTS_DOCMARKS_TIER_M", "50000")),
    "l": int(os.environ.get("VTS_DOCMARKS_TIER_L", "200000")),
}

#: Ordered smallest-to-largest.  Used for the nesting invariant.
TIER_ORDER: tuple[str, ...] = ("s", "m", "l")

#: Salt for the deterministic distractor rank.  Change it and every tier
#: membership is reshuffled — so don't, unless you mean to.
TIER_SALT = os.environ.get("VTS_DOCMARKS_TIER_SALT", "docmarks-v1")

# --------------------------------------------------------------------------
# Class admission
# --------------------------------------------------------------------------

#: A class needs at least this many instances to be admitted as a *query* class.
#: Tobacco800's published "21 logo classes" uses a >=2 bar, which cannot support
#: a train-and-search eval: with two instances there is nothing left to retrieve
#: once one is the query.  ``build_corpus.py`` prints the survival curve over
#: every threshold so this can be set from data rather than taste.
MIN_INSTANCES = int(os.environ.get("VTS_DOCMARKS_MIN_INSTANCES", "10"))

#: Marks smaller than this (longest side, px, at the page's native scan
#: resolution) are recorded but never promoted to a query class.  The
#: 2026-07-13 study found a hard floor around 32 px below which no structural
#: pipeline recovers anything; a class made of sub-floor instances measures the
#: floor, not the method.
MIN_MARK_PX = int(os.environ.get("VTS_DOCMARKS_MIN_MARK_PX", "32"))

#: Connected components smaller than this fraction of the page are dropped as
#: mask speckle rather than treated as marks.
MIN_MARK_AREA_FRAC = float(os.environ.get("VTS_DOCMARKS_MIN_MARK_AREA_FRAC", "0.0002"))

# --------------------------------------------------------------------------
# Contamination — which sources may serve as distractors for which classes
# --------------------------------------------------------------------------
#
# The trap this exists to avoid: RVL-CDIP and Tobacco800 are both drawn from
# IIT-CDIP, so an American Tobacco letterhead is *certain* to appear in an
# RVL-CDIP "distractor" pool.  Unlabelled positives in the distractor set do not
# make the benchmark slightly noisy, they make a correct retrieval count as a
# false positive — the metric punishes the model for being right.  No amount of
# hand annotation fixes that at 200k pages, so it is fixed by construction here.
#
# Read as: "a class from source K may be scored against distractors from any
# source NOT listed in CONTAMINATES[K]".

CONTAMINATES: dict[str, frozenset[str]] = {
    # Indian pseudo-official documents authored for the dataset.  Their marks
    # exist nowhere else on earth, so every other source is a safe distractor.
    "spods": frozenset({"spods"}),
    # Stamps on European scanned invoices.  Likewise self-contained.
    "staver": frozenset({"staver"}),
    # IIT-CDIP tobacco litigation documents.  UCSF's Tobacco industry is the
    # *same underlying archive*, so it is excluded; the other UCSF industries
    # are different companies and are admitted.
    "tobacco800": frozenset({"tobacco800", "ucsf:Tobacco"}),
    # Weakly-labelled UCSF letterhead classes contaminate all of UCSF: the same
    # company's letterhead recurs across industries (Philip Morris reaches Food
    # through Kraft), and the label is metadata-derived rather than observed.
    "ucsf": frozenset({"ucsf"}),
    # Synthetic pastes contaminate only their own backgrounds, which
    # build_corpus.py holds out of every other stratum.
    "synth": frozenset({"synth"}),
}

#: UCSF industries pulled for the distractor pool.  Tobacco is deliberately
#: *first* and deliberately excluded from Tobacco800's eligible distractors by
#: ``CONTAMINATES`` above — it is pulled because it is the richest source of
#: scanned letterhead for the weakly-labelled classes, not despite the clash.
UCSF_INDUSTRIES: tuple[str, ...] = ("Tobacco", "Opioids", "Chemical", "Fossil Fuel", "Drug", "Food")

#: Fraction of page height treated as the letterhead band on a UCSF candidate
#: page.  UCSF ships no boxes, and a mark nobody can see cannot be adjudicated;
#: a letterhead is at the top of the page by definition, so the top strip is a
#: coarse but honest locator to cluster on.  It is never a ground-truth box —
#: the tight box comes from the hand-drawn query crop after adjudication.
LETTERHEAD_BAND_FRAC = float(os.environ.get("VTS_DOCMARKS_LETTERHEAD_BAND_FRAC", "0.22"))

#: Companies whose single-page letters form the letterhead **candidate pool**.
#: ``author`` (who wrote it), not ``collection`` (whose files it sat in): a
#: letter *in* the Philip Morris collection is as likely to be incoming mail on
#: a law firm's letterhead.  Live counts of single-page ``type:letter``
#: documents per author, measured 2026-08-25, are in the README.
#:
#: An author is a pool, never a class.  See ``sources/ucsf.py`` for why turning
#: one into a class id would write two guaranteed errors into the ground truth.
UCSF_LETTERHEAD_AUTHORS: tuple[str, ...] = (
    "PHILIP MORRIS",
    "RJR",
    "LOR, LORILLARD",
    "AMERICAN TOBACCO",
    "BROWN & WILLIAMSON",
    "BATCO",
    "COUNCIL FOR TOBACCO RESEARCH",
    "TOBACCO INSTITUTE",
)

# --------------------------------------------------------------------------
# Identity clustering
# --------------------------------------------------------------------------

#: Default backend for turning per-mark crops into identity classes.  ``phash``
#: is cheap, deterministic and runs with no models — good enough to build the
#: audit slate a human then corrects.  ``siglip`` is the quality option and
#: needs the pile's models dir.
CLUSTER_BACKEND = os.environ.get("VTS_DOCMARKS_CLUSTER_BACKEND", "phash")

#: Agglomerative merge threshold, in the backend's own distance units
#: (normalised Hamming for ``phash``, cosine distance for ``siglip``).
CLUSTER_THRESHOLD = float(os.environ.get("VTS_DOCMARKS_CLUSTER_THRESHOLD", "0.18"))

# --------------------------------------------------------------------------
# Synthesis (layer 3)
# --------------------------------------------------------------------------

#: Instances generated per synthetic class.
SYNTH_INSTANCES_PER_CLASS = int(os.environ.get("VTS_DOCMARKS_SYNTH_PER_CLASS", "30"))

#: Longest-side pixel sizes the pasted mark is drawn from, log-uniformly.  The
#: band spans the 2026-07-13 study's measured cliff (nothing works below ~32 px;
#: 128-256 px is where SIFT recovers) so a sweep can locate it rather than
#: straddle it.
SYNTH_SIZE_PX = (24, 320)

#: Rotation range in degrees.  Scanned marks are near-upright but not exactly:
#: a rubber stamp is applied by hand, a letterhead is not.
SYNTH_ROTATION_DEG = (-8.0, 8.0)

#: Seed for every random choice in synthesis.  One seed, one corpus.
SYNTH_SEED = int(os.environ.get("VTS_DOCMARKS_SYNTH_SEED", "20260825"))


def eligible_distractor(class_source: str, page_source: str, page_industry: str | None = None) -> bool:
    """Is *page_source* safe to score as a distractor for a *class_source* class?

    ``page_industry`` qualifies UCSF pages, so that Tobacco800 classes can use
    UCSF's non-tobacco industries while excluding the tobacco archive they
    overlap with.
    """
    banned = CONTAMINATES.get(class_source, frozenset())
    if page_source in banned:
        return False
    if page_source == "ucsf" and page_industry and f"ucsf:{page_industry}" in banned:
        return False
    return True
