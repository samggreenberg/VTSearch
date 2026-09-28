#!/usr/bin/env python3
"""Random verification (#4257): what an honest precision-floor promise costs in audit votes.

The #4224 objective lets the user name a precision floor X ("show me what's at
least X right, and as much of it as possible"). A model-based estimate of the
precision above a cut, fitted on the session's own votes, carries the selection
bias of the sorts that chose those votes (#4221). **Random verification** does not:
show the user m items drawn uniformly from a candidate returned set, and the share
that are right is an unbiased estimate of that set's precision. A one-sided
Clopper-Pearson lower bound at level 1 - alpha turns it into a promise that, for
every corpus whose returned set is really below X, is made with probability at
most alpha. The price is the user's audit votes. This prices it, offline, from
the #4224 rank frames (``docs/experiments/2026-09-28-rank-frames/``): a frame
carries the ranks of its corpus's positives, so every audit label is simulated
exactly, and the audit draws are the only randomness.

Rules (``RULES``):

* ``a:*`` - **one round.** Audit ``min(m, k0)`` items uniformly from the top
  ``k0`` and promise the top ``k0`` iff the lower bound is >= X; otherwise promise
  nothing. ``k0`` is the consistent estimator's cut (``a:consistent``: nothing to
  audit where it promised nothing) or a fixed rank (``a:top16`` ...).
* ``b:*`` - **sequential shrinking.** As ``a``, but a failed round halves ``k``
  and audits again, for at most R rounds, each at alpha / R (Bonferroni). Every
  label already seen inside the new top ``k`` is kept, and ``m`` fresh items are
  drawn from the rest of it: the kept items are a uniform sample of the new top
  ``k`` whose size depends only on positions, so each round's sample is uniform
  and the union bound holds. ``b:consistent`` starts at the consistent cut (the
  model proposes, the audit disposes: the issue's rule (d)); ``b:top128`` starts
  at 128 and can shrink to 4, ``b:top128/3`` stops at 32.
* ``c:*`` - **stratified curve.** Split the top 128 into geometric bins
  (``EDGES``), spread the m audits across them (round-robin from the top, capped
  at each bin's size), bound each sampled bin's precision, and promise the
  largest bin edge whose weighted cumulative lower bound is >= X. ``c:union``
  bounds every sampled bin at alpha / J (J sampled, uncensused bins), which makes
  every cumulative bound hold at once. ``c:seq`` tests the edges in increasing
  order, the j-th at alpha / J_j over its own bins, and stops at the first
  failure: a fixed-sequence procedure, level alpha without the global split.

Whenever an audit covers every item of the set it bounds (``m >= k``), it is a
census and its precision is exact, with no error to budget.

Metrics per rule x m x alpha x X x prevalence (``summary.csv``; also by vote
checkpoint ``t`` and by object-size ``band``):

* ``broken_of_made`` - achieved precision of the promised set < X, of promises
  made. Its cluster (category, seed) standard error is ``se_broken_of_made``.
* ``broken_of_all`` - the same, of every frame x draw. **This is what
  Clopper-Pearson bounds by alpha**: a valid rule keeps it <= alpha on every
  frame. The share of promises made that break is a different number and can
  exceed alpha when most candidates are below X.
* ``promised`` - share of frame x draws with a promise; ``reachable`` - share of
  frames where some top k meets X at all (the ceiling on ``promised``).
* ``recall``, ``oracle``, ``recall_share`` - mean recall (no promise counts as
  0), mean oracle recall (the largest recall of any top k with precision >= X),
  and their ratio.
* ``census_of_made``, ``audited_share``, ``unaudited_recall``, ``new_pos`` -
  how much of the promise the user already looked at: the share of promises
  whose every item was audited, the audited share of the promised items, and the
  recall (and mean count) of the positives inside the promise that were **not**
  audited. A census promises nothing the user has not seen; ``new_pos`` is what
  the promise adds to their own labour.
* ``votes_mean``, ``votes_p90`` - audit votes spent; ``pos_per_vote`` and
  ``new_pos_per_vote`` - positives delivered in the promised set, and unaudited
  positives, per audit vote spent (a rate, so a rule that delivers nothing reads
  0 rather than an infinite cost).
* ``d_recall_vs_shipped`` (+ ``se_``) - the paired recall gain over the shipped
  estimator's cut on the same frames.

The two stored baselines (``shipped``, ``consistent``) appear with ``m = 0``.

    python analyze_random_verification.py [--frames DIR] [--out DIR] [--draws 20] [--no-figures]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[3]
FRAMES_DIR = REPO / "docs" / "experiments" / "2026-09-28-rank-frames"
OUT_DIR = REPO / "docs" / "experiments" / "2026-09-28-random-verification"

#: Pool prevalence -> rank-frame file. Scenarios, not claims about users.
WORLDS = {
    "0.44%": "rank_frames_0p44pct.csv.gz",
    "0.1%": "rank_frames_0p1pct.csv.gz",
    "5%": "rank_frames_5pct.csv.gz",
}
FLOORS = (0.25, 0.5, 0.75)
ALPHAS = (0.05, 0.10)
BUDGETS = (5, 10, 20, 40)
DRAWS = 20
SEED = 4257
#: Stratified bins: the top 128 in geometric bins.
EDGES = (0, 4, 8, 16, 32, 64, 128)
#: The (X, alpha, m) the literal examples are drawn at.
EXAMPLE = (0.5, 0.05, 20)
#: Largest audited sample the lower-bound table must cover (R rounds of m, kept).
NMAX = 6 * max(BUDGETS) + 1
EPS = 1e-9


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str  # "a" one round, "b" sequential shrinking, "c" stratified
    start: int | str = 0  # a fixed k, or "consistent"
    rounds: int = 1
    seq: bool = False  # c only: fixed-sequence instead of a global union bound


RULES = (
    Rule("a:consistent", "a", "consistent"),
    Rule("a:top16", "a", 16),
    Rule("a:top32", "a", 32),
    Rule("a:top64", "a", 64),
    Rule("b:consistent", "b", "consistent", rounds=4),
    Rule("b:top128", "b", 128, rounds=6),
    Rule("b:top128/3", "b", 128, rounds=3),
    Rule("c:union", "c"),
    Rule("c:seq", "c", seq=True),
)
BASELINES = ("shipped", "consistent")


# --------------------------------------------------------------------------- frames


@dataclass
class Frames:
    """One prevalence's rank frames, as arrays the rules can vectorise over."""

    world: str
    meta: pd.DataFrame  # one row per frame
    ranks: np.ndarray  # (F, P) sorted 0-based positive ranks, padded with n_corpus + 1

    @property
    def n(self) -> int:
        return len(self.meta)

    def hits(self, k: np.ndarray) -> np.ndarray:
        """Positives in the top ``k``; ``k`` is (F,) or (F, D)."""
        k = np.asarray(k)
        if k.ndim == 1:
            return (self.ranks < k[:, None]).sum(axis=1)
        return (self.ranks[:, None, :] < k[:, :, None]).sum(axis=2)


def parse_frames(df: pd.DataFrame, world: str) -> Frames:
    ranks = [np.sort(np.array(str(s).split(), dtype=np.int64)) for s in df["pos_ranks"]]
    width = max(len(r) for r in ranks)
    pad = np.full((len(ranks), width), np.iinfo(np.int64).max // 4, dtype=np.int64)
    for i, r in enumerate(ranks):
        pad[i, : len(r)] = r
    meta = df.drop(columns=["pos_ranks"]).reset_index(drop=True)
    if not (meta["n_pos"].to_numpy() == np.array([len(r) for r in ranks])).all():
        raise ValueError(f"{world}: n_pos disagrees with pos_ranks")
    meta["cell"] = meta["category"].astype(str) + "#" + meta["seed"].astype(str)
    return Frames(world, meta, pad)


def load_frames(frames_dir: Path, worlds: dict[str, str]) -> dict[str, Frames]:
    return {w: parse_frames(pd.read_csv(frames_dir / f), w) for w, f in worlds.items()}


def meets(h: np.ndarray, k: np.ndarray, x: float) -> np.ndarray:
    """Is ``h / k >= x`` (with ``k > 0``)?"""
    return h >= x * k - EPS


def oracle_k(fr: Frames, x: float) -> np.ndarray:
    """The largest k whose top-k precision is >= x (0 if none): the oracle's cut.

    At the largest such k holding c positives, k <= floor(c / x) and the c-th
    positive sits inside it, so the answer is the largest ``floor(i / x)`` that
    still contains the i-th positive.
    """
    i = np.arange(1, fr.ranks.shape[1] + 1)[None, :]
    kk = np.floor(i / x + EPS).astype(np.int64)
    ok = (kk >= fr.ranks + 1) & (i <= fr.meta["n_pos"].to_numpy()[:, None])
    k = np.where(ok, kk, 0).max(axis=1)
    return np.minimum(k, fr.meta["n_corpus"].to_numpy())


# --------------------------------------------------------------------------- bounds


@lru_cache(maxsize=None)
def lb_table(level: float) -> np.ndarray:
    """``T[n, s]``: one-sided Clopper-Pearson lower bound on a proportion at ``1 - level``."""
    n = np.arange(NMAX + 1)[:, None]
    s = np.arange(NMAX + 1)[None, :]
    with np.errstate(invalid="ignore"):
        t = beta.ppf(level, np.maximum(s, 1), np.maximum(n - s + 1, 1))
    return np.where((s == 0) | (s > n), 0.0, t)


def lower_bound(s: np.ndarray, n: np.ndarray, level: float) -> np.ndarray:
    if np.max(n, initial=0) > NMAX:
        raise ValueError(f"an audit of {np.max(n)} items exceeds NMAX={NMAX}")
    return lb_table(level)[n, s]


def min_all_positive(x: float, alpha: float) -> int:
    """Fewest audits, all of them positive, whose lower bound reaches x: alpha**(1/m) >= x."""
    return int(np.ceil(np.log(alpha) / np.log(x) - EPS))


# --------------------------------------------------------------------------- rules


@dataclass
class Outcome:
    """One run's draws, each (F, D): the promised k (0 for none), the audit votes it
    cost, and how many audited items (and audited positives) lie inside the promise."""

    k: np.ndarray
    votes: np.ndarray
    seen: np.ndarray
    seen_pos: np.ndarray
    trace: list[dict]


def start_k(fr: Frames, rule: Rule, x: float) -> np.ndarray:
    if rule.start == "consistent":
        return fr.meta[f"consistent_k_x{round(x * 100)}"].to_numpy().astype(np.int64)
    return np.minimum(np.full(fr.n, int(rule.start)), fr.meta["n_corpus"].to_numpy())


def simulate_rounds(fr: Frames, k0: np.ndarray, rounds: int, m: int, x: float, alpha: float, rng, draws: int):
    """Rules a (``rounds=1``) and b."""
    shape = (fr.n, draws)
    level = alpha / rounds
    promised = np.zeros(shape, dtype=np.int64)
    votes = np.zeros(shape, dtype=np.int64)
    seen = np.zeros(shape, dtype=np.int64)
    seen_pos = np.zeros(shape, dtype=np.int64)
    done = np.broadcast_to((k0 <= 0)[:, None], shape).copy()
    n = np.zeros(shape, dtype=np.int64)  # labels known inside the current top k
    s = np.zeros(shape, dtype=np.int64)  # ... of which positive
    k_prev = c_prev = None
    trace = []
    for r in range(rounds):
        kr = np.maximum(k0 >> r, 1)
        valid = (k0 >> r) >= 1
        cr = fr.hits(kr)
        k_b, c_b = kr[:, None], cr[:, None]
        if r:
            # Keep the known labels that fall inside the new top k: given how many
            # positives the sample holds, they are a uniform subset of the positives.
            s_top = rng.hypergeometric(np.broadcast_to(c_b, shape), np.broadcast_to(c_prev - c_b, shape), s)
            neg_top = rng.hypergeometric(
                np.broadcast_to(k_b - c_b, shape), np.broadcast_to((k_prev - c_prev) - (k_b - c_b), shape), n - s
            )
            s, n = s_top, s_top + neg_top
        fresh = np.minimum(m, k_b - n)
        s_new = rng.hypergeometric(c_b - s, (k_b - c_b) - (n - s), fresh)
        live = ~done & valid[:, None]
        votes += np.where(live, fresh, 0)
        n, s = n + fresh, s + s_new
        census = n >= k_b
        passed = np.where(census, meets(s, k_b, x), lower_bound(s, n, level) >= x - EPS)
        newly = live & passed
        promised[newly] = np.broadcast_to(k_b, shape)[newly]
        seen[newly], seen_pos[newly] = n[newly], s[newly]
        trace.append({"k": kr, "n": n.copy(), "s": s.copy(), "pass": passed, "live": live, "census": census})
        done |= newly
        k_prev, c_prev = k_b, c_b
    return Outcome(promised, votes, seen, seen_pos, trace)


def allocate(m: int, sizes: list[int]) -> list[int]:
    """Round-robin the m audits over the bins from the top, capped at each bin's size."""
    alloc = [0] * len(sizes)
    left = m
    while left > 0 and any(a < z for a, z in zip(alloc, sizes)):
        for b, z in enumerate(sizes):
            if left and alloc[b] < z:
                alloc[b] += 1
                left -= 1
    return alloc


def simulate_stratified(fr: Frames, rule: Rule, m: int, x: float, alpha: float, rng, draws: int):
    """Rule c."""
    shape = (fr.n, draws)
    edges = np.array(EDGES)
    sizes = list(np.diff(edges))
    alloc = allocate(m, sizes)
    cum = np.stack([fr.hits(np.full(fr.n, e)) for e in edges], axis=1)
    c_bin = np.diff(cum, axis=1)  # (F, B) positives per bin
    s_bin = [
        rng.hypergeometric(np.repeat(c_bin[:, b, None], draws, 1), sizes[b] - c_bin[:, b, None], alloc[b])
        for b in range(len(sizes))
    ]
    census = [a >= z for a, z in zip(alloc, sizes)]
    sampled = [0 < a < z for a, z in zip(alloc, sizes)]

    def bin_lbs(nbins: int, level: float) -> list[np.ndarray]:
        out = []
        for b in range(nbins):
            if census[b]:
                out.append(s_bin[b] / sizes[b])
            elif sampled[b]:
                out.append(lower_bound(s_bin[b], np.full(shape, alloc[b]), level))
            else:
                out.append(np.zeros(shape))
        return out

    top = np.full(shape, -1)  # index of the last bin inside the promise, -1 for none
    if not rule.seq:
        j_all = max(1, sum(sampled))
        lbs = bin_lbs(len(sizes), alpha / j_all)
        acc = np.zeros(shape)
        for j in range(len(sizes)):
            acc = acc + sizes[j] * lbs[j]
            top = np.where(acc / edges[j + 1] >= x - EPS, j, top)
    else:
        alive = np.ones(shape, dtype=bool)
        for j in range(len(sizes)):
            lbs = bin_lbs(j + 1, alpha / max(1, sum(sampled[: j + 1])))
            cum_lb = sum(sizes[b] * lbs[b] for b in range(j + 1)) / edges[j + 1]
            alive &= cum_lb >= x - EPS
            top = np.where(alive, j, top)
    cum_alloc = np.concatenate([[0], np.cumsum(alloc)])
    cum_s = np.concatenate([np.zeros((1, *shape), dtype=np.int64), np.cumsum(np.stack(s_bin), axis=0)])
    promised = edges[top + 1]
    seen = cum_alloc[top + 1]
    seen_pos = np.take_along_axis(cum_s, (top + 1)[None], axis=0)[0]
    votes = np.full(shape, sum(alloc), dtype=np.int64)
    trace = [
        {"k": np.full(fr.n, edges[b + 1]), "n": np.full(shape, alloc[b]), "s": s_bin[b]} for b in range(len(sizes))
    ]
    return Outcome(promised, votes, seen, seen_pos, trace)


def simulate(fr: Frames, rule: Rule, m: int, x: float, alpha: float, draws: int, seed: int) -> Outcome:
    """One (rule, m, alpha, X) run over every frame of a prevalence, ``draws`` audits each."""
    key = [zlib.crc32(fr.world.encode()), zlib.crc32(rule.name.encode()), round(x * 100), round(alpha * 100), m]
    rng = np.random.default_rng([seed, *key])
    if rule.kind == "c":
        return simulate_stratified(fr, rule, m, x, alpha, rng, draws)
    return simulate_rounds(fr, start_k(fr, rule, x), rule.rounds if rule.kind == "b" else 1, m, x, alpha, rng, draws)


# --------------------------------------------------------------------------- metrics


def cluster_se(num: np.ndarray, den: np.ndarray, cells: np.ndarray) -> float:
    """Linearised standard error of sum(num) / sum(den), clustered by cell."""
    g = pd.DataFrame({"c": cells, "a": num, "b": den}).groupby("c")[["a", "b"]].sum()
    tot = g["b"].sum()
    if tot <= 0 or len(g) < 2:
        return float("nan")
    r = g["a"].sum() / tot
    return float(np.sqrt(len(g) / (len(g) - 1) * ((g["a"] - r * g["b"]) ** 2).sum()) / tot)


def slices(fr: Frames):
    yield "all", np.ones(fr.n, dtype=bool)
    for t in sorted(fr.meta["t"].unique()):
        yield f"t={t}", (fr.meta["t"] == t).to_numpy()
    for b in ("large", "medium", "small"):
        yield f"band={b}", (fr.meta["band"] == b).to_numpy()


def summarise(fr: Frames, out: Outcome, x: float, oracle_rec: np.ndarray, shipped_rec: np.ndarray) -> list[dict]:
    """Rows for one (rule, m, alpha, X) run, per slice."""
    k, votes = out.k, out.votes
    n_pos = fr.meta["n_pos"].to_numpy()[:, None]
    h = fr.hits(k)
    made = k > 0
    broken = made & ~meets(h, np.maximum(k, 1), x)
    rec = np.where(made, h / n_pos, 0.0)
    new_rec = np.where(made, (h - out.seen_pos) / n_pos, 0.0)
    census = made & (out.seen >= k)
    draws = k.shape[1]
    cells = fr.meta["cell"].to_numpy()
    rows = []
    for name, sel in slices(fr):
        if not sel.any():
            continue
        f_made, f_broken = made[sel].sum(1), broken[sel].sum(1)
        f_rec = rec[sel].mean(1)
        f_votes, f_pos = votes[sel].mean(1), np.where(made[sel], h[sel], 0).mean(1)
        f_new = np.where(made[sel], h[sel] - out.seen_pos[sel], 0).mean(1)
        cl = cells[sel]
        ones = np.full(sel.sum(), draws)
        d_rec = f_rec - shipped_rec[sel]
        rows.append(
            {
                "slice": name,
                "frames": int(sel.sum()),
                "promises": int(f_made.sum()),
                "reachable": float((oracle_rec[sel] > 0).mean()),
                "promised": f_made.sum() / ones.sum(),
                "broken_of_made": f_broken.sum() / f_made.sum() if f_made.sum() else np.nan,
                "se_broken_of_made": cluster_se(f_broken, f_made, cl),
                "broken_of_all": f_broken.sum() / ones.sum(),
                "se_broken_of_all": cluster_se(f_broken, ones, cl),
                "recall": f_rec.mean(),
                "oracle": oracle_rec[sel].mean(),
                "recall_share": f_rec.mean() / oracle_rec[sel].mean() if oracle_rec[sel].mean() else np.nan,
                "census_of_made": census[sel].sum() / f_made.sum() if f_made.sum() else np.nan,
                "audited_share": out.seen[sel][made[sel]].sum() / k[sel][made[sel]].sum() if f_made.sum() else np.nan,
                "unaudited_recall": new_rec[sel].mean(),
                "new_pos": f_new.mean(),
                "new_pos_per_vote": f_new.mean() / f_votes.mean() if f_votes.mean() else np.nan,
                "votes_mean": f_votes.mean(),
                "votes_p90": float(np.percentile(votes[sel], 90)),
                "pos_delivered": f_pos.mean(),
                "pos_per_vote": f_pos.mean() / f_votes.mean() if f_votes.mean() else np.nan,
                "d_recall_vs_shipped": d_rec.mean(),
                "se_d_recall_vs_shipped": cluster_se(d_rec, np.ones(sel.sum()), cl),
            }
        )
    return rows


def baseline_k(fr: Frames, name: str, x: float) -> np.ndarray:
    return fr.meta[f"{name}_k_x{round(x * 100)}"].to_numpy().astype(np.int64)


def recall_of(fr: Frames, k: np.ndarray) -> np.ndarray:
    return np.where(k > 0, fr.hits(k) / fr.meta["n_pos"].to_numpy(), 0.0)


def run(
    frames: dict[str, Frames],
    draws: int,
    seed: int,
    rules=RULES,
    budgets=BUDGETS,
    alphas=ALPHAS,
    floors=FLOORS,
    keep=lambda x, a, m: False,
):
    """Summary rows for every run; the raw draws are kept only where ``keep(X, alpha, m)``."""
    rows, traces = [], {}
    for w, fr in frames.items():
        for x in floors:
            orec = recall_of(fr, oracle_k(fr, x))
            srec = recall_of(fr, baseline_k(fr, "shipped", x))
            for name in BASELINES:
                k = baseline_k(fr, name, x)[:, None]
                zero = np.zeros_like(k)
                for r in summarise(fr, Outcome(k, zero, zero, zero, []), x, orec, srec):
                    rows.append({"world": w, "rule": name, "X": x, "alpha": np.nan, "m": 0, **r})
            for rule in rules:
                for a in alphas:
                    for m in budgets:
                        out = simulate(fr, rule, m, x, a, draws, seed)
                        if keep(x, a, m):
                            traces[(w, rule.name, x, a, m)] = out
                        for r in summarise(fr, out, x, orec, srec):
                            rows.append({"world": w, "rule": rule.name, "X": x, "alpha": a, "m": m, **r})
    return pd.DataFrame(rows), traces


# --------------------------------------------------------------------------- examples


def trace_text(rule: Rule, trace: list[dict], i: int, d: int) -> str:
    """One frame's audit, one draw, as the rounds it went through."""
    parts = []
    for step in trace:
        if rule.kind == "c":
            n = int(step["n"][i, d])
            if n:
                parts.append(f"bin<{int(step['k'][i])}: {int(step['s'][i, d])}/{n}")
            continue
        if not step["live"][i, d]:
            break
        k, n, s = int(step["k"][i]), int(step["n"][i, d]), int(step["s"][i, d])
        mark = "pass" if step["pass"][i, d] else "fail"
        parts.append(f"top {k}: {s}/{n}{' (all)' if step['census'][i, d] else ''} {mark}")
    return "; ".join(parts) if parts else "nothing to audit"


#: Cells the #4220 report quoted, plus the ones picked below for what they show.
NAMED = (("bird@large", 0), ("apple@large", 2), ("tv@small", 3), ("airplane@large", 0), ("skis@medium", 0))


def examples(frames: dict[str, Frames], traces: dict, x: float, alpha: float, m: int, t: int = 150) -> pd.DataFrame:
    """Literal audits: draw 0 of each shown rule, on named and on picked cells."""
    rules = [r for r in RULES if r.name in ("a:consistent", "a:top32", "b:top128", "c:seq")]
    rows = []
    for w, fr in frames.items():
        at_t = fr.meta.index[fr.meta["t"] == t].to_numpy()
        ok = oracle_k(fr, x)
        picks = [i for c, s in NAMED for i in at_t if fr.meta.at[i, "category"] == c and fr.meta.at[i, "seed"] == s]
        b = traces[(w, "b:top128", x, alpha, m)]
        kb, seen_b = b.k[:, 0], b.seen[:, 0]
        cons = baseline_k(fr, "consistent", x)
        hits_c = fr.hits(np.maximum(cons, 1))
        # One of each: the audit catching a broken model cut, a promise larger than
        # its audit, a promise that is a census of the top, and a timid miss.
        caught = [i for i in at_t if cons[i] > 0 and not meets(hits_c[i], cons[i], x)]
        beyond = [i for i in at_t if kb[i] >= 32 and seen_b[i] < kb[i]]
        census = [i for i in at_t if 0 < kb[i] <= seen_b[i]]
        timid = [i for i in at_t if kb[i] == 0 and ok[i] >= 32]
        chosen = list(picks)
        for group in (caught, beyond, census, timid):
            fresh = [i for i in group if fr.meta.at[i, "category"] not in set(fr.meta.loc[chosen, "category"])]
            chosen += fresh[:2]
        for i in chosen:
            meta = fr.meta.loc[i]
            for rule in rules:
                out = traces[(w, rule.name, x, alpha, m)]
                kk = int(out.k[i, 0])
                h = int((fr.ranks[i] < kk).sum())
                rows.append(
                    {
                        "world": w,
                        "cell": meta["category"],
                        "seed": int(meta["seed"]),
                        "t": t,
                        "n_pos": int(meta["n_pos"]),
                        "consistent_k": int(cons[i]),
                        "consistent_precision": round(float(hits_c[i] / cons[i]), 3) if cons[i] else np.nan,
                        "oracle_k": int(ok[i]),
                        "oracle_recall": round(float((fr.ranks[i] < ok[i]).sum() / meta["n_pos"]), 3),
                        "rule": rule.name,
                        "audit": trace_text(rule, out.trace, i, 0),
                        "votes": int(out.votes[i, 0]),
                        "promised_k": kk,
                        "audited_inside": int(out.seen[i, 0]),
                        "precision": round(h / kk, 3) if kk else np.nan,
                        "recall": round(h / meta["n_pos"], 3),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures

INK, SOFT, GRIDC = "#0b0b0b", "#52514e", "#e3e6ea"
#: The dataviz skill's categorical order (light surface), slots 1-6, fixed per rule
#: and never cycled; validate_palette.js passes it. Three slots sit below 3:1 on
#: the surface, so every figure's numbers are also in the report's tables, and each
#: rule has its own marker as well as its colour.
SERIES = {
    "a:consistent": ("#2a78d6", "o"),
    "a:top32": ("#eb6834", "s"),
    "b:consistent": ("#1baf7a", "^"),
    "b:top128": ("#eda100", "D"),
    "b:top128/3": ("#e87ba4", "v"),
    "c:seq": ("#008300", "P"),
}
WORLD_ORDER = ("0.44%", "0.1%", "5%")
WORLD_TITLE = {"0.44%": "0.44% (COCO Better's default)", "0.1%": "0.1%", "5%": "5%"}
#: A broken-of-made point read off fewer promises than this is drawn hollow.
FEW_PROMISES = 100


def _plt():
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.size": 10.5,
            "text.color": INK,
            "axes.edgecolor": SOFT,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRIDC,
            "grid.linewidth": 0.8,
            "xtick.color": SOFT,
            "ytick.color": SOFT,
            "savefig.dpi": 130,
            "legend.frameon": False,
        }
    )
    return plt


def _budget_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
    ax.minorticks_off()


def _rule_lines(ax, s: pd.DataFrame, w: str, metric: str, alpha: float, hollow_if_few: bool = False) -> None:
    for rule, (color, marker) in SERIES.items():
        d = s[(s.world == w) & (s.rule == rule) & (s.alpha == alpha)].sort_values("m")
        ax.plot(d["m"], d[metric], color=color, lw=2, label=rule)
        few = (d["promises"] < FEW_PROMISES) if hollow_if_few else np.zeros(len(d), dtype=bool)
        ax.plot(d["m"][~few], d[metric][~few], ls="none", marker=marker, ms=7, color=color, mec="white", mew=1)
        ax.plot(d["m"][few], d[metric][few], ls="none", marker=marker, ms=7, mfc="white", mec=color, mew=1.5)


def _baseline(ax, s: pd.DataFrame, w: str, metric: str, name: str, top: float) -> None:
    b = s[(s.world == w) & (s.rule == name)]
    v = b[metric]
    if not len(v) or not np.isfinite(v.iloc[0]) or not b["promises"].iloc[0]:
        return  # a baseline that promises nothing has nothing to draw
    y = float(v.iloc[0])
    label = f"{name}, 0 votes: {y:.2g}"
    if y <= top:
        ax.axhline(y, color=SOFT, lw=1.2, ls=":")
        ax.annotate(label, (40, y), xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", color=SOFT)
    else:  # off the axis: say so rather than clip it
        ax.annotate(
            f"{label} (off scale)",
            (40, top),
            xytext=(0, -3),
            textcoords="offset points",
            ha="right",
            va="top",
            color=SOFT,
        )


def _save(fig, axes, out: Path, name: str, title: str) -> Path:
    """One legend for the figure, under its title and clear of the data."""
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.965), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path


def figure_honesty(summary: pd.DataFrame, out: Path, x: float = 0.5, alpha: float = 0.05) -> Path:
    """Promises broken, of those made and of all frames, against the audit budget m."""
    plt = _plt()
    s = summary[(summary.slice == "all") & (summary.X == x)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True)
    rows = (("broken_of_made", "broken, of promises made", 1.0), ("broken_of_all", "broken, of all frames", 0.12))
    for col, w in enumerate(WORLD_ORDER):
        for row, (metric, label, top) in enumerate(rows):
            ax = axes[row, col]
            _rule_lines(ax, s, w, metric, alpha, hollow_if_few=metric == "broken_of_made")
            for name in BASELINES:
                _baseline(ax, s, w, metric, name, top)
            ax.axhline(alpha, color=INK, lw=1, ls="--")
            ax.annotate(f"α = {alpha:g}", (5, alpha), xytext=(2, 3), textcoords="offset points", color=INK)
            ax.set_ylim(-0.02 * top, top)
            _budget_axis(ax)
            if row == 0:
                ax.set_title(WORLD_TITLE[w], fontsize=11)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("audit votes per round, m")
    title = (
        f"Honesty: promises broken (X = {x:g}, α = {alpha:g}, pooled over t). "
        f"Hollow: read off fewer than {FEW_PROMISES} promises"
    )
    path = _save(fig, axes, out, "honesty.png", title)
    plt.close(fig)
    return path


def figure_recall(summary: pd.DataFrame, out: Path, x: float = 0.5, alpha: float = 0.05) -> Path:
    """Recall / oracle, and the part of it the user did not audit, against the audit budget m."""
    plt = _plt()
    s = summary[(summary.slice == "all") & (summary.X == x)].copy()
    s["unaudited_share"] = s["unaudited_recall"] / s["oracle"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharex=True, sharey=True)
    rows = (("recall_share", "recall ÷ oracle recall"), ("unaudited_share", "unaudited recall ÷ oracle recall"))
    for col, w in enumerate(WORLD_ORDER):
        for row, (metric, label) in enumerate(rows):
            ax = axes[row, col]
            _rule_lines(ax, s, w, metric, alpha)
            _baseline(ax, s, w, metric, "shipped", 1.0)
            ax.set_ylim(-0.02, 1.0)
            _budget_axis(ax)
            if row == 0:
                ax.set_title(WORLD_TITLE[w], fontsize=11)
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("audit votes per round, m")
    title = f"Recall bought (X = {x:g}, α = {alpha:g}, pooled over t). Bottom: only positives the user did not audit"
    path = _save(fig, axes, out, "recall.png", title)
    plt.close(fig)
    return path


def figure_cost(summary: pd.DataFrame, out: Path, alpha: float = 0.05) -> Path:
    """Unaudited positives delivered against mean audit votes spent, per X and prevalence."""
    plt = _plt()
    s = summary[(summary.slice == "all") & ((summary.alpha == alpha) | summary.alpha.isna())]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.5), sharex=True, sharey="row")
    for col, w in enumerate(WORLD_ORDER):
        for row, x in enumerate(FLOORS):
            ax = axes[row, col]
            for rule, (color, marker) in SERIES.items():
                d = s[(s.world == w) & (s.rule == rule) & (s.X == x)].sort_values("m")
                ax.plot(d["votes_mean"], d["new_pos"], color=color, marker=marker, lw=2, ms=7, label=rule, mec="white")
            b = s[(s.world == w) & (s.rule == "shipped") & (s.X == x)]
            if len(b):
                ax.plot([0], b["new_pos"], marker="*", ms=12, color=SOFT, ls="none", label="shipped (0 votes)")
            if row == 0:
                ax.set_title(WORLD_TITLE[w], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"X = {x:g}\nunaudited positives delivered")
            if row == 2:
                ax.set_xlabel("mean audit votes spent")
    title = (
        "What the votes buy: positives inside the promise that the user never audited, "
        f"m = 5, 10, 20, 40 along each line (α = {alpha:g})"
    )
    path = _save(fig, axes, out, "cost.png", title)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- main


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--frames", type=Path, default=FRAMES_DIR, help="directory holding the rank_frames_*.csv.gz")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    worlds = {w: f for w, f in WORLDS.items() if (args.frames / f).exists()}
    if not worlds:
        raise SystemExit(f"no rank frames under {args.frames}")
    frames = load_frames(args.frames, worlds)
    args.out.mkdir(parents=True, exist_ok=True)
    ex_x, ex_a, ex_m = EXAMPLE
    summary, traces = run(frames, args.draws, args.seed, keep=lambda x, a, m: (x, a, m) == EXAMPLE)
    num = summary.select_dtypes("number").columns
    summary[num] = summary[num].astype(float).round(4)
    summary[summary.slice == "all"].drop(columns="slice").to_csv(args.out / "summary.csv", index=False)
    summary[summary.slice.str.startswith("t=")].to_csv(args.out / "summary_by_t.csv", index=False)
    summary[summary.slice.str.startswith("band=")].to_csv(args.out / "summary_by_band.csv", index=False)
    ex = examples(frames, traces, ex_x, ex_a, ex_m)
    ex.to_csv(args.out / "examples.csv", index=False)
    pd.DataFrame(
        [{"X": x, "alpha": a, "min_all_positive_audits": min_all_positive(x, a)} for x in FLOORS for a in ALPHAS]
    ).to_csv(args.out / "min_audit.csv", index=False)
    prov = {
        "issue": 4257,
        "frames": {w: {"file": f, "sha256_16": sha(args.frames / f), "frames": frames[w].n} for w, f in worlds.items()},
        "draws_per_frame": args.draws,
        "seed": args.seed,
        "floors": FLOORS,
        "alphas": ALPHAS,
        "budgets": BUDGETS,
        "strata_edges": EDGES,
        "rules": [r.__dict__ for r in RULES],
    }
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    if not args.no_figures:
        figure_honesty(summary, args.out)
        figure_recall(summary, args.out)
        figure_cost(summary, args.out)
    for w, fr in frames.items():
        print(f"{w}: {fr.n} frames x {args.draws} draws")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
