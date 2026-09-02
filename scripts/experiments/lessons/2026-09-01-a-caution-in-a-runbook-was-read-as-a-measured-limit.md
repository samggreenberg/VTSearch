# 2026-09-01 — a caution in a runbook was read as a measured limit (#3343)

**Study:** #3343 DocMarks full-scale build, the 216k-page UCSF pull.
**Cost:** ~2 days of wall clock. The pull ran at 1.94 pages/s for two days and
finishes at 8.58. Nothing failed; it was just slow, which is why nobody looked.

## What happened

The issue and `docmarks/GRID-RUNBOOK.md` both say the UCSF pull cannot be
parallelised "without being rude to UCSF and probably rate-limited". That is
good advice and it was written before anyone had pulled at scale. It was then
treated, for two days and by me, as a statement about UCSF's *measured*
tolerance — so the pull stayed strictly serial and the question of what UCSF
actually does never got asked.

The measurement that settles it took one 30-minute job. The build reported
`ucsf: skipped 4003 document(s) that failed to download or render`; retrying a
sample of those ids returned **403 Access Denied, 40 times out of 40, and not a
single 429, 503 or 509**. They are documents indexed in Solr whose PDFs are not
public — permanent, per-document, unrelated to pacing. Across ~120,000 requests
at ~3/s over two days UCSF never once pushed back.

That evidence does not say concurrency is safe. It says we were nowhere near
the limit and had no idea where it was. Those are different claims and the
design follows from the second: fetch 3-wide behind a `_Throttle` that gives up
a worker and doubles its delay on any rate-limit response, so sustained pushback
**converges on the serial behaviour that ran before** rather than hammering
through it. The limit is now detected instead of assumed, in either direction.

Measured end to end on 60 real documents:

| configuration | pages/s |
|---|---:|
| serial fetch, serial render (as shipped) | 1.94 |
| serial fetch, 4-process render pool | 2.88 |
| 3-wide fetch, 4-process render pool | **8.58** |
| resume, everything cached | 88.75 |

## The wrong lever came first, off a contaminated number

The first diagnosis read the running job at **76% CPU** and concluded rendering
was the bottleneck. It was not. That sample spanned the job's startup, where
SPODS mask decomposition is pure single-threaded CPU. Sampled in **steady
state** the same job was 37% CPU — 0.33 s/page waiting on UCSF against 0.19 s
rendering — and the render pool that looked like a 4× was worth 1.5×.

Both numbers are honest measurements of the process. Only one of them measures
the phase whose cost you are trying to change.

## Why it stayed invisible

A pull that is 2.5× slower than it needs to be produces no error, no warning and
no failed cell. It produces a correct corpus, later. The only symptom is an ETA,
and an ETA is easy to accept as the cost of the work rather than as a number
with a cause.

## Now prevented

- `load_ucsf` prints its own throughput and utilisation when the pull finishes —
  pages/s, wall clock, and the fetch/render split — so "we ran two days at a
  third of the achievable rate" is a line in the log rather than something
  nobody thought to measure.
- The same line reports `_Throttle`'s backoff count. A pull that *never* backs
  off is now visibly under-driven; a pull that backs off constantly is visibly
  over-driven. Either way the number is on screen.
- `TestFetchThrottle` pins that sustained pushback converges on serial and that
  `RateLimited` stays distinct from `FetchError` — a 403 is permanent and must
  be skipped, a 429 means the document is still there.

## Still only advice

**A caution is not a measurement, and the difference is usually one cheap job.**
When a runbook says "don't do X, it will probably break", check whether anyone
ever saw it break. Here the evidence was already sitting in the build's own
warning line — 4,003 failures nobody had classified — and classifying them
answered the question in half an hour.

**Sample steady state, never the process lifetime.** A long job's startup phase
has a different cost profile from its body, and an average over both will point
at the wrong resource with complete confidence.
