# 2026-09-01 — "resume is free" was true of downloads and false of rendering (#3343)

**Study:** #3343 DocMarks full-scale build. **Cost:** near-miss — caught while
timing the render pool, before any of the four restarts that followed needed it.
Each restart would have re-rendered every page already on disk: **~10 h** at
200k pages, in a builder whose stated design premise is that resuming is free.

## What happened

`GRID-RUNBOOK.md` promises "**Resume is free.** Downloads are atomic, rendered
pages are skipped when present, and the Solr cursor order is stable." The first
two thirds were true. The last was not, in the way that mattered.

`fetch_and_render` did skip the *save*:

```python
image_path = out_images / f"{doc_id}_{idx}.png"
if not image_path.exists():
    image.save(image_path)
```

but `image` only exists because `render_pdf_pages(pdf_path)` ran unconditionally
one line earlier. So a resumed job re-rendered every page it already had, at
0.188 s each, and threw the result away. The check guarded the cheap half of the
operation and left the expensive half outside it.

Measured on 60 cached documents: resume went **2.88 → 88.75 pages/s** once
`_render_to_disk` returns dimensions read off the existing PNGs instead. That is
the difference between a restart costing minutes and costing most of a day.

It mattered four times in two days. The pull was restarted for a render pool, a
dev merge that changed what a mark is, an industry-stamping fix, and fetch
concurrency — none of which would have been worth doing at a 10-hour re-render
per attempt. A promise that resume is cheap is what makes a long unattended job
*correctable*; when it is quietly false, every improvement gets priced out and
the run you have is the run you keep.

## Why it was invisible

Resuming produced exactly the right corpus. The only symptom was that a resumed
job took about as long as a fresh one, and there was no baseline to compare it
against — nobody resumes a 200k pull twice to time it.

The docstring reinforced it: the file *says* rendered pages are skipped, and
they are, in the sense the author meant. "Skipped" described the write, and the
reader hears it about the work.

## Now prevented

- `TestResumeSkipsRendering` asserts `_render_to_disk` does not call
  `render_pdf_pages` at all when every target PNG exists, by making that call
  raise. The fast path is the tested behaviour rather than an optimisation
  someone could tidy away.
- `verify_pipeline` reports cold and resumed throughput side by side, so the
  ratio is a number in the log instead of an assumption.

## Still only advice

**When a document promises an operation is cheap, time it.** A resume path is
the least-exercised code in a long-running job and the most load-bearing when
something goes wrong — it is what decides whether a multi-day run can be
*improved* mid-flight or only endured.

**Check that a skip guard wraps the expensive half.** `if not exists: save()`
reads like a skip and is one, for the write. The render above it is where the
time was.
