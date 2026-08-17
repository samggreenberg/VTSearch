# 2026-08-12 — a header-only CSV passes `-size +0` (#3129)

**What happened.** The array's watcher reported `non-empty cells: 189/189, zero-byte: 0`
and the run looked complete. It was not: **seven cells held a header row and no
data**. `find -size +0` counts bytes, and a header is bytes. The analyzer caught
it only because it counted loaded rows separately from files found and printed
both numbers.

Those seven turned out to be the most interesting result in the study — runs that
never surfaced a single positive in 150 votes — so the cost of missing them would
have been a wrong headline, not just a small denominator.

**Prevented?** *Partly.* The wave-2 watcher counts cells with `wc -l > 1` rather
than bytes. The general control is the one already in the playbook: **count what
you dropped, and print both numbers**. A completion count that cannot distinguish
"wrote results" from "wrote a header" is the same blind gate as a preflight that
reports ok without looking.

**Related, worth knowing:** `simulate_voting_iterations` never emits a row with
`n_good == 0`, so a starved cell writes a header and exits 0 with no warning
anywhere. Verifying `min(n_good) == 1` across every row is what proved the seven
were starvation rather than an I/O incident. A `starved` column and a one-line
warning would make this visible without an analyst noticing a row-count mismatch.
