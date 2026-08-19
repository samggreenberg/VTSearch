# A figure script's own loader read 34× the rows the analyzer sees

**Study:** #2808 linear-head convergence. **Cost:** caught by a row count printed
during the first figure run; the figures were regenerated before anything was
committed to a report.

`make_linhead_figs.py` opened the cell CSVs itself and dropped "tagged"
counterfactual rows with a private filter that guessed at the tag column names.
The analyzer's `load_arm` sees **42,003** base rows; the private loader loaded
**1,411,458** — every counterfactual variant row the cells carry beside each
arm's base row, ~32 per step.

The figures still rendered. Curves still descended, arms still separated, error
bands still looked plausible. Only the row count betrayed it, and only because
the script happened to print one.

**The generalisation:** a figure script that re-implements loading is a second
source of truth for the same numbers, and the two only have to agree the first
time to look permanently trustworthy. The stated goal of these scripts — "a
figure and a table can never disagree" — is only enforceable if the figure and
the table go through *the same loader*.

**Status: prevented for this study, advice in general.**
`make_linhead_figs.py` now imports `analyze_spikes.load_arm` instead of
re-implementing it, and prints the dropped-cell provenance (`unreadable`,
`zero_byte`, `no_positive_found`) that loader returns. Any new figure script
should import the analyzer's loader rather than open CSVs directly; there is no
mechanical check for this today.
