# 2026-08-12 — heredocs through ssh (#3129)

**What happened.** Three separate attempts to send a Python or bash snippet
through `ssh grid '... <<EOF ...'` were mangled by shell quoting — one silently
produced a syntactically invalid script that only failed at run time.

**Prevented?** *Advice only.* Write the script to a file locally and `scp` it.
The round trip is cheaper than one debugging cycle, and the file is then a
reviewable artifact rather than a string inside a command.
