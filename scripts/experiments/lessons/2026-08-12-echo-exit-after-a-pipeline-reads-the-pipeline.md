# 2026-08-12 — `echo "exit=$?"` after a pipeline reads the pipeline (#3129)

**What happened.** The playbook already says to check a commit's own exit code.
It was checked as `git commit -q -m "..." 2>&1 | tail -3; echo "commit exit=$?"`,
which reports **`tail`'s** status. It printed `exit=0` while the pre-commit hooks
had rewritten the files and failed the commit; `git push` then pushed nothing and
`git log` still showed the previous head.

Also on the same run: `git commit -F /tmp/commitmsg2.txt` picked up a **stale
file from an earlier session** and committed the report under an unrelated #2877
message, which had to be amended.

**Prevented?** *Advice only, but sharpened.* Never pipe the command whose status
you are about to read — run it bare, then `echo $?`, then inspect `git log -1`
to confirm the head actually moved. Use a run-unique path for `-F` message files
(`/tmp/msg-<jobid>.txt`); `/tmp` on a shared login node is not yours alone.
