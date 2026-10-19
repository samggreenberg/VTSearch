# 2026-10-19 — `export -p` copied tokens into a world-readable file (#4278)

**Cost:** an `HF_TOKEN` and Claude Code session tokens sat in two world-readable files on `/expscratch` for
about a minute.

**What broke.** To run queued cells exactly as the launcher would, #4278 captured the launcher's environment
with `export -p` into the study's work directory. The login shell exports secrets, and the directory's default
umask made the files world-readable. They were made owner-only and stripped of every token within a minute.

**Prevented?** *Advice only.* Filter before writing (drop `*TOKEN*`, `CLAUDE*`, `SSH_*` and the like) and
create the file with `umask 077`. A job launched with `--export=ALL` already inherits the submitting shell's
secrets, so a captured environment never needs them.
