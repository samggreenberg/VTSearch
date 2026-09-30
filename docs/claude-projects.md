# Claude.ai projects for VTSearch

One-time setup for a [Claude.ai project](https://code.claude.com/docs/en/claude-projects) that works on this repository. The rule that governs it — **one project, this repository alone** — and why a second repository silently disables the repo's hooks live in [`CLAUDE.md`](../CLAUDE.md#claudeai-projects-one-project-one-repository-critical); this page is the setup procedure behind it.

`CLAUDE.md` is read by every thread from its clone, so it stays the home
for repo rules. Project memory (`MEMORY.md`, written by Claude, edited in **Project
settings > Memory**) is for notes about the *project*; do not restate repo rules there.

**Three project defaults contradict rules in `CLAUDE.md`, and project instructions must
override each one:**

- **Threads branch from the repository's default branch**, which is `main`. CLAUDE.md's Branch
  Policy requires `dev`. The SessionStart hook corrects this — but only because the
  project is single-repo, so state `dev` in the project instructions too.
- **A thread that opens a PR watches it with auto-fix on**, "whether or not auto-fix
  is on for your other cloud sessions." That is exactly what CLAUDE.md's *PR Activity
  Subscription (do not ask)* forbids, and it also spends plan quota every time CI
  fails on an otherwise-idle thread. Instruct threads to stop watching after opening
  the PR.
- **A new project runs every thread on Opus at high effort.** Size it to the work
  instead, per CLAUDE.md's *Recommend a Claude model in every issue you file* — the same ladder
  applies.

**These are settings outside this repository, so paste the override text.** Nothing in
`CLAUDE.md` and no hook can enforce the three bullets above: they live in the project,
not the clone, and take effect only once this is in **Project settings > Memory >
Project instructions**. Paste it verbatim when creating a VTSearch project.

```text
Base every branch on dev, not main, and target dev with every pull request.

After you open a pull request, stop watching it. Do not call subscribe_pr_activity,
and if auto-fix is on for your pull request, turn it off. Report the PR number in
your final message and stop; I review and merge pull requests myself.

Choose the model per thread, sized to the work; do not leave every thread on Opus.
```

**Then verify the watch half rather than assuming it.** Project instructions are not
an enforced setting, and auto-fix is turned on *around* the thread, so a thread that
never calls `subscribe_pr_activity` can still end up watching its own PR. Check once,
after pasting: open a throwaway PR from a thread and confirm the thread's CI status bar
at claude.ai/code shows **Auto-fix** cleared. A watch left on pushes commits and review
replies under the owner's account and wakes the idle thread — spending plan quota —
on every review comment.

If it does not stick, give up the *Auto-PR* convenience for threads rather than the
rule: instruct them to push the branch and report the compare URL instead of opening
the PR. That costs a step and is enforceable by instruction, which a toggle Claude does
not own may not be.
