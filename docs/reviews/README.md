# VTSearch Reviews & Audits

Point-in-time reports from hands-on UI driving and codebase audits. Each
file is dated and stands on its own; screenshots live under
[`assets/<filename>/`](assets/). Findings carry stable IDs (`V#` visual,
`U#` UX, `O#` observability, `B#` bug, plus per-report slugs) so they can
be referenced from plans and PRs.

These are **not** deleted when findings are fixed: a fixed finding is
struck through in place so the report stays an accurate record of what was
seen on that date. Forward-looking design work lives in
[`../plans/`](../plans/README.md).

| Report | Date | What it covers |
| --- | --- | --- |
| [2026-05-27-style-audit.md](2026-05-27-style-audit.md) | 2026-05-27 | Rendered style audit, light/dark, every major view (V7/V10/V11/V13 fixed; rest open) |
| [2026-05-28-edge-states.md](2026-05-28-edge-states.md) | 2026-05-28 | Empty / edge-state sweep (cold boot, long names, single-item, errors) — open |
| [2026-05-28-e2e-flows.md](2026-05-28-e2e-flows.md) | 2026-05-28 | End-to-end flow walkthroughs + UX friction — open |
| [2026-05-28-longops.md](2026-05-28-longops.md) | 2026-05-28 | Long-running-op observability (progress, cancel, gates) — open |
| [2026-06-04-standard-workflow.md](2026-06-04-standard-workflow.md) | 2026-06-04 | Standard "train here, find there" workflow audit (5 fixed; ~10 polish items open) |

The methodology that produced the four 2026-05 reports is the reusable
playbook in [`../plans/browser-vision-testing.md`](../plans/browser-vision-testing.md).
