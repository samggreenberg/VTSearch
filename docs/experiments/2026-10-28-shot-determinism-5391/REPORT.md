# The container draws every shot the same, in a newer Chromium than the GRID's — 2026-10-28 (#5391)

**Two full `scripts/screenshots/check.sh` runs on the cloud container gave all 162 shots
byte-identical to each other and to the committed baselines, `autofind-tab.light` and
`export-server-csv.dark` included. The container is deterministic for the Detectors header, so the
baseline is a reference, not a sample, and the GRID's 522 px (#5385) is a consistent difference. The
two machines do not run the same browser. The container draws in Chrome for Testing 153.0.8010.12,
the full build, which `scripts/screenshots/launch.mjs` falls back to because the build Playwright
pins (148.0.7778.96) is not installed there. The GRID drew in that pinned 148 build. No GRID render
has used the container's build yet, and that comparison is what is owed next.**

**GRID update, 2026-10-29: in 153 the GRID draws `autofind-tab.light` byte for byte like the
baseline. The version was the cause. `export-server-csv.dark` keeps its 220 px in every browser.
The full build also needs a fontconfig without hinting on the GRID. See
[On the GRID, in 153](#on-the-grid-in-153-2026-10-29).**

| Run | Browser | Harness | Shots byte-identical to the baselines | `autofind-tab.light`, px past 0% / 5% | `export-server-csv.dark` |
|---|---|---|---:|---:|---|
| 1 | 153, full build | `dev` | 162 of 162 | 0 / 0 | identical |
| 2 | 153, full build | `dev` | 162 of 162 | 0 / 0 | identical |
| C | 153, full build | without `--font-render-hinting=none` | 162 of 162 | 0 / 0 | identical |
| A | 153, headless shell | `dev` | 59 of 162, the rest within 11 px past 5% | 36 382 / 5 | identical |
| B | 153, full build | the Detectors card moved to a whole CSS px | 160 of 162 (not the `autofind-tab` pair) | 101 551 / 19 593 | identical |

Every run was a full `check.sh`, in manifest order, on a fresh app data dir. Each was a fresh
render. A subset run changes panel widths, so it does not count. Each took 24–26 minutes. Pixel
counts are ImageMagick's, counted from `compare`'s mask as `check.sh` counts them. The runs used
`dev` at af608171a. The commits `dev` gained since touch neither the app's frontend nor the
screenshot harness.

## The container does not vary from run to run

Runs 1 and 2 differ in no byte of any of the 162 renders. Arms C and B each reproduced, byte for
byte, every shot they did not set out to change. That makes four full runs with the default browser
that agree. They also agree with the committed baselines, which the 2026-10-28 release drew
(20925b615):

| Render | sha256 |
|---|---|
| `autofind-tab.light.webp`: runs 1 and 2, and the baseline | `14016baa9e3e5ca0b65b2fd394d4026fccc4abdcf67b99b9a6026f56fe1cd4dc` |
| `export-server-csv.dark.webp`: runs 1 and 2, and the baseline | `bb5b35341316cb081d85beaa00f7dd452d2095655b556f8ee4ec46bca3d69fcf` |

The renders are in [`renders/run1/`](renders/run1/) and [`renders/run2/`](renders/run2/), as #5391
asked. Since they are the baselines, a GRID session can just as well compare with
`docs/user/assets/` on `dev`, which is what `check.sh` does.

## The gap between the 10-26 and 10-28 baselines is not the hinting flag

#5385 found the container's own light baselines of this header differ between 10-26 (df9e45397)
and 10-28 (20925b615). That pair brackets the commit that added `--font-render-hinting=none`
(#5379). Arm C drops the flag, and every shot comes out byte-identical to the 10-28 baselines. So on
the container the flag changes nothing, as #5379 assumed when it added it ("as the container draws
by default"). Nor is the gap noise, since four runs agree. It came from something else that
changed between the two commits: that release drain moved 53 shots, mostly through the callouts'
typeface. This study did not chase it further. One clue is that the Datasets header, at a whole y,
also differs between the two baselines at 0% fuzz:

| 10-26 vs 10-28 baseline, light | past 0% | past 5% |
|---|---:|---:|
| Detectors header row | 3 923 | 107 |
| its TYPE and # TRAINING labels | 818 | 18 |
| Datasets header row (y = 128) | 1 021 | 4 |

The boxes, in ImageMagick geometry: the Detectors header row `1120x32+160+503`, its TYPE and #
TRAINING labels `167x32+845+503`, the Datasets header row `1120x32+160+128`. Each stops short of the
callout ring under it. The GRID's figures of 89 px and 1 382 px came from its own box.

## The two machines run different Chromium builds

On the container, Playwright 1.60.0 asks for revision 1223 (148.0.7778.96), which the image does not
carry. `launchChromium()` then takes the newest build under `/opt/pw-browsers`, preferring the full
one over the headless shell, and every run logged
`using /opt/pw-browsers/chromium-1243/chrome-linux64/chrome`. That is Chrome for Testing
153.0.8010.12. The image's browsers have been in place since 2026-10-22, before both baselines,
which fits run 1 matching all 162 of them.

The GRID's arms in #5385 drew in the pinned 148 headless shell, and its "full chrome" arm was 148's
full build (revision 1223), not the container's 153. Arm A puts the binary type alone to the test, on
one machine: the container's own 153 headless shell against its 153 full build. The two agree within
11 px past 5% on every shot. The Detectors header agrees exactly past 5% (376 px differ at 0%), and
`export-server-csv.dark` is byte-identical. So at 153 the binary type barely matters, and the
version, 148 against 153, is the leading suspect for both of #5385's residuals.
[`renders/shell-153/`](renders/shell-153/) holds arm A's `autofind-tab.light`.

Two harness fixes came out of this. Until now `CHROMIUM_PATH` was read only in the fallback, so a
machine that has the pinned build, such as the GRID, could not draw in any other.
`launchChromium()` now honours it first. And `capture.ts` names the build it launched
(`[launch] Chromium 153.0.8010.12`), so a run's log says what drew it.

## The fractional y (step 4)

In this frame the Detectors card's top sits at y = 443.796875 CSS px and its header at 502.796875.
That is 51/64 of a pixel past a whole one in both: the header sits at a whole offset inside the
card. The card most likely takes its fraction from the two dashboard panels, which split the
viewport's height between them (`flex: 1 1 0`). Arm B adds a harness-only nudge to the recipe, `position: relative;
top: 0.203125px` on the card ([`nudge.patch`](nudge.patch), against the manifest only), which put
the card at 444 and the header at 503.

Moving the whole card by 13/64 of a pixel redraws every edge and glyph in it, so the nudged
`autofind-tab` differs from today's baseline by 19 593 px past 5% (light) and 23 291 px (dark). That
makes it a new reference, not something to compare with the committed shot. Even the Datasets
header, which did not move, differs at 0% (705 px, 4 past 5%), so the raster of text depends on more
than the text's own position. The other 160 shots stayed byte-identical. Arm B ran once. Run-to-run
stability is already settled by the four runs above. [`renders/nudge/`](renders/nudge/) holds both
themes.

## On the GRID, in 153 (2026-10-29)

Four full `check.sh` runs on the GRID, `dev` at cffe6773a, each drawn in Chrome for Testing
153.0.8010.12, which Playwright 1.63.0 pins (revision 1243) and the container already carries. Every
log printed `[launch] Chromium 153.0.8010.12`. All four ran on AMD EPYC 7713 nodes (rack8n09,
rack8n10). [`grid-check.sbatch`](grid-check.sbatch) is the job script. Its argument names the
arm, and a second worktree keeps two concurrent arms from sharing `refresh.sh`'s app data
directory.

| Arm | Browser | Fontconfig | Shots that pass `check.sh` | `autofind-tab.light`, px past 5% | `export-server-csv.dark`, px past 5% |
|---|---|---|---:|---:|---:|
| a | 153, full build | the GRID's own (`hintstyle` slight) | 20 of 162 | 8 198 | 27 210 |
| b | 153, full build, on another node | the GRID's own | 20 of 162 | 8 198 | 27 210 |
| c | 153, full build | #5385's `pinned.conf` (no hinting) | **161 of 162** | **0** | 220 |
| d | 153, headless shell | the GRID's own | **161 of 162** | **0** | 220 |

**The version was the header's cause. In 153, the GRID draws `autofind-tab.light` byte for byte
like the container** (sha256 `14016baa…`, the committed baseline), with no nudge, so step 2 was not
run. Arms c and d agree with each other in every byte of all 162 renders, and arms a and b agree
with each other in every byte as well. So the GRID is deterministic in 153, as it was in 148.

**The full build ignores `--font-render-hinting=none` on the GRID.** Arms a and b drift on 142
shots (median 7 718 px past 5%, at most 29 515). The difference is the anti-aliasing of every
glyph and border, with layout unchanged: the same symptom #5302 found for 148 before the flag.
The full build takes the GRID's fontconfig hinting regardless of the flag. The headless shell
honours the flag (arm d), and the full build matches once fontconfig itself turns hinting off (arm
c). The container's fontconfig evidently does not hint, which is why arm C there found the flag a
no-op. [`renders/grid-153-full/`](renders/grid-153-full/) holds arm a's two shots. Its
`autofind-tab.light` differs from the baseline in 116 031 px at 0%, 151 of them past 5% in the
`TYPE` / `# TRAINING` labels' box.

**`export-server-csv.dark` does not depend on the browser.** Its 220 px past 5% are the same in
every arm, and arm c's render is byte-identical to the GRID's 148 render from #5378
(`/expscratch/sgreenberg/runs/5378/renders/`). It is the same #5385 residual: edge anti-aliasing
on the step-1 callout's outline and on one dimmed thumbnail's left edge, invisible at 4x. A browser
change will not move it.

What follows, for #5385: drawing in 153's headless shell on both machines (Playwright 1.63.0, whose
pin the container already holds) leaves one residual, `export-server-csv.dark`, instead of two. A
full build on a hinting machine needs a fontconfig without hinting as well.

## Setup

Cloud container: Intel Xeon @ 2.80 GHz, 4 vCPUs with AVX-512, kernel 6.18.44, ImageMagick
6.9.12-98, Node 22.22.0, Pillow 13.0.0, Playwright 1.60.0. The browsers were
`/opt/pw-browsers/chromium-1243/chrome-linux64/chrome` and, for arm A,
`/opt/pw-browsers/chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell`.
Both report 153.0.8010.12. The renders were kept by running a copy of `check.sh` whose temp directory
is not deleted on exit. Nothing else in it changed.
