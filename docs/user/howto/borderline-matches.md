# Catch the borderline matches

Every detector draws a line: pictures above it are matches, pictures below
are not. Some real matches always land just under the line. Find already
shows you pictures from both sides of it, and once the detector can promise
its **precision floor**, lowering the floor moves the line down to let the
next band of pictures in. This page shows how to review the pictures near the
line, lower the floor, and see what that costs you.

It picks up where [Step by step: your first search](../USER_GUIDE.md#step-by-step-your-first-search)
ends: the `Yellow Smileys` detector, trained on `drawings`, has just been run
over `drawings-new` with **Find**. The red numbers in each screenshot show
where to click, in order.

## How the floor moves the line

The floor sits at the top of the left-hand panel and reads **At least 50%
right**. It offers **25%**, **50%**, **75%** and **90%**: how much of what
the detector returns should be right. The line returns as many pictures as it
can while at least that share of them is estimated right, so a lower floor
returns more. While the floor is promised, lower floors *nest*: everything
the line returns at 75% it still returns at 50%, along with a band of extra
borderline pictures.

The line only follows the floor once the detector can promise it. The note
under the picker says whether it can:

- **At least 50% right**, with a count - the line keeps the floor, and moving
  the floor moves the line.
- **Not enough evidence yet** or **Can't reach 50% on this dataset** - the line
  stays at the default cut whatever floor you pick, and is marked
  *unpromised*. A new detector starts here: see
  [When the line is unpromised](../USER_GUIDE.md#when-the-line-is-unpromised)
  for what brings the promise.

Moving the floor never re-scores anything and never changes the order of the
pictures. Only the line moves. The user guide explains the line itself in
[Matches, the line, precision and recall](../USER_GUIDE.md#matches-the-line-precision-and-recall).

## Step 1: Check the pictures either side of the line

Check the pictures near the line first, as in
[Check and correct a detector's calls](check-and-correct.md). Find serves
them from both sides, alternating above and below the line, so the real
matches just under it come up whatever the floor says. Every one you mark
**Good** joins **Verified Good**, and counts as a match from then on.

## Step 2: Lower the floor

At the top of the left-hand panel:

1. Pick a lower floor, such as **25%**.
2. Read the note under it. **At least 25% right** means the line has moved
   down, so more pictures sit above it and the count of **Unverified Good**
   on the right grows by the pictures it has just let in.
3. **Not enough evidence yet**, as here, means the line in the list has
   stayed where it was: keep checking pictures as in Step 1, which is also
   what earns the promise.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/borderline-floor.dark.webp" />
  <img src="../assets/borderline-floor.light.webp" alt="Step 2: (1) the floor lowered to 25%, (2) the note under it, which says whether the line has moved, (3) the line in the list" width="720" />
</picture>

## Step 3: Review the pictures it let in

When the line has moved, carry on checking with **Good** and **Bad** as
before. Find now starts from the bottom of the band it just let in, the
lowest-scoring picture still above the new line, and works outwards from
there. Nothing on screen marks which pictures are new: they are the ones just
above the line, and every one you check moves to the right-hand panel.

Stop when the pictures above the line stop being matches. If you see only
misses, go back to the floor you had. If you are still finding real matches,
lower the floor another step and keep going.

## Step 4: See the trade-off

1. Click **Stats** <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/icon-stats.dark.webp" /><img src="../assets/icon-stats.light.webp" alt="The Stats button in the Find view" height="24" /></picture>, the pie-chart button at the top of the **Verified Good** pile.
2. Scroll to **Precision by Number Returned**. It reads down the ranked list:
   for the top N pictures, how many of them are real matches. Returning more
   (to the right) catches more matches, but the share that are right falls.
   The solid line across the chart is your **Floor**, and the upright line is
   where your **Line** is now: dashed when it is the unpromised default cut.
   Lowering a promised floor moves the Line to the right.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/borderline-chart.dark.webp" />
  <img src="../assets/borderline-chart.light.webp" alt="The Precision by Number Returned chart for the top N pictures, with the floor drawn across it, the line marked, and the line under the chart reading it there" width="720" />
</picture>

The chart has two curves:

- **Estimated (at least)** is VTSearch's cautious estimate, worked out from the
  detector's own answers. It is the estimate the floor is kept by. It needs
  enough **Good** answers to test itself on; until then the curve is missing,
  and the chart says how many more it needs.
- **Checked by you** counts only the pictures you have checked. You check the
  ones near the line, where the detector is least sure, so it can read lower
  than the matches as a whole.

Point at the chart to read both at any count; with the pointer off it, the
line under the chart reads them at the Line. The chart is drawn when the Stats
window opens, so close it and open it again after you move the floor.

## Where the setting goes

The detector keeps its floor while VTSearch runs, and the floor decides where
the line sits the next time you run Find with this detector, on this dataset
or any other. The line decides which unchecked pictures count as matches when
you **Export** or use **To Dataset**
([Send your matches somewhere](export-matches.md)).

Pictures that crossed the line when it moved count as *corrections* (the
detector called them one way, and the line now calls them the other). If you
then click **Add Corrections to Detector**, they are handed to the detector
along with the ones you checked by hand.

## Where next

- [Decide how far to trust a detector](trust-a-detector.md): the rest of the
  **Stats** window.
- [Manual mode](../USER_GUIDE.md#3-precision-floor), in the user guide,
  describes the same floor while you train.
