# Catch the borderline matches

Every detector draws a line: pictures above it are matches, pictures below
are not. Some real matches always land just under the line. Find already
shows you pictures from both sides of it, and lowering the **precision
floor** moves the line down to let the next band of pictures in. This page
shows how to review the pictures near the line, lower the floor, and see what
that costs you.

It picks up where [Step by step: your first search](../USER_GUIDE.md#step-by-step-your-first-search)
ends: the `Yellow Smileys` detector, trained on `drawings`, has just been run
over `drawings-new` with **Find**. The red numbers in each screenshot show
where to click, in order.

## How the floor moves the line

The floor sits at the top of the left-hand panel and reads **Lean:
Centered**. It offers three: **Correct** returns only the pictures most likely
to be matches, **Complete** returns as many as it can while accepting more
misses among them, and **Centered** sits between the two. The line keeps the
top of the ranking, among the pictures you haven't checked: the top 32 at
**Centered** and **Correct**, and the top 128 at **Complete**. So leaning
toward **Complete** moves the line down by a band of borderline pictures, and
everything the line kept before it still keeps.

The note under the picker says what the line keeps, and how close it got:

- **Top 32 kept, unchecked** - the floor's starting set. Nothing has measured
  it yet. A floor that keeps a different count moves the line straight away.
- **Confirmed · likely 55–100% right (checked 5) · 32 kept**, or **Aimed at
  Centered: likely 19–92% right (checked 5) · top 32 kept** - a spot check has
  measured it. **Check 5 picks**, beside the note, runs one: see
  [How close the line got](../USER_GUIDE.md#how-close-the-line-got).

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

1. Pick a lower floor: from **Centered**, that is **Complete**.
2. Read the note under it. **Top 128 kept, unchecked** means the line has
   moved down to keep the top 128, so more pictures sit above it and the count
   of **Unverified Good** on the right grows by the pictures it has just let
   in.
3. Find the line in the list: the pictures just above it are the ones the
   lower floor let in.

To know how much of that longer list is right, click **Check 5 picks** and
answer the random picks it shows you: at **Complete** a check can take up to
three rounds (see
[How close the line got](../USER_GUIDE.md#how-close-the-line-got)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/borderline-floor.dark.webp" />
  <img src="../assets/borderline-floor.light.webp" alt="Step 2: (1) the floor lowered to Complete, (2) the note under it, which says how many pictures the line keeps now, (3) the line in the list" width="720" />
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
   where your **Line** is now. Lowering the floor moves the Line to the right.
   Once you have run a spot check, a bar stands on the Line where it meets
   the Floor: the check's likely range for how much of the list is right.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/borderline-chart.dark.webp" />
  <img src="../assets/borderline-chart.light.webp" alt="The Precision by Number Returned chart for the top N pictures, with the floor drawn across it, the line marked, and the line under the chart reading it there" width="720" />
</picture>

The chart has two curves:

- **Estimated (at least)** is VTSearch's cautious estimate, worked out from the
  detector's own answers. It shows the shape of the trade-off; the line
  itself is measured by the spot check, never by this curve. It needs enough
  **Good** answers to test itself on; until then the curve is missing, and
  the chart says how many more it needs.
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
