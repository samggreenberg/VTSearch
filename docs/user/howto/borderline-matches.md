# Catch the borderline matches

Every detector draws a line: pictures above it are matches, pictures below
are not. Some real matches always land just under the line. Find already
shows you pictures from both sides of it, and moving the **Threshold**
toward **False Positives** moves the line down to let the next band of
pictures in. This page shows how to review the pictures near the line, move
the Threshold, and see what that costs you.

It picks up where [Step by step: your first search](../USER_GUIDE.md#step-by-step-your-first-search)
ends: the `Yellow Smileys` detector, trained on `drawings`, has just been run
over `drawings-new` with **Find**. The red numbers in each screenshot show
where to click, in order.

## How the Threshold moves the line

The **Threshold** sits at the top of the left-hand panel: a spectrum from
**False Positives** to **False Negatives**, with three radio buttons under
it. Each radio is a **balance** of precision and recall - how many wrong
pictures you will take in the results against how many real matches you will
accept missing - and the line is drawn where that balance is best. Toward
False Negatives it returns only the pictures most likely to be matches, and
misses more; toward False Positives it returns the most, with more wrong ones
among them; the middle radio weighs the two mistakes equally. The line keeps
the top of the ranking, among the pictures you haven't checked: up to the
top 32 on the middle and False Negatives radios, and up to the top 128 on the
False Positives radio, fewer when the detector's own estimate says the
balance peaks sooner. So moving toward False Positives moves the line down by
a band of borderline pictures, and everything the line kept before it still
keeps.

The note under the spectrum says what the line keeps, and what a check found
on it:

- **Top 32 kept, unchecked** - the Threshold's starting set. Nothing has
  measured it yet. A Threshold that keeps a different count moves the line
  straight away.
- **Checked · likely 55–80% right, about half of them found (checked 15) ·
  48 kept** - a spot check in Train has measured it: it walked the list and
  ended on the set where the balance peaked, and the note says how much of
  that set is likely right and how much of what the list holds it likely
  found; see
  [How close the line got](../USER_GUIDE.md#how-close-the-line-got). Find
  offers no check of its own; it is where you test the Threshold.

Moving the Threshold never re-scores anything and never changes the order of
the pictures. Only the line moves. The user guide explains the line itself in
[Matches, the line, precision and recall](../USER_GUIDE.md#matches-the-line-precision-and-recall).

## Step 1: Check the pictures either side of the line

Check the pictures near the line first, as in
[Check and correct a detector's calls](check-and-correct.md). Find serves
them from both sides, alternating above and below the line, so the real
matches just under it come up wherever the Threshold sits. Every one you mark
**Good** joins **Verified Good**, and counts as a match from then on.

## Step 2: Move the Threshold toward False Positives

At the top of the left-hand panel:

1. Click the radio under the **False Positives** end of the spectrum: from
   the middle radio, that is the one to its left.
2. Read the note under it. **Top 128 kept, unchecked** means the line has
   moved down to keep the top 128, so more pictures sit above it and the count
   of **Unverified Good** on the right grows by the pictures it has just let
   in.
3. Find the line in the list: the pictures just above it are the ones the
   move let in.

Checking the pictures in the band, as Step 3 does, is how you learn how much
of that longer list is right here. Find has no spot check: that is for
setting the Threshold in Train, before you test it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/borderline-floor.dark.webp" />
  <img src="../assets/borderline-floor.light.webp" alt="Step 2: (1) the Threshold moved toward False Positives, (2) the note under it, which says how many pictures the line keeps now, (3) the line in the list" width="720" />
</picture>

## Step 3: Review the pictures it let in

When the line has moved, carry on checking with **Good** and **Bad** as
before. Find now starts from the bottom of the band it just let in, the
lowest-scoring picture still above the new line, and works outwards from
there. Nothing on screen marks which pictures are new: they are the ones just
above the line, and every one you check moves to the right-hand panel.

Stop when the pictures above the line stop being matches. If you see only
misses, go back to the radio you had. If you are still finding real matches,
there is no further step toward False Positives: the line already keeps the
most it can.

## Step 4: See the trade-off

1. Click **Stats** <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/icon-stats.dark.webp" /><img src="../assets/icon-stats.light.webp" alt="The Stats button in the Find view" height="24" /></picture>, the pie-chart button at the top of the **Verified Good** pile.
2. Scroll to **Precision by Number Returned**. It reads down the ranked list:
   for the top N pictures, how many of them are real matches. Returning more
   (to the right) catches more matches, but the share that are right falls.
   The upright line is where your **Line** is now, and the legend under the
   chart says whether it was checked (**Line: checked (48 kept)**, or
   **Line: the top 32, unchecked**). Moving the Threshold toward False
   Positives moves the Line to the right. Once a spot check has run in Train,
   a bar stands on the Line: the check's likely range for how much of the
   list is right.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/borderline-chart.dark.webp" />
  <img src="../assets/borderline-chart.light.webp" alt="The Precision by Number Returned chart for the top N pictures, with the Threshold drawn across it, the line marked, and the line under the chart reading it there" width="720" />
</picture>

The chart's curve, **Checked by you**, counts only the pictures you have
checked. You check the ones near the line, where the detector is least sure,
so it can read lower than the matches as a whole. The chart makes no guess
about the pictures nobody checked: the bar on the Line is the only measure of
those, and only a spot check draws it.

Point at the chart to read the curve at any count; with the pointer off it,
the line under the chart reads it at the Line. The chart is drawn when the Stats
window opens, so close it and open it again after you move the Threshold.

## Where the setting goes

The detector keeps its Threshold while VTSearch runs, and the Threshold
decides where the line sits the next time you run Find with this detector, on this dataset
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
- [Manual mode](../USER_GUIDE.md#3-threshold), in the user guide,
  describes the same Threshold while you train, and the spot check that
  measures it.
