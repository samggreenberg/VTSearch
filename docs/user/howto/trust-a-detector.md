# Decide how far to trust a detector

A detector is only as good as the examples it learned from. Point it at
pictures unlike any of them and it still gives an answer for every one; it
just has nothing to base it on. The **Stats** window in Find tells you how
often that is happening, so you know which calls to check by hand before you
rely on them.

This page picks up where [Step by step: your first search](../USER_GUIDE.md#step-by-step-your-first-search)
ends: the `Yellow Smileys` detector, trained on `drawings`, has just been run
over `drawings-new` with **Find**. It is at its most useful for a detector
someone *else* trained, since you don't know what that one has seen. The red
numbers in each screenshot show where to click, in order.

## Step 1: Open the Stats

In Find, click **Stats** <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/icon-stats.dark.webp" /><img src="../assets/icon-stats.light.webp" alt="The Stats button in the Find view" height="24" /></picture>, the pie-chart button at the top of the
**Verified Good** pile. The **Detector Stats** window opens.

## Step 2: Read the two trust checks

1. **Compare against**: pick the dataset the detector was trained on
   (`drawings` here). VTSearch can't tell which one that was, so it asks.
2. **Training-domain overlap** says how much of *this* dataset looks unlike
   the one you picked, for example "**12%** of this dataset looks atypical vs
   **drawings** — largely in-domain".
3. **Evidence coverage** says how much of the dataset the detector is calling
   with no labelled example anything like it behind the call, for example
   "**8%** of this dataset sits in an evidence vacuum — mostly backed by
   labeled evidence".

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/trust-stats.dark.webp" />
  <img src="../assets/trust-stats.light.webp" alt="Step 2: in Detector Stats, (1) Compare against the training dataset, (2) the share of this dataset that looks unlike it, (3) the share the detector calls with no labelled example behind it" width="720" />
</picture>

The two checks answer the same question from different ends:

- **Training-domain overlap** compares the *pictures*. It needs the training
  dataset loaded, and it only lists datasets set up with the same embedder as
  this one; if there are none, the section is not shown. A large share means
  the detector is being asked about a different kind of picture from the
  ones it was trained on (domain shift).
- **Evidence coverage** compares the *detector's own examples* with this
  dataset, so it works even when you don't have the training data. A large
  share means many calls are guesses.

The verdict after each figure (*largely in-domain* or *likely domain shift*;
*mostly backed by labeled evidence* or *often calling without support*) is
VTSearch's reading of it. The line under each gives the numbers behind it.

## Step 3: Read what you have already checked

Lower down, **Detector Accuracy** sets the detector's calls against the
answers the dataset has now: the ones you checked by hand, and the detector's
own call for the rest.

- **Kept Good** / **Detector Good**: matches you both agree on.
- **Marked Bad** / **Detector Good**: wrong matches you took out.
- **Kept Good** / **Detector Bad**: matches the detector missed and you
  rescued.
- **Agreement rate** and **Kept rate** sum these up.

Until you have checked some pictures ([Check and correct a detector's calls](check-and-correct.md)),
every answer is the detector's own and it agrees with itself completely. The
figures start to mean something once you have checked a few dozen pictures
near the line.

## Step 4: Act on it

- **Both shares small, accuracy high**: trust the unchecked calls and send the
  matches on ([Send your matches somewhere](export-matches.md)).
- **A large share atypical, or in an evidence vacuum**: the detector needs
  examples from this kind of picture. Check a few dozen pictures by hand, hand
  them to the detector with **Add Corrections to Detector**, and run Find
  again ([Check and correct a detector's calls](check-and-correct.md)). That
  teaches it more than moving the **Inclusion** line would.

There is no way to list just the flagged pictures: the figures are a count,
not a selection. The pictures nearest the detector's line are the ones to
check first.

## Where next

- [Catch the borderline matches](borderline-matches.md): the chart at the
  bottom of the same window.
- [How far to trust the score](../USER_GUIDE.md#how-far-to-trust-the-score),
  in the user guide.
