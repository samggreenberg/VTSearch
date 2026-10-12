# Toasty's hints

Every hint King Toasty gives (#4680, #4807), by its id: the name `hidden_hints`
stores when a user ticks **Hide this hint**. One id is one bubble with one
text. A hint that would say something else in another state is a second hint,
with a condition disjoint from the first. The ids are `HINT_IDS` in
`frontend/src/app/services/hints.service.ts`, and
`tests_lib/meta/test_hints_doc.py` fails when this chart and the templates
disagree on the ids, a hint's Toasty, or its text.

**Toasty** is the face the hint shows: **happy** for a next step, **surprised**
when something needs fixing. No hint uses the **sad** face; that one is for
error toasts. **Text** is what the bubble says, with the bold words in bold;
`{…}` is filled in at runtime (see the notes under each table).

**What a hint says.** It names the state that raised it, then the one step to
take, so it reads as an answer to where the user is rather than an order out
of the blue ("You have detectors, but none is selected. Click one…", never
"Now click a detector"). It stops short of the step after that, and a short
hint earns a clause of why. `add-dataset` alone opens with "Start here!": its
state is the app's opening state.

**One at a time.** Only one hint is ever on screen. Each view shows at most
one (the Dashboard's ladder below, the Train view's vote hint or hand-off, the
Test view's stage or Review hint), and `ToastyHintComponent` shows a hint
inside a dialog only while that dialog is the topmost one open, and a hint on
a view only while no dialog is open at all. So opening any dialog, a confirm
prompt included, hides the view's hint, and stacking a second dialog hides the
first one's. A dialog's hint stands below its box, reaching right, or beside
the box's top corner when the box fills the window.

## Dashboard

At most one at a time: the first row, top to bottom, whose condition holds
(`dashboardHint` in `dashboard.component.ts`). So each row also means "and no
row above it fires". None shows while Add Dataset or New Detector is open, or
while the Dashboard is switching context. While one is on show, the reason
under a disabled Train / Test / Find button is hidden: Toasty is saying it.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `add-dataset` | happy | Start here! Click **+** to add a dataset: the {media} you want to train or search on. | There are no datasets, and none is importing. |
| `select-dataset` | happy | You have data, but none is selected. Select the dataset you want to search over, or click **+** to add a new one. | There are datasets, but none is selected. |
| `mixed-datasets` | surprised | The datasets you selected hold different kinds of media, so they can't be searched together. Select datasets of one kind. | The selected datasets hold more than one media type. |
| `add-detector` | happy | You need a detector to do the searching. Click **+** to add one and say what you're looking for. | There are no detectors. |
| `empty-detector-tab` | happy | The {tab} tab is empty, but you have detectors on the {other} tab. Click **{other}** to select one, or click **+** to make a new one. | There are detectors, but the open tab (Drafts or AutoFind) lists none of them, so none is selected. Stands under the tab bar. |
| `select-detector` | happy | You have detectors, but none is selected. Click one to select it, or click **+** to make a new one. | The open detector tab lists detectors, but none is selected. |
| `mismatch` | surprised | The detector you selected is for a different kind of media than your dataset, so they can't work together. Select ones that match, or click **+** to make a detector for this dataset. | The selected datasets and detectors are not all one media type. |
| `wait-for-import` | happy | Your dataset is still importing. It joins the Datasets table when it's done, so select it then. | A detector is selected and the only dataset is still importing, so there is no dataset row to select yet. |
| `train` | happy | Your new detector has no training labels yet. Click **Train** to teach it by marking examples Good or Bad. | One detector with no training labels is selected, beside one dataset, so Train is enabled. |
| `train-one-dataset` | surprised | You selected {N} datasets, but a new detector trains on one at a time. Select just one, then click **Train** to teach it. | Several datasets are selected with one detector that has no labels and no typed description, so nothing is enabled. |
| `untrained-detector` | surprised | **{detector}** has no training labels yet, so it can't search. Deselect it, or select it alone with one dataset and click **Train**. | Several detectors are selected and one of them has no labels and no typed description (#4758), so Test and Find are off. |
| `test-or-find` | happy | Your detector is trained. Click **Test** to measure it on this dataset, or **Find** to collect its matches. To search new media, add another dataset with **+** first. | One trained detector and one dataset are selected, so Test is enabled. |
| `find` | happy | You selected several datasets or detectors, so Train and Test (one of each) are off. Click **Find** to run every selected detector on every selected dataset. | Several datasets or detectors are selected, each detector with labels or a typed description (#4758), so Find is enabled but Test is not. |

`{media}` is "images, sounds or other media", or the one media type's plural
("images", "audio clips", …) when the server is locked to one
(`solo_media_type`). `{tab}` and `{other}` are the detector tabs' names,
"Drafts" and "AutoFind". `{N}` is the number of selected datasets.
`{detector}` is the name of the first selected detector with nothing to score
with.

## Train view

At most one at a time: the vote hints stand over the **Good** / **Bad**
buttons (`voteHint` in `label-view.component.ts`), and the hand-offs under the
top bar's **Dashboard** button once the run is over. The phase hints and the
hand-offs are for the run that trains the detector: one that began on a
detector with no labels (`AutopilotStateService.startedUntrained`). Coming
back to a trained detector raises none of them.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `start-voting` | happy | Autopilot is showing you its first pick. Click **Good** (→) if this is what you're looking for, or **Bad** (←) if it isn't. Your detector learns from every answer. | Autopilot is running on a detector with no labels, nothing has been voted, and an item is on screen. The first vote ends it. |
| `find-goods` | happy | Autopilot is looking for your first {M Goods} so your detector has something to learn from: {N} found so far. Keep voting. | Autopilot is in **Find Initial Goods** on a run that trains the detector, with an item on screen and a vote cast. |
| `find-bads` | happy | You've found your Goods. Autopilot needs {M Bads} too, so the detector learns what you're not looking for: {N} so far. Keep voting. | As `find-goods`, in **Find Initial Bads**. |
| `find-more` | happy | Your detector has its first Goods and Bads. Autopilot is now walking down the sort for more Goods; keep voting until it moves on. | As `find-goods`, in the **More** walk: a document dataset, or a More target. |
| `keep-voting` | happy | Your detector now retrains after every vote, and Autopilot asks about the items it's least sure of. Keep voting until the lights on the left all turn green. | As `find-goods`, in **Refine Boundary** or **Explore Diversity**, until the run reaches Done. |
| `resort-prompt` | happy | Your {N clicks} {found only K positives}, and Autopilot needs {M} before it can move on. Switch to a different example sort, or keep clicking with this one. | The **Update Sort Example?** prompt is up, below its dialog: Autopilot is still finding its first positives (**Find Initial Goods**) by a text or media example, and that sort has had ten votes (half as many again after each **Continue**) without enough. Answering the prompt ends it, and so does opening its media picker. |
| `autopilot-done` | happy | Your detector is trained! Every quality light is green. Keep voting to sharpen it, or click **Dashboard** to test it or put it to work. | An Autopilot run that started on an untrained detector finishes with every quality light green. The next vote ends it. |
| `autopilot-ran-dry` | happy | Your detector is trained! {N} of its best matches in a row were not good, so it has likely found what it can. Keep voting to sharpen it, or click **Dashboard** to test it or put it to work. | As `autopilot-done`, on a document dataset, where Autopilot finishes when its best matches run dry rather than on the lights. |
| `all-labeled` | happy | You've labeled every item in this dataset! Click **Dashboard** to test or run your detector on another dataset. | An Autopilot run that started on an untrained detector labels every item before it finishes. The next vote ends it. |

`{M Goods}` and `{M Bads}` are Autopilot's Good and Bad targets ("3 Goods",
"4 Bads" by default) and `{N}` the labels of that class so far. In
`resort-prompt`, `{N clicks}` is the votes cast under the current sort,
`{found only K positives}` how many of them were Good ("found no positives"
when none), and `{M}` the Good target. In `autopilot-ran-dry`, `{N}` is the
dry-run length, 16.

## Test view

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `test-vote` | happy | Test is checking your detector's line with picks from both sides of it, in random order. Vote **Good** or **Bad** on each; the result on the right firms up as you go. | The test's first round is on the stage and no pick has been voted yet. The first vote ends it. Stands over **Good** / **Bad**. |
| `test-done` | happy | The picks have said what they can about the line. Read the verdict on the right and take one of its exits, or open the **Review** tab to walk the full list. | The test reached Done. Stands under the stage's **Test done** note. |
| `review` | happy | This is every item, best match first, with the line where the detector stops calling them matches. Click any item to check it with Good or Bad, or **Export** to send the matches on. | The **Review** tab is open on a scored dataset and no item has been checked yet. The first check ends it. Stands under the tab. |

## Dialogs

One per dialog, saying what the user is doing there; each fires while its
dialog is open and on top, so none needs a "how you got here" (the user just
clicked the button that opened it). Where a dialog has views, the view is
named.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `import-dataset` | happy | A dataset is the {media} you'll train a detector on or search. Pick where they come from, fill in the form, and click **Import**. | **Add Dataset** is open. |
| `new-detector` | happy | A detector finds one kind of thing. Describe what you're looking for in a few words, or give an example, then click **Create**. | **New Detector** is open on its main view (not its media picker). |
| `use-example` | happy | You picked a file as the example. **OK** uses all of it; **OK but Crop** lets you mark just the part you mean. | **Use This Example?** is open; the crop view carries its own instructions. |
| `choose-clipper` | happy | A clipper cuts each item into pieces before it's analyzed, so one long recording becomes many short clips. Pick one, set its options, and click **OK**. | **Choose MediaClipper** is open. |
| `combine-datasets` | happy | Combining merges the selected datasets into one new dataset, keeping one copy of anything they share and leaving the originals as they are. Name it and click **Combine**. | **Combine Datasets** is open. |
| `combine-detectors` | happy | Combining pools the selected detectors' labels into one new detector and leaves the originals as they are. Name it and click **Combine**. | **Combine Trainable Detectors** is open. |
| `export-labels` | happy | Export sends this detector's labels out of VTSearch. Pick which labels and which columns to include, then choose where they go. | **Export Labels** is open (the detector-scoped export). |
| `export-results` | happy | Export sends these results out of VTSearch. Pick which items and which columns to include, then choose where they go. | **Export Results** is open (the Test view's export). |
| `import-labels` | happy | Importing labels gives a detector ready-made Good and Bad examples from a file, a URL or a service. Pick a source and fill in its form. | **Import Labels** is open. |
| `dataset-stats` | happy | These are the facts about this dataset: what it holds, how it was imported, and when. Nothing here changes it. | A dataset's **Stats** is open. |
| `detector-stats` | happy | These are the facts about this detector: its labels, what a test measured it to ship, and how it was made. Nothing here changes it. | A detector's **Stats** is open. |
| `duplicates` | happy | These items were collapsed as duplicates when the dataset was imported: each set counts as one item, with every origin it came from. Tick the columns to show, and copy the table if you need it. | **Duplicates** is open, over the dataset's Stats. |
| `find-results` | happy | Find has run on {dataset}, and these are its matches. **Export** sends them to a file, the clipboard or a service; **Browse** lays them out as a map. | **Find Results** is open. |
| `settings` | happy | Settings are yours alone, and every change here saves as you make it. Each tab on the left is one area; **Appearance** is where Toasty's hints are hidden or brought back. | **Settings** is open. |
| `import-settings` | happy | Importing settings replaces yours with a saved set, from a file or a service. Pick a source and fill in its form. | **Import Settings** is open, over Settings. |
| `export-settings` | happy | Exporting settings saves yours as a set you can bring back later or hand to someone else. Pick where they go and fill in its form. | **Export Settings** is open, over Settings. |
| `achievements` | happy | Achievements are milestones tracked as you use VTSearch, with a tier for each. Your progress is saved with your settings, so it carries over between sessions. | **Achievements** is open. |
| `help` | happy | Keyboard shortcuts are listed by the view they work in, and the **User guide** tab has the full manual. | **Help** is open. |
| `load-sort` | happy | Load Sort ranks the items on the left by something other than your detector: another detector, or an example item. Pick one to sort by. | **Load Sort** is open, from the Train view's Manual tab. |
| `spot-check` | happy | A spot check draws a few picks at random from the list and asks about each, to measure how right the detector's line is. Vote **Good** or **Bad** on every pick; the last one sends the round. | **Spot check** is open. |
| `progress-chart` | happy | This chart replays this indicator over your label history, one point for every detector trained along the way, so you can see when it settled. | A quality indicator's history chart is open, from the Train view's phase lights. |
| `source-photo` | happy | This is the photo the face was cut from, with the face outlined. **Back** returns you to where you were. | **Show in photo** is open and the photo is up. |

`{media}` is as on the Dashboard. `{dataset}` is the dataset Find ran on, by
name.
