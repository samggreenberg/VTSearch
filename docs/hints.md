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

**What a hint says.** The state that raised it, then the one step to take, in
a sentence or two: "You have detectors, but none is selected. Click one…",
never "Now click a detector". Nothing about the step after that. Only
`add-dataset` opens with "Start here!": its state is the app's opening state.
No semicolons: each instruction is its own sentence.

**Where hints are.** On the path from an empty Dashboard to a detector's
results, and its detours; not on optional side trips (Settings, Help, a
dataset's Stats), which need no help to progress and can always be cancelled.

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
under a disabled Train / Test / Find button is hidden, and so is an empty
table's "Click **+** to add one": Toasty is saying it.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `add-dataset` | happy | Start here! Click **+** to add a dataset: the {media} you want to train or search on. | There are no datasets, and none is importing. |
| `select-dataset` | happy | You have data, but none is selected. Select the dataset you want to search over, or click **+** to add a new one. | There are datasets, but none is selected. |
| `mixed-datasets` | surprised | The datasets you selected hold different kinds of media, so they can't be searched together. Select datasets of one kind. | The selected datasets hold more than one media type. |
| `add-detector` | happy | You need a detector to do the searching. Click **+** to add one and say what you're looking for. | There are no detectors. |
| `empty-detector-tab` | happy | The {tab} tab is empty. Click **{other}** to select one of your detectors, or click **+** to make a new one. | There are detectors, but the open tab (Drafts or AutoFind) lists none of them, so none is selected. Stands under the tab bar. |
| `select-detector` | happy | You have detectors, but none is selected. Click one to select it, or click **+** to make a new one. | The open detector tab lists detectors, but none is selected. |
| `mismatch` | surprised | The detector you selected is for a different kind of media than your dataset, so they can't work together. Select ones that match, or click **+** to make one that does. | The selected datasets and detectors are not all one media type. |
| `wait-for-import` | happy | Your dataset is still importing. Select it when it joins the table. | A detector is selected and the only dataset is still importing, so there is no dataset row to select yet. |
| `train` | happy | Your new detector has no training labels yet. Click **Train** to teach it with Good and Bad examples. | One detector with no training labels is selected, beside one dataset, so Train is enabled. |
| `train-one-dataset` | surprised | You selected {N} datasets, but a new detector trains on one at a time. Select just one, then click **Train**. | Several datasets are selected with one detector that has no labels and no typed description, so nothing is enabled. |
| `untrained-detector` | surprised | **{detector}** has no training labels yet, so it can't search. Deselect it, or select it alone with one dataset and click **Train**. | Several detectors are selected and one of them has no labels and no typed description (#4758), so Test and Find are off. |
| `test-or-find` | happy | Your detector is trained. Click **Test** to measure it on this dataset, or **Find** to collect its matches. | One trained detector and one dataset are selected, so Test is enabled. |
| `find` | happy | You selected several datasets or detectors, so only **Find** is on. Click it to run each detector on each dataset. | Several datasets or detectors are selected, each detector with labels or a typed description (#4758), so Find is enabled but Test is not. |

`{media}` is "images, sounds or other media", or the one media type's plural
("images", "audio clips", …) when the server is locked to one
(`solo_media_type`). `{tab}` and `{other}` are the detector tabs' names,
"Drafts" and "AutoFind". `{N}` is the number of selected datasets.
`{detector}` is the name of the first selected detector with nothing to score
with.

## Train view

At most one at a time (`trainHint` in `label-view.component.ts`): the vote
hints stand over the **Good** / **Bad** buttons, `fix-votes` beside the
**Labels** panel, and the hand-offs under the top bar's **Dashboard** button
once the run is over; none of the others shows while a hand-off is up. The
phase hints, `fix-votes` and the hand-offs are for the run that trains the
detector: one that began on a detector with no labels
(`AutopilotStateService.startedUntrained`). Coming back to a trained detector
raises none of them. `revote` answers an item the user picked, so it shows on
any run.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `start-voting` | happy | Autopilot picked this item for you. Click **Good**, or press the right arrow key, if it's what you're looking for. Click **Bad**, or press the left arrow key, if not. | Autopilot is running on a detector with no labels, nothing has been voted, and an item is on screen. The first vote ends it. |
| `fix-votes` | happy | Your vote went to **Labels** on the right. Misclicked? Open Labels, click the item, and vote again. | Autopilot is running with an item on screen, and the run that trains the detector has its first vote. Stands beside the **Labels** panel, folded or open. The next vote ends it, and `find-goods` takes over. |
| `revote` | happy | You already labeled this item **{label}**. Click **{other}**, or press the {key} arrow key, to change it. Click Autopilot's current step on the left, or press the up arrow key, to go back to its pick. | Autopilot is running and the item on screen is one the user picked (from **Labels**, or with ↓) that was already labeled, and still is. Any run, not just the one that trains the detector; not while Autopilot has nothing left to pick, so its step has nowhere to go. A vote on the item ends it, and so does going back to Autopilot's pick. |
| `find-goods` | happy | Autopilot needs {M Goods} to start training. You've found {N}. Keep voting. | Autopilot is in **Find Initial Goods** on a run that trains the detector, with an item on screen and a vote cast. |
| `find-bads` | happy | Your Goods are in. Autopilot needs {M Bads} too, and has {N}. Keep voting. | As `find-goods`, in **Find Initial Bads**. |
| `find-more` | happy | Autopilot has its first Goods and Bads, and is walking the sort for more Goods. Keep voting. | As `find-goods`, in the **More** walk: a document dataset, or a More target. |
| `keep-voting` | happy | Your detector now retrains on every vote. Keep voting until the lights on the left are all green. | As `find-goods`, in **Refine Boundary** or **Explore Diversity**, until the run reaches Done. |
| `spot-check` | happy | Autopilot is checking the line: these picks are drawn at random from bands of the list. Vote **Good** or **Bad** on each. | Autopilot's **Spot Check** step is on a run that trains the detector, its first round is on the stage, and no pick has been voted yet: the check before Done, or one Autopilot runs while the labels still overlap. The first vote ends it. |
| `resort-prompt` | happy | Your {N clicks} {found only K positives}, and Autopilot needs {M} before it can move on. Switch to a different example sort, or keep clicking with this one. | The **Update Sort Example?** prompt is up, below its dialog: Autopilot is still finding its first positives (**Find Initial Goods**) by a text or media example, and that sort has had ten votes (half as many again after each **Continue**) without enough, or the run has reached Autopilot's 200-vote cap without them (once, #4903). Answering the prompt ends it, and so does opening its media picker. |
| `autopilot-done` | happy | Your detector is trained! Every quality light is green. Keep voting to sharpen it, or click **Dashboard** to test it or put it to work. | An Autopilot run that started on an untrained detector finishes with every quality light green. The next vote ends it. |
| `autopilot-done-level` | happy | Your detector is trained! It expects an F-beta of {F} at your balance. Keep voting to sharpen it, or click **Dashboard** to test it or put it to work. | As `autopilot-done`, when the detector's labels line already expects an F-beta of 0.5 at the balance with a quality light still short of green (#4878). The next vote ends it. |
| `autopilot-done-cap` | happy | Your detector is trained! That's {N} votes, as many as Autopilot asks for. Keep voting to sharpen it, or click **Dashboard** to test it or put it to work. | As `autopilot-done`, when the votes reached Autopilot's cap of 200 with a quality light still short of green and the labels line short of its level (#4903). The next vote ends it. |
| `autopilot-ran-dry` | happy | Your detector is trained! {N} of its best matches in a row were not good, so it has likely found what it can. Keep voting to sharpen it, or click **Dashboard** to test it or put it to work. | As `autopilot-done`, on a document dataset, where Autopilot finishes when its best matches run dry rather than on the lights. |
| `all-labeled` | happy | You've labeled every item in this dataset! Click **Dashboard** to test or run your detector on another dataset. | An Autopilot run that started on an untrained detector labels every item before it finishes. The next vote ends it. |

`{M Goods}` and `{M Bads}` are Autopilot's Good and Bad targets ("3 Goods",
"4 Bads" by default) and `{N}` the labels of that class so far. In `revote`,
`{label}` is the item's label ("Good" or "Bad"), `{other}` the other one, and
`{key}` the arrow key that votes it ("left" for Bad, "right" for Good). In
`resort-prompt`, `{N clicks}` is the votes cast under the current sort,
`{found only K positives}` how many of them were Good ("found no positives"
when none), and `{M}` the Good target. In `autopilot-ran-dry`, `{N}` is the
dry-run length, 16. In `autopilot-done-level`, `{F}` is the F-beta the labels line expects, two decimals. In `autopilot-done-cap`, `{N}` is the vote count at Done (the cap, plus the check before Done).

## Test view

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `test-vote` | happy | Test shows you picks from both sides of your detector's line, in random order. Vote **Good** or **Bad** on each. | The test's first round is on the stage and no pick has been voted yet. The first vote ends it. Stands over **Good** / **Bad**. |
| `test-done` | happy | The test is done. Read the verdict on the right, or open the **Review** tab to check the list yourself. | The test reached Done. Stands under the stage's **Test done** heading, which says nothing more. |
| `review` | happy | Every item is here, best match first. Click one to check it with Good or Bad, or **Export** to send the matches on. | The **Review** tab is open on a scored dataset and no item has been checked yet. The first check ends it. Stands under the tab. |

## Dialogs

Only the dialogs on the path have one, saying what to do there; none needs a
"how you got here", since the user just clicked the button that opened it.
Each fires while its dialog is open and on top.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `import-dataset` | happy | Pick where your {media} come from, fill in the form, and click **Import**. | **Add Dataset** is open. Its picker's "Select what type of dataset to add." line hides while this is up. |
| `new-detector` | happy | Describe what you're looking for in a few words, or give an example, then click **Create**. | **New Detector** is open on its main view (not its media picker). |
| `use-example` | happy | **OK** uses the whole file as the example. **OK but Crop** lets you mark just the part you mean. | **Use This Example?** is open; the crop view carries its own instructions. |
| `find-results` | happy | These are the detector's matches in {dataset}. **Export** sends them on. **Browse** lays them out as a map. | **Find Results** is open. |
| `export-results` | happy | Pick which items and columns to include, then where they go. | **Export Results** is open (the Test view's export; a detector's own export has no hint). |

`{media}` is as on the Dashboard. `{dataset}` is the dataset Find ran on, by
name.
