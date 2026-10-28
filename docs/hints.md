# Toasty's hints

Every hint King Toasty gives (#4680, #4807), by its id: the name `hidden_hints`
stores when a user ticks **Hide this hint**. One id is one bubble with one
text. A hint that would say something else in another state is a second hint,
with a condition disjoint from the first. The ids are `HINT_IDS` in
`frontend/src/app/services/hints.service.ts`, and
`tests_lib/meta/test_hints_doc.py` fails when this chart and the templates
disagree on the ids, a hint's Toasty, or its text.

**Toasty** is the face the hint shows: **happy** for a next step, **surprised**
when something needs fixing. Only `offline` uses the **sad** face: the server
going away is the one thing gone wrong a hint speaks to, and otherwise sad is
for error toasts. **Text** is what the bubble says, with the bold words in bold;
`{…}` is filled in at runtime (see the notes under each table).

**What a hint says.** The state that raised it, then the one step to take, in
a sentence or two: "You have detectors, but none is selected. Select one…",
never "Now click a detector". Nothing about the step after that, and nothing
the step already says: picking where a dataset comes from *is* filling in
**Add Dataset**'s form, so its hint doesn't also say "fill in the form" (#5175).
Leave out a "so …" clause that the next sentence makes plain ("has no
training labels yet, so it can't search"). Only
`add-dataset` opens with "Start here!": its state is the app's opening state.
**Bold is for buttons**, tabs included: every one a hint names is bold, by the
label it carries (**Move to AutoFind**, not "move it to **AutoFind**").
Nothing else is bold, so a detector's or a dataset's name stays plain. **Good**
and **Bad** stay bold when they name a vote ("You already labeled this
**Good**"): they are the buttons' own words.
No semicolons: each instruction is its own sentence. Autopilot's hints are for
someone who just wants to click (#5143): no counts, targets, scores or metric
names, and no keyboard shortcuts. Say what Autopilot is doing and what to do
next, and leave the details to its panel.

**Where hints are.** On the path from an empty Dashboard to a detector's
results, and its detours; not on optional side trips (Settings, Help, a
dataset's Stats), which need no help to progress and can always be cancelled.
The Browse map is on the path however it was opened (#5360): Find Results and
Tune send the user there, and its hints teach the map itself, which works the
same from a Dashboard row's **⋯** menu. The exception is `offline`, which
shows wherever the user is when the server goes away.

**One at a time.** Only one hint is ever on screen. Each view shows at most
one (the Dashboard's ladder below, the Train view's vote hint or hand-off, the
Tune view's stage or Review hint, the Browse map's ladder), and
`ToastyHintComponent` shows a hint
inside a dialog only while that dialog is the topmost one open, and a hint on
a view only while no dialog is open at all. So opening any dialog, a confirm
prompt included, hides the view's hint, and stacking a second dialog hides the
first one's. A dialog's hint stands below its box, reaching right, or beside
the box's top corner when the box fills the window. While the server is
unreachable, `offline` is the only hint (#5149): it shows over any dialog, and
`HintsService` keeps every other one off, so no hint points at a step that
can't work.

## Anywhere

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `offline` | sad | The server isn't answering. Check it's still running, then click **Retry**. | The offline banner is up: a few requests in a row got no answer, or the browser went offline. Stands over the banner's **Retry**, over any dialog. It has no Hide boxes, since nothing can be saved until the server is back. **Retry** ends it once the server answers. |

## Dashboard

At most one at a time: the first row, top to bottom, whose condition holds
(`dashboardHint` in `dashboard.component.ts`). So each row also means "and no
row above it fires". None shows while Add Dataset or New Detector is open, or
while the Dashboard is switching context. While one is on show, the reason
under a disabled Train / Tune / Find button is hidden, and so is an empty
table's "Click **+** to add one": Toasty is saying it.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `add-dataset` | happy | Start here! Click **+** to add a dataset of the {media} you want to search. | There are no datasets, and none is importing. |
| `select-dataset` | happy | You have datasets, but none is selected. Select one, or click **+** to add another. | There are datasets, but none is selected. |
| `mixed-datasets` | surprised | The selected datasets hold different kinds of media. Select datasets of one kind. | The selected datasets hold more than one media type. |
| `add-detector` | happy | You need a detector to do the searching. Click **+** to add one. | There are no detectors. |
| `empty-detector-tab` | happy | The **{tab}** tab is empty. Click **{other}** for your detectors, or **+** to make a new one. | There are detectors, but the open tab (Drafts or AutoFind) lists none of them, so none is selected. Stands under the tab bar. |
| `select-detector` | happy | You have detectors, but none is selected. Select one, or click **+** to make a new one. | The open detector tab lists detectors, but none is selected. |
| `mismatch` | surprised | The selected dataset and detector are different kinds of media. Select ones that match, or click **+** to add a new one. | The selected datasets and detectors are not all one media type. |
| `wait-for-import` | happy | Your dataset is still importing. Select it when it joins the table. | A detector is selected and the only dataset is still importing, so there is no dataset row to select yet. |
| `train` | happy | Your new detector has no training labels yet. Click **Train** to teach it. | One detector with no training labels is selected, beside one dataset, so Train is enabled. |
| `train-one-dataset` | surprised | You selected {N} datasets, but a new detector trains on one at a time. Select just one, then click **Train**. | Several datasets are selected with one detector that has no labels and no typed description, so nothing is enabled. |
| `untrained-detector` | surprised | {detector} has no training labels yet. Select it and one dataset to **Train**. | Several detectors are selected and one of them has no labels and no typed description (#4758), so Tune and Find are off. |
| `trained-here` | happy | You trained this detector on this dataset. Select a different dataset or detector to **Tune** or **Find**. | One detector and one dataset are selected, so Tune is enabled, and the detector holds labels from that dataset: its Train votes or Tune corrections there (#5142). Tune and Find would show it items it has already seen. Stands by the datasets' **+**. |
| `test-or-find` | happy | Your detector is trained. Click **Tune** to check it on this dataset, or **Find** to collect its matches. | One trained detector and one dataset it holds no labels from are selected, so Tune is enabled. |
| `find` | happy | You selected several datasets or detectors, so only **Find** is on. Click it to run each detector on each dataset. | Several datasets or detectors are selected, each detector with labels or a typed description (#4758), so Find is enabled but Tune is not. |

`{media}` is "images or other media", or the one media type's plural
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
detector: one that began on a detector without both a Good and a Bad
(`AutopilotStateService.startedUntrained`), such as one with no labels, or one
holding only the Goods of the examples it was made from, which every Face
detector is (#5229). Their votes are the run's own: an example's Good is not
one. Coming back to a trained detector raises none of them. `revote` answers
an item the user picked, so it shows on any run.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `start-voting` | happy | Autopilot picked this for you. Click **Good** if it's what you're looking for, or **Bad** if not. | Autopilot is running with an item on screen, on a detector with no labels and nothing voted, or on the run that trains the detector before it has cast a vote. The first vote ends it. |
| `fix-votes` | happy | Your vote went to **Labels** on the right. Open here to change it, or keep clicking. | Autopilot is running with an item on screen, and the run that trains the detector has its first vote. Stands beside the **Labels** panel, folded or open. The next vote ends it, and `find-goods` takes over. |
| `revote` | happy | You already labeled this **{label}**. Click **{other}** to change it, or Autopilot's step on the left to carry on. | Autopilot is running and the item on screen is one the user picked (from **Labels**, or with ↓) that was already labeled, and still is. Any run, not just the one that trains the detector; not while Autopilot has nothing left to pick, so its step has nowhere to go. A vote on the item ends it, and so does going back to Autopilot's pick. |
| `find-goods` | happy | Autopilot is looking for Goods now. Keep clicking. | Autopilot is in **Find Initial Goods** on a run that trains the detector, with an item on screen and a vote cast. |
| `find-bads` | happy | Autopilot is looking for Bads now. Keep clicking. | As `find-goods`, in **Find Initial Bads**. |
| `find-more` | happy | Autopilot is looking for more Goods now. Keep clicking. | As `find-goods`, in the **More** walk: a document dataset, or a More target. |
| `keep-voting` | happy | Autopilot is sharpening your detector now. Keep clicking. | As `find-goods`, in **Refine Boundary** or **Explore Diversity**, until the run reaches Done. |
| `resort-prompt` | happy | This example isn't finding many Goods. Try a different one or keep clicking. | The **Update Sort Example?** prompt is up, below its dialog: Autopilot is still finding its first positives (**Find Initial Goods**) by a text or media example, and that sort has had ten votes (half as many again after each **Continue**) without enough, or the run has reached Autopilot's 200-vote cap without them (once, #4903). Answering the prompt ends it, and so does opening its media picker. |
| `autopilot-done` | happy | Your detector is trained! Click **Dashboard** to put it to work, or keep clicking to sharpen it. | An Autopilot run that started on an untrained detector reaches Done with items still unlabeled, whatever said so: every quality light green, the labels line's expected level (#4878), the vote cap (#4903), or a document walk run dry. The next vote ends it, and so does leaving the Train view (#5354). |
| `all-labeled` | happy | You've labeled everything! Click **Dashboard** to put your detector to work. | An Autopilot run that started on an untrained detector labels every item, before it finishes or as it reaches Done (#5151): with nothing left to label, this one takes over from `autopilot-done`. The next vote ends it, and so does leaving the Train view. |

In `revote`, `{label}` is the item's label ("Good" or "Bad") and `{other}`
the other one.

## Tune view

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `test-vote` | happy | **Tune** tests your detector's line with picks from both sides of it. Vote **Good** or **Bad** on each. | The test's first round is on the stage and no pick has been voted yet. The first vote ends it. Stands over **Good** / **Bad**. |
| `test-done` | happy | The test is done. Click **Move to AutoFind** if the verdict is good enough, or open the **Review** tab to check the list yourself. | The test reached Done. Stands by the stage's verdict, which has taken the centre pane (#5100). |
| `review` | happy | Every item is here, best match first. Click one to check it with **Good** or **Bad**, or **Export** to send the matches on. | The **Review** tab is open on a scored dataset and no item has been checked yet. The first check ends it. Stands under the tab. |

## Browse view

Every map, whichever button opened it, shows at most one: the first row, top
to bottom, that applies to the map and whose feature the user has never used
(`browseHint` in `browse-view.component.ts`). A feature's first use, on any
visit, retires its row for good and the next one takes over; a row the user
hid is skipped. The record is the per-user `used_features` setting, keyed by
the hint ids (`FEATURE_IDS` in `hints.service.ts`), and Settings' Show All
clears it with the hidden hints, so the ladder starts over. None shows while
the map loads, builds, or is still under its cover. While `browse-tile-details`
is up, the empty bin-details panel's "Right-click a bin" line is hidden, and
while `browse-select` or `browse-verify` is up, so is the empty Selection
list's "Click bins on the canvas" line: Toasty is saying it.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `browse-pan-zoom` | happy | This map shows your {media}, with similar ones near each other. Drag to pan, or double-click to zoom in. | The user has never moved a map. Stands under the zoom buttons. Any move retires it: a drag, the wheel, a double-click, the zoom buttons, the arrow keys, a minimap jump or zoom to fit. |
| `browse-thumbnail-size` | happy | These buttons change how large the thumbnails appear. Try making them larger or smaller. | The user has never resized a map's thumbnails, on a map whose tiles show them (not text, whose tiles are density cells). Stands under the two size buttons, and either one retires it. |
| `browse-tile-details` | happy | Each tile holds the items that landed there. Right-click one to see them all. | The user has never right-clicked a tile to open its bin details. Stands under the buttons at the map's top left (**Rebuild Map**), reaching over the map. |
| `browse-select` | happy | Nothing is selected yet. Click a tile to add its items here. | The user has never selected anything on a map. Stands under the Selection panel's sort row, in place of the empty list's own line. The first selection retires it: a tile clicked or boxed, select-all, or an item picked in the bin details. |
| `browse-verify` | happy | These are your detector's matches. Select any that are wrong, then click **Verified Bad**. | On a map opened from Tune, the one kind whose Selection panel has **Verified Good** and **Verified Bad**, the user has never verified items on a map. Stands where `browse-select` does, below **Verified Bad**, and either button retires it. |

`{media}` is the plural of the map's media type ("images", "audio clips", …),
as on the Dashboard.

## Dialogs

Only the dialogs on the path have one, saying what to do there; none needs a
"how you got here", since the user just clicked the button that opened it.
Each fires while its dialog is open and on top.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `import-dataset` | happy | Pick where your {media} come from, then click **Import**. | **Add Dataset** is open. Its picker's "Select what type of dataset to add." line hides while this is up. |
| `new-detector` | happy | Describe what you're looking for or give an example. | **New Detector** is open on its main view (not its media picker). |
| `use-example` | happy | **OK** uses the whole {item}. **OK but Crop** lets you mark just the part you mean. | **Use This Example?** is open; the crop view carries its own instructions. |
| `find-results` | happy | These are the detector's matches in {dataset}. **Export** sends them on. **Browse** lays them out as a map. | **Find Results** is open. |
| `export-results` | happy | Pick which items and columns to include, then where they go. | **Export Results** is open (the Tune view's export; a detector's own export has no hint). |

`{media}` is as on the Dashboard. `{item}` is one item of the example's media
type ("image", "audio clip", …), or "file" when it has none. `{dataset}` is
the dataset Find ran on, by
name.
