# Toasty's hints

Every hint King Toasty gives (#4680), by its id: the name `hidden_hints`
stores when a user ticks **Hide this hint**. One id is one bubble with one
text. A hint that would say something else in another state is a second hint,
with a condition disjoint from the first. The ids are `HINT_IDS` in
`frontend/src/app/services/hints.service.ts`, and
`tests_lib/meta/test_hints_doc.py` fails when this chart and the templates
disagree on the ids, a hint's Toasty, or its text.

**Toasty** is the face the hint shows: **happy** for a next step,
**surprised** when something needs fixing. No hint uses the **sad** face; that
one is for error toasts. **Text** is what the bubble says, with the bold
words in bold; `{…}` is filled in at runtime (see the notes under each table).

## Dashboard

At most one at a time: the first row, top to bottom, whose condition holds
(`dashboardHint` in `dashboard.component.ts`). So each row also means "and no
row above it fires". None shows while Add Dataset or New Detector is open, or
while the Dashboard is switching context.

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `add-dataset` | happy | Start here! Click **+** to add a dataset: the {media} you want to train or search on. | There are no datasets, and none is importing. |
| `select-dataset` | happy | Select a dataset or click **+** to add a new one. | There are datasets, but none is selected. |
| `mixed-datasets` | surprised | These datasets hold different kinds of media, so they can't be searched together. Select datasets of one kind. | The selected datasets hold more than one media type. |
| `add-detector` | happy | Next, click **+** to make a detector. Tell it what you're looking for, and it learns to find it in your datasets. | There are no detectors. |
| `select-detector` | happy | Now click a detector to select it, or click **+** to make a new one. | The open detector tab lists detectors, but none is selected. |
| `mismatch` | surprised | This detector is for a different kind of media than your dataset, so they can't work together. Select ones that match, or click **+** to make a detector for this dataset. | The selected datasets and detectors are not all one media type. |
| `train` | happy | Click **Train** to teach your new detector. Mark examples Good or Bad for it to learn. | One detector with no training labels is selected, beside one dataset, so Train is enabled. |
| `test-or-find` | happy | Click **Test** to eval or improve this detector. Or **Find** to collect its matches. | One trained detector and one dataset are selected, so Test is enabled. |
| `find` | happy | Click **Find** to run these detectors on this dataset. | Several datasets or detectors are selected, all trained, so Find is enabled but Test is not. |

`{media}` is "images, sounds or other media", or the one media type's plural
("images", "audio clips", …) when the server is locked to one
(`solo_media_type`).

## Train view

| Hint | Toasty | Text | Fires when |
|---|---|---|---|
| `start-voting` | happy | Click **Bad** / **Good** (or ← / →) to vote whether this is what you're looking for. Autopilot picks what to ask next. Your detector learns from every answer. | Autopilot is running on a detector with no labels, nothing has been voted, and an item is on screen. The first vote ends it. |
| `resort-prompt` | happy | We need {N positives} before Autopilot can move on. This is an opportunity to try a different example sort, or just keep clicking with the original sort. | The **Update Sort Example?** prompt is up, below its dialog: Autopilot is still finding its first positives (**Find Initial Goods**) by a text or media example, and that sort has had ten votes (half as many again after each **Continue**) without enough. Answering the prompt ends it, and so does opening its media picker. |
| `autopilot-done` | happy | Your detector is trained! Every quality light is green. Keep voting to sharpen it, or click **Dashboard** to test it or put it to work. | An Autopilot run that started on an untrained detector finishes with every quality light green. The next vote ends it. |
| `autopilot-ran-dry` | happy | Your detector is trained! {N} of its best matches in a row were not good, so it has likely found what it can. Keep voting to sharpen it, or click **Dashboard** to test it or put it to work. | As `autopilot-done`, on a document dataset, where Autopilot finishes when its best matches run dry rather than on the lights. |
| `all-labeled` | happy | You've labeled every item in this dataset! Click **Dashboard** to test or run your detector on another dataset. | An Autopilot run that started on an untrained detector labels every item before it finishes. The next vote ends it. |

`{N}` is the dry-run length, 16. `{N positives}` is Autopilot's Good target, "3 positives" by default.
