# Combine Models — UI Plan

## Status

- **Backend:** ✅ shipped
  - `LabelSet.merge(*others, conflict_policy="drop")` in `vtsearch/datasets/labelset.py`
  - `POST /api/trainable-models/combine` in `vtsearch/routes/trainable_models.py`
  - Tests: `tests/test_combine_models.py`
- **Frontend:** ⏳ this document

## Goal

Let a user pick two or more trainable models of the same media type and produce a new trainable model whose labelset is the merge of the sources. The new model is ordinary in every other way: it appears in the trainable-models list, can be activated against a dataset, trained, and calibrated like any other.

## API contract (already implemented)

`POST /api/trainable-models/combine`

```json
{
  "names": ["Dog Barks", "More Dog Barks"],
  "new_name": "All Dog Barks",
  "conflict_policy": "drop"
}
```

Responses:

| Status | Meaning |
|--------|---------|
| `201`  | Combined model created. Body includes `name`, `media_type`, `num_labels`, `combined_from`, `source_label_counts`, `examples`. |
| `400`  | Bad input: <2 names, missing `new_name`, mixed `media_type`, unsupported `conflict_policy`. |
| `404`  | At least one source name not found. |
| `409`  | A trainable model already exists with `new_name`. |
| `422`  | Merge produced an empty labelset (every key was a conflict). |

## Where it lives in the UI

Mirror **Combine Datasets**:
- Combine Datasets is a dataset *importer* surfaced inside the dataset-add flow.
- Combine Models is **not** a dataset operation. Surface it on the **Trainable Models** management surface.

Concretely: the trainable-models list/management view (the screen that already lists all trainable models with `num_labels`, rename, delete) gets a new top-bar button **"Combine…"**. The button opens a modal.

> If trainable-models management is currently embedded in the Detector / Model Registry area rather than its own page, add the button there next to the existing "New trainable model" CTA. The exact host component is whichever one calls `GET /api/trainable-models` to render the list — see `frontend/src/app/services/` for the corresponding service.

## Modal: "Combine Trainable Models"

### Step 1 — Pick sources

- Multi-select list of all trainable models, grouped by `media_type` (audio / image / text / video / document).
- The first model the user checks **locks the media-type filter**: the rows for other media types either dim out or get hidden, with a small "Showing only audio models" hint and a "Clear" link to reset.
- Each row shows: name, num_labels, last_trained_at, current text_query.
- Below the list: a running tally — *"3 selected · 247 total labels (will dedupe + drop conflicts)"*.
- The button advances to step 2 once ≥2 are checked.

### Step 2 — Name and review

- **New name** text input.
  - Inline validation: required, no name collision (call `GET /api/trainable-models` to check; show "A model named 'X' already exists" inline).
- **Sources summary** — read-only list of selected models with their label counts.
- **Conflict policy** — dropdown with a single option for now: *"Drop conflicting labels"*. Render the dropdown disabled with a tooltip ("Only one policy supported today") so the field is discoverable when we add more.
- **Examples preview** — show the deduped examples list that will be carried forward (the API will recompute the same way; this is informational).
- Primary button: **Combine**. Secondary: Back / Cancel.

### Step 3 — Result

After the `POST /api/trainable-models/combine` call:

- **201**: success toast — *"Combined model 'All Dog Barks' created with 247 labels (dropped 12 conflicts)."* Auto-select the new model in the list. Compute `dropped = sum(source_label_counts) - num_labels`; show "+ N labels deduped" if `dropped > 0`.
- **422 empty**: error banner inside the modal — *"Every label was a conflict — no model was created. Try fewer or more aligned sources."* Keep the modal open.
- **409 collision**: error banner asking for a different name. Keep the modal open.
- **400 / 404 / 500**: generic error banner with the server's `error` message.

## Data flow

1. On modal open, fetch `/api/trainable-models` to populate the picker. Already cached by whatever service backs the trainable-models page — reuse it.
2. On submit, call the endpoint. On success, refresh the list (the existing service should expose a `refresh()` or the equivalent observable trigger).
3. No new state in `ActiveContextService`; the user can choose to activate the new combined model afterwards through the normal flow.

## Components to add / touch

- New modal component: `frontend/src/app/components/combine-trainable-models-dialog/` (Angular standalone component or whichever pattern the app already uses — match siblings).
- New method on the trainable-models service (likely `frontend/src/app/services/trainable-models.service.ts` — confirm path): `combine(names: string[], newName: string, conflictPolicy: 'drop'): Observable<CombineResult>`.
- Trigger button on the trainable-models list view.
- Type definition for `CombineResult` matching the 201 response shape.

## Out of scope (deliberately)

- **Conflict-resolution UI** — no per-row conflict review. Drop-on-conflict is the only policy; if users want richer resolution we add it later as additional options on the dropdown.
- **Pre-merge preview** — we don't pre-call a "dry-run" endpoint to show the exact merged count before submit. The result toast tells the user. (If we want this later, add a `dry_run: true` flag to the combine endpoint that returns the would-be counts without writing.)
- **Re-train on combine** — combining is a labelset op only; the threshold + MLP are computed at activation time, same as any other trainable model. No "train now" checkbox.
- **Inheriting labelset_source** — never carry a source from the inputs. (Already enforced server-side.)

## Open questions for follow-up

1. Should the modal offer to **delete the source models** after a successful combine? Probably no by default — destructive, irreversible — but a checkbox might be nice. Default unchecked.
2. Should the combined model's `combined_from` field be **surfaced in the list view** as a badge ("Combined from A + B")? Low-priority but useful for provenance.
3. When more `conflict_policy` values land (e.g. `majority`, `last_wins`), does the dropdown grow, or do we move to a richer "Conflicts" sub-step? Defer until we actually add a second policy.
