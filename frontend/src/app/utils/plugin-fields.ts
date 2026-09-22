/**
 * Filtering a plugin's declared fields down to the ones a form should render.
 *
 * A `PluginField` carrying `hidden: true` is the plugin author's to fill, not
 * the user's: the value is fixed in the field's `default` and the framework
 * normalises it in server-side, so the form has nothing to ask for. Dropping
 * it from the rendered list is the whole of the frontend's job — the field is
 * still on the wire, still seeded into `formValues`, and still submitted, so
 * a hidden field behaves in every other way like a visible one.
 *
 * Shared rather than re-filtered per component because nine different forms
 * render plugin fields (export, settings import/export, label import,
 * Auto-Find, autodetect results, the dataset-importer pickers, new-detector).
 * A `hidden` honoured by eight of them is the failure this single-sources
 * away: the author sees a value they fixed still offered for editing on
 * whichever surface was missed.
 */

/** The `hidden` half of a plugin field; the only part this module reads. */
interface HideableField {
  hidden?: boolean;
}

/**
 * Return the fields a form should render: every field that isn't `hidden`.
 *
 * Accepts `null` / `undefined` so callers can hand over an optional `fields`
 * property without a separate `?? []`.
 */
export function visibleFields<T extends HideableField>(
  fields: readonly T[] | null | undefined,
): T[] {
  return (fields ?? []).filter((f) => !f.hidden);
}
