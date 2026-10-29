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
 * A field carrying `visible_when` is shown only while the fields it names
 * hold one of the listed values (a checkbox that switches an e-mail box on,
 * a select whose choice decides which inputs apply). While its condition
 * fails the field is *inactive*: not rendered and not required. Its value
 * stays in `formValues`, so ticking the box back brings back what the user
 * typed, and the server drops whatever is submitted for it — so the form
 * need not.
 *
 * Shared rather than re-filtered per component because nine different forms
 * render plugin fields (export, settings import/export, label import,
 * AutoFind, autodetect results, the dataset-importer pickers, new-detector).
 * A `hidden` honoured by eight of them is the failure this single-sources
 * away: the author sees a value they fixed still offered for editing on
 * whichever surface was missed.
 */

import { isChecked } from './plugin-checkbox';

/** The parts of a plugin field this module reads. */
interface ConditionalField {
  key: string;
  field_type?: string;
  default?: string;
  hidden?: boolean;
  visible_when?: { [key: string]: Array<string> };
}

/** A form's current values: whatever shape the owning component keeps them
 *  in (strings, mostly; a checkbox may hold a boolean). */
type FieldValues = Readonly<Record<string, unknown>>;

/** The string a `visible_when` comparison sees for *value*. */
function conditionToken(value: unknown): string {
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (value === null || value === undefined) return '';
  return String(value).trim();
}

/** *value* as a comparison token for the controlling field *field*: a
 *  checkbox reads through `isChecked`, so `"True"` and a blank box compare
 *  as `"true"` / `"false"`. */
function fieldToken(field: ConditionalField, value: unknown): string {
  if (field.field_type === 'checkbox') return isChecked(value) ? 'true' : 'false';
  return conditionToken(value);
}

/** The token for the value *field* would reach the plugin with: the form's,
 *  or the declared default when the form holds a blank. */
function submittedToken(field: ConditionalField, values: FieldValues): string {
  let raw = values[field.key];
  if (raw === null || raw === undefined || (typeof raw === 'string' && !raw.trim())) {
    raw = field.default;
  }
  return fieldToken(field, raw);
}

/** Whether every entry of *field*'s `visible_when` matches *values*.
 *
 *  The TypeScript twin of `_condition_holds` in `vtscore/plugins/field.py`
 *  and it must agree with it, rule for rule: a key naming no field is
 *  ignored, a controller that is itself switched off counts as its default,
 *  and a cycle stops at the field it re-enters, which counts as active. */
function conditionHolds(
  field: ConditionalField,
  byKey: ReadonlyMap<string, ConditionalField>,
  values: FieldValues,
  chain: ReadonlySet<string>,
): boolean {
  const condition = field.visible_when ?? {};
  const entries = Object.entries(condition);
  if (entries.length === 0) return true;
  const inner = new Set(chain).add(field.key);
  for (const [ctrlKey, accepted] of entries) {
    const ctrl = byKey.get(ctrlKey);
    if (!ctrl) continue;
    const actual =
      inner.has(ctrl.key) || conditionHolds(ctrl, byKey, values, inner)
        ? submittedToken(ctrl, values)
        : fieldToken(ctrl, ctrl.default);
    if (!(accepted ?? []).some((v) => fieldToken(ctrl, v) === actual)) return false;
  }
  return true;
}

/**
 * Keys of the fields whose `visible_when` condition fails on *values*.
 *
 * Mirrors `vtscore.plugins.inactive_field_keys`, the evaluator the server
 * applies to the same submission, so the form a user sees and the values the
 * server accepts cannot disagree. *fields* must be the plugin's whole field
 * list, `hidden` fields included: a hidden field can still control a
 * visible one.
 */
export function inactiveFieldKeys(
  fields: readonly ConditionalField[] | null | undefined,
  values: FieldValues,
): Set<string> {
  const list = fields ?? [];
  if (!list.some((f) => Object.keys(f.visible_when ?? {}).length > 0)) return new Set();
  const byKey = new Map(list.map((f) => [f.key, f] as const));
  return new Set(list.filter((f) => !conditionHolds(f, byKey, values, new Set())).map((f) => f.key));
}

/**
 * Return the fields a form should render: every field that isn't `hidden`,
 * and — when the form's current *values* are given — whose `visible_when`
 * condition holds on them.
 *
 * Pass *values* wherever the result decides what is rendered or required.
 * Without them only `hidden` is applied, which is what a caller wants when
 * it needs every field the user could reach (refreshing dynamic options, so
 * a field that reappears does not show a stale list).
 *
 * Accepts `null` / `undefined` so callers can hand over an optional `fields`
 * property without a separate `?? []`.
 */
export function visibleFields<T extends ConditionalField>(
  fields: readonly T[] | null | undefined,
  values?: FieldValues,
): T[] {
  const inactive = values ? inactiveFieldKeys(fields, values) : new Set<string>();
  return (fields ?? []).filter((f) => !f.hidden && !inactive.has(f.key));
}
