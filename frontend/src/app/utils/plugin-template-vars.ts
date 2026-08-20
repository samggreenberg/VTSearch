/**
 * Client-side preview of the `{template_vars}` the server substitutes into
 * plugin field values.
 *
 * A plugin field can declare `template_vars=("detector_name", "YYYYMMDD", …)`,
 * and `vtscore.plugins.normalize.normalize_field_values` resolves those
 * placeholders **server-side, at run time**. That is the authoritative
 * substitution and it stays that way — nothing here changes what the plugin
 * finally receives.
 *
 * What it *did* mean, though, is that a field whose default is
 * `"{detector_name}"` rendered in the GUI as the literal string
 * `{detector_name}` (or, when the plugin resolved the value lazily in its own
 * body, as an empty box). The user never saw the value they were about to
 * export with, and had nothing to edit (issue #3199). These helpers close that
 * gap by resolving the *declared* variables for display, so the form opens on
 * the same string the server would have produced.
 *
 * Three rules keep the preview honest:
 *
 * 1. **Only declared variables are touched.** A `{detector_name}` in a field
 *    that doesn't declare it is left alone, because the server would leave it
 *    alone too — the `portable_detector` exporter deliberately withholds that
 *    declaration so it can substitute per-detector itself.
 * 2. **Unresolvable variables are left as placeholders.** When the detector
 *    registry hasn't landed yet, `{detector_name}` stays literal rather than
 *    collapsing to an empty string, so the server still fills it in on submit.
 *    That makes the worst case exactly the old behaviour.
 * 3. **Resolved values are sanitised the same way the server sanitises them**
 *    (`vtscore.security.path_validation.sanitize_template_value`), so the
 *    preview can't promise a path the server would spell differently.
 *
 * Only ever use this for **run-now** forms (the Export modal, the Auto-Detect
 * results export). A *persisted* plugin config — Auto-Find's saved exporter
 * fields, a detector's labelset-sync source — must keep the placeholder
 * verbatim: those templates are re-resolved on every later run, which is the
 * whole point of `results_{YYYY}.{MM}.{DD}.csv` on a daily Auto-Find and of
 * `labels/{detector_name}.json` on a sync source. Freezing them at edit time
 * would quietly pin every future run to the day the box was filled in.
 */

/** Field types whose values the server template-substitutes (`_TEXT_LIKE_TYPES`). */
const TEXT_LIKE_FIELD_TYPES: ReadonlySet<string> = new Set([
  'text',
  'url',
  'email',
  'password',
  'folder',
  'server_path',
  'select',
]);

/** Values the caller can supply for the non-date template variables. */
export interface TemplateVarContext {
  /** Resolves `{detector_name}`; `''` when no detector is known yet. */
  detectorName?: string;
  /** Resolves `{detector_id}`; `''` when no detector is known yet. */
  detectorId?: string;
  /** Resolves `{username}`; `''` when auth status hasn't landed. */
  username?: string;
  /** Clock for the date variables. Defaults to now; injectable for tests. */
  now?: Date;
}

/**
 * Mirror of `sanitize_template_value`: make a resolved value safe to splice
 * into a path template. Path separators and NULs become `_`, and an empty or
 * all-dots value collapses to `_` (`.` and `..` address directories).
 */
export function sanitizeTemplateValue(value: string): string {
  const sanitized = value.replace(/[/\\\0]/g, '_');
  if (!sanitized || /^\.+$/.test(sanitized)) return '_';
  return sanitized;
}

/** Zero-pad to two digits. */
function pad2(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

/**
 * Resolve one date/time variable in UTC — the server formats with
 * `datetime.now(timezone.utc)`, so a local-time preview would be off by the
 * viewer's offset (and name a different day either side of midnight).
 */
function resolveDateVar(name: string, now: Date): string | null {
  const yyyy = String(now.getUTCFullYear()).padStart(4, '0');
  const mm = pad2(now.getUTCMonth() + 1);
  const dd = pad2(now.getUTCDate());
  switch (name) {
    case 'YYYYMMDD-HHMMSS':
      return `${yyyy}${mm}${dd}-${pad2(now.getUTCHours())}${pad2(now.getUTCMinutes())}${pad2(now.getUTCSeconds())}`;
    case 'YYYYMMDD':
      return `${yyyy}${mm}${dd}`;
    case 'YYYY':
      return yyyy;
    case 'MM':
      return mm;
    case 'DD':
      return dd;
    default:
      return null;
  }
}

/** Resolve a single `{var}`, or `null` when this client can't (rule 2 above). */
function resolveVar(name: string, ctx: TemplateVarContext): string | null {
  const dated = resolveDateVar(name, ctx.now ?? new Date());
  if (dated !== null) return dated;
  if (name === 'detector_name') return ctx.detectorName || null;
  if (name === 'detector_id') return ctx.detectorId || null;
  if (name === 'username') return ctx.username || null;
  // Unknown to this client (a newer server variable, or a plugin typo the
  // server will reject on submit): leave it for the server to deal with.
  return null;
}

/**
 * Substitute the *declared* `templateVars` in `value` for display.
 *
 * Returns `value` unchanged when nothing is declared, when the field type
 * isn't one the server substitutes, or when no declared placeholder resolves.
 */
export function resolveTemplateVars(
  value: string,
  templateVars: readonly string[] | null | undefined,
  ctx: TemplateVarContext,
  fieldType?: string,
): string {
  if (!value || !templateVars?.length) return value;
  if (fieldType !== undefined && !TEXT_LIKE_FIELD_TYPES.has(fieldType)) return value;

  let out = value;
  for (const name of templateVars) {
    const placeholder = `{${name}}`;
    if (!out.includes(placeholder)) continue;
    const resolved = resolveVar(name, ctx);
    if (resolved === null) continue;
    out = out.split(placeholder).join(sanitizeTemplateValue(resolved));
  }
  return out;
}
