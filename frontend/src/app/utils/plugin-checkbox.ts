/**
 * Reading and writing the value of a ``"checkbox"`` ``PluginField``.
 *
 * A checkbox field's value travels as a *string* everywhere it is persisted or
 * submitted (form payloads, saved settings, CLI arguments), because every other
 * ``PluginField`` type does too and the plugin form values are one flat
 * ``Record<string, string>``. Only the rendered `<input type="checkbox">` deals
 * in booleans, so the conversion happens at that one boundary — here.
 *
 * ``isChecked`` is the TypeScript half of ``vtscore.plugins.parse_checkbox``
 * and must agree with it: the box the user ticks and the value the plugin's
 * ``run()`` receives have to mean the same thing. Both accept a native boolean
 * (already-coerced values from the CLI, or a plugin whose ``default`` was
 * written as a Python ``bool``), treat the strings case-insensitively so a
 * ``default="False"`` or ``default="True"`` reads the way its author meant it,
 * and fall back to unchecked for anything unrecognised — including ``null`` and
 * ``undefined``, which is how an omitted field with no default arrives.
 */
export function isChecked(value: unknown): boolean {
  if (typeof value === 'boolean') return value;
  if (value === null || value === undefined) return false;
  return String(value).trim().toLowerCase() === 'true';
}

/** The canonical string a ticked/unticked box submits. */
export function checkboxValue(checked: boolean): string {
  return checked ? 'true' : 'false';
}
