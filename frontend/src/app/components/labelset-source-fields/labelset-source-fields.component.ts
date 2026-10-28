import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { ImporterField } from '../../models/api.models';
import { inactiveFieldKeys, visibleFields } from '../../utils/plugin-fields';
import { PluginCheckboxComponent } from '../plugin-checkbox/plugin-checkbox.component';

/** True when *field* must be filled in before the source can be used: it is
 *  required and the plugin declares no default the server would fall back to. */
export function needsValue(field: ImporterField): boolean {
  return !!field.required && !field.default && field.field_type !== 'file';
}

/** Whether every field that {@link needsValue} has a non-blank value. A
 *  field its ``visible_when`` switches off on *values* needs none. */
export function fieldsComplete(fields: readonly ImporterField[], values: Record<string, string>): boolean {
  const inactive = inactiveFieldKeys(fields, values);
  return fields.every((f) => inactive.has(f.key) || !needsValue(f) || !!(values[f.key] ?? '').trim());
}

/**
 * The field values a labelset source needs from the user (#4913): the New
 * Detector form's "Auto-export" section and the Dashboard's Auto-export
 * dialog both render the source's open fields with this.
 *
 * Deliberately narrow: labelset sources declare static fields (no
 * ``dynamic_options``), so this renders text-like inputs, number inputs,
 * static selects and checkboxes, and emits a fresh values object on every
 * edit. ``hidden`` fields are dropped, as in every plugin form, and so is a
 * field whose ``visible_when`` fails on the values. A condition naming a
 * field the server left out of *fields* (admin-fixed, or ``hidden``) is
 * ignored here; the server, which sees that field's value, still applies it.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-labelset-source-fields',
  standalone: true,
  imports: [FormsModule, PluginCheckboxComponent],
  templateUrl: './labelset-source-fields.component.html',
  // Stack the fields with the gap the surrounding plugin forms use.
  styles: [':host { display: flex; flex-direction: column; gap: var(--space-lg); }'],
})
export class LabelsetSourceFieldsComponent {
  /** The fields to render (the server already left out what the admin fixed). */
  readonly fields = input<ImporterField[]>([]);
  readonly values = input<Record<string, string>>({});
  /** Prefix for the inputs' element ids, so two forms on a page don't clash. */
  readonly idPrefix = input('lss');
  readonly valuesChange = output<Record<string, string>>();

  get shownFields(): ImporterField[] {
    return visibleFields(this.fields(), this.values());
  }

  value(key: string): string {
    return this.values()[key] ?? '';
  }

  set(key: string, value: string): void {
    this.valuesChange.emit({ ...this.values(), [key]: value ?? '' });
  }

  inputType(field: ImporterField): string {
    if (field.field_type === 'password') return 'password';
    if (field.field_type === 'email') return 'email';
    if (field.field_type === 'url') return 'url';
    return 'text';
  }

  readonly needsValue = needsValue;
}
