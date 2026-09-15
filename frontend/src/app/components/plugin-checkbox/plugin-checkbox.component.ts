import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

import { checkboxValue, isChecked } from '../../utils/plugin-checkbox';

/**
 * The `<input type="checkbox">` for a ``"checkbox"`` ``PluginField``.
 *
 * Every surface that renders plugin fields (the exporter and importer modals,
 * the label importer, the New Model form, Auto-Find, the dataset importer
 * pickers, …) is a near-duplicate of the others, and each one open-codes a
 * chain of ``@if (field.field_type === …)`` branches. When ``checkbox`` was
 * added, only two of the ten chains grew a branch for it; on the other eight a
 * checkbox field fell through to the generic text input, so the user got a box
 * to *type* ``true`` into (issue #3851).
 *
 * Pulling the widget into one component is what makes that non-repeatable: the
 * branch is now a single element to copy rather than an eight-line block, the
 * string/boolean conversion lives in exactly one place (see
 * ``utils/plugin-checkbox``), and ``tests_lib/meta/test_plugin_field_checkbox.py``
 * can gate on this tag so a new field-rendering surface cannot ship without it.
 *
 * Designed to sit *inside* the field's `<label>`, before the label text, so the
 * box reads as belonging to the label beside it rather than as a nameless
 * control stranded underneath it.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-plugin-checkbox',
  standalone: true,
  imports: [],
  template: `
    <input
      class="plugin-checkbox"
      type="checkbox"
      [id]="inputId()"
      [checked]="checked()"
      [title]="title()"
      (change)="onToggle($event)"
    />
  `,
  styles: [
    `
      .plugin-checkbox {
        // Nudged up a hair: the browser's default baseline alignment leaves
        // the box sitting low against the label text it precedes.
        margin: 0 var(--space-xs) 0 0;
        vertical-align: -1px;
        cursor: pointer;
      }
    `,
  ],
})
export class PluginCheckboxComponent {
  /** The field's current form value; a ``'true'``/``'false'`` string, or a bool. */
  readonly value = input<unknown>('');
  /** DOM id, so the surrounding `<label for="…">` points at the box. */
  readonly inputId = input('');
  readonly title = input('');

  /** Emits the canonical ``'true'``/``'false'`` string the form should store. */
  readonly valueChange = output<string>();

  readonly checked = computed(() => isChecked(this.value()));

  onToggle(event: Event): void {
    this.valueChange.emit(checkboxValue((event.target as HTMLInputElement).checked));
  }
}
