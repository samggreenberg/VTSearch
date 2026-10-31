import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

import { IconComponent } from '../icon/icon.component';

/** One column a list can show: its key, the name it goes by, and whether it is on. */
export interface ColumnChoice {
  key: string;
  label: string;
  enabled: boolean;
  /** Tooltip on the column's checkbox, when its name needs a word more. */
  title?: string;
}

/**
 * Which columns a list shows and exports: a checkbox per column, under a
 * heading with a tri-state select-all (#5464). Shared by the Export dialog,
 * where the choice decides the preview and the export, and Find Results, where
 * it decides the list, the clipboard and the export alike.
 *
 * Presentational: the host projects its own heading (each dialog heads its
 * sections its own way), owns the columns, and applies each toggle, so it can
 * keep them however it likes (a plain array, a signal).
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-column-chooser',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './column-chooser.component.html',
  styleUrl: './column-chooser.component.scss',
})
export class ColumnChooserComponent {
  readonly columns = input.required<readonly ColumnChoice[]>();

  /** A column's box was ticked or cleared. */
  readonly toggled = output<{ key: string; enabled: boolean }>();
  /** The select-all box was clicked; the host decides all-on or all-off. */
  readonly allToggled = output<void>();

  /** How many columns are on: none, some, or all. Read at render, since a
   *  host may keep its columns as a mutable array. */
  get state(): 'none' | 'some' | 'all' {
    const columns = this.columns();
    const on = columns.filter((c) => c.enabled).length;
    if (on === 0) return 'none';
    return on >= columns.length ? 'all' : 'some';
  }
}
