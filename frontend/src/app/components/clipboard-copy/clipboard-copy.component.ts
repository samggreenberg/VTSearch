import { ChangeDetectionStrategy, Component, input, OnChanges, OnDestroy, signal, SimpleChanges } from '@angular/core';

import { FormsModule } from '@angular/forms';

export interface ClipboardColumn {
  key: string;
  label: string;
}

interface DelimiterOption {
  value: string;
  label: string;
  /** Only valid in list mode; a newline can't separate columns within a row. */
  listOnly?: boolean;
}

/**
 * Shared clipboard-copy control used by every export surface so the column
 * model and delimiter vocabulary can't drift between modals. Two modes:
 *
 *  - `table`: a header row plus one delimited line per row, across all
 *    `columns`. The delimiter separates columns; rows are newline-separated.
 *  - `list`: a single column's values joined by the chosen separator. The
 *    user picks which column from a dropdown.
 *
 * Rows are pre-flattened by the parent to `{ columnKey: value }` so this
 * component stays agnostic about the source model (LabeledElement, hit, …).
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-clipboard-copy',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './clipboard-copy.component.html',
  styleUrl: './clipboard-copy.component.scss',
})
export class ClipboardCopyComponent implements OnChanges, OnDestroy {
  readonly mode = input<'table' | 'list'>('table');
  readonly rows = input<Record<string, unknown>[]>([]);
  readonly columns = input<ClipboardColumn[]>([]);
  /** Extra disable condition layered on top of the empty-data guards. */
  readonly disabled = input(false);
  readonly copyLabel = input('Copy');
  readonly buttonVariant = input<'primary' | 'secondary'>('primary');

  /** Unified delimiter vocabulary, shared across both modes. */
  static readonly DELIMITERS: DelimiterOption[] = [
    { value: '\n', label: 'Newline (↵)', listOnly: true },
    { value: ',', label: 'Comma (,)' },
    { value: '\t', label: 'Tab (⇥)' },
    { value: ' ', label: 'Space (␣)' },
    { value: '|', label: 'Pipe (|)' },
    { value: ';', label: 'Semicolon (;)' },
  ];

  delimiter = ',';
  /** List mode: which single column to copy. */
  selectedColumnKey = '';
  /** Transient copy-feedback label ("Copied!" / "Copy failed"). A signal so the
   *  `setTimeout` reset in {@link flash} repaints back to the default label under
   *  zoneless change detection. */
  readonly buttonText = signal('');

  private feedbackTimer: ReturnType<typeof setTimeout> | null = null;

  get delimiterOptions(): DelimiterOption[] {
    return this.mode() === 'list'
      ? ClipboardCopyComponent.DELIMITERS
      : ClipboardCopyComponent.DELIMITERS.filter((d) => !d.listOnly);
  }

  get isDisabled(): boolean {
    if (this.disabled()) return true;
    if (this.rows().length === 0) return true;
    if (this.mode() === 'table') return this.columns().length === 0;
    return !this.selectedColumnKey;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['mode']) {
      // Default delimiter per mode: list → newline, table → comma.
      this.delimiter = this.mode() === 'list' ? '\n' : ',';
    }
    if (changes['columns'] && this.mode() === 'list') {
      // Keep the single-column selector pointed at a valid column.
      const columns = this.columns();
      if (!columns.some((c) => c.key === this.selectedColumnKey)) {
        this.selectedColumnKey = columns[0]?.key ?? '';
      }
    }
  }

  ngOnDestroy(): void {
    if (this.feedbackTimer) clearTimeout(this.feedbackTimer);
  }

  async copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(this.buildText());
      this.flash('Copied!');
    } catch {
      this.flash('Copy failed');
    }
  }

  private buildText(): string {
    if (this.mode() === 'list') {
      const key = this.selectedColumnKey;
      return this.rows().map((r) => this.cell(r, key)).join(this.delimiter);
    }
    const columns = this.columns();
    if (columns.length === 0) return '';
    const header = columns.map((c) => c.label).join(this.delimiter);
    const body = this.rows().map((r) =>
      this.columns().map((c) => this.cell(r, c.key)).join(this.delimiter),
    );
    return [header, ...body].join('\n');
  }

  private cell(row: Record<string, unknown>, key: string): string {
    return String(row[key] ?? '');
  }

  private flash(text: string): void {
    this.buttonText.set(text);
    if (this.feedbackTimer) clearTimeout(this.feedbackTimer);
    this.feedbackTimer = setTimeout(() => {
      this.buttonText.set('');
      this.feedbackTimer = null;
    }, 2000);
  }
}
