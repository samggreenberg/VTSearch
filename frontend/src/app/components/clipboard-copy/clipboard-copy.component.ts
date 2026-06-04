import { Component, Input, OnChanges, OnDestroy, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
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
  selector: 'vt-clipboard-copy',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './clipboard-copy.component.html',
  styleUrl: './clipboard-copy.component.scss',
})
export class ClipboardCopyComponent implements OnChanges, OnDestroy {
  @Input() mode: 'table' | 'list' = 'table';
  @Input() rows: Record<string, unknown>[] = [];
  @Input() columns: ClipboardColumn[] = [];
  /** Extra disable condition layered on top of the empty-data guards. */
  @Input() disabled = false;
  @Input() copyLabel = 'Copy';
  @Input() buttonVariant: 'primary' | 'secondary' = 'primary';

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
  buttonText = '';

  private feedbackTimer: ReturnType<typeof setTimeout> | null = null;

  get delimiterOptions(): DelimiterOption[] {
    return this.mode === 'list'
      ? ClipboardCopyComponent.DELIMITERS
      : ClipboardCopyComponent.DELIMITERS.filter((d) => !d.listOnly);
  }

  get isDisabled(): boolean {
    if (this.disabled) return true;
    if (this.rows.length === 0) return true;
    if (this.mode === 'table') return this.columns.length === 0;
    return !this.selectedColumnKey;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['mode']) {
      // Default delimiter per mode: list → newline, table → comma.
      this.delimiter = this.mode === 'list' ? '\n' : ',';
    }
    if (changes['columns'] && this.mode === 'list') {
      // Keep the single-column selector pointed at a valid column.
      if (!this.columns.some((c) => c.key === this.selectedColumnKey)) {
        this.selectedColumnKey = this.columns[0]?.key ?? '';
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
    if (this.mode === 'list') {
      const key = this.selectedColumnKey;
      return this.rows.map((r) => this.cell(r, key)).join(this.delimiter);
    }
    if (this.columns.length === 0) return '';
    const header = this.columns.map((c) => c.label).join(this.delimiter);
    const body = this.rows.map((r) =>
      this.columns.map((c) => this.cell(r, c.key)).join(this.delimiter),
    );
    return [header, ...body].join('\n');
  }

  private cell(row: Record<string, unknown>, key: string): string {
    return String(row[key] ?? '');
  }

  private flash(text: string): void {
    this.buttonText = text;
    if (this.feedbackTimer) clearTimeout(this.feedbackTimer);
    this.feedbackTimer = setTimeout(() => {
      this.buttonText = '';
      this.feedbackTimer = null;
    }, 2000);
  }
}
