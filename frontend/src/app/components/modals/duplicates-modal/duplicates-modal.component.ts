import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  OnInit,
  output,
  signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import {
  ClipboardColumn,
  ClipboardCopyComponent,
} from '../../clipboard-copy/clipboard-copy.component';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { apiErrorMessage } from '../../../utils/api-error';
import type { DuplicateSet } from '../../../generated/api-client/models/duplicate-set';

/** One preview/clipboard row: a duplicate-set member flattened with its
 *  1-based set number, so members of the same set share a "Dupe Set" value. */
export interface DuplicateRow {
  dupe_set: string;
  md5: string;
  filename: string;
  category: string;
  origin_name: string;
  importer: string;
}

export interface DupeColumnDef {
  key: keyof DuplicateRow;
  label: string;
  enabled: boolean;
}

/**
 * Browse the collapsed duplicate sets of a loaded dataset (issue #2697): a
 * lightweight take on the Export window's preview-table + clipboard shape,
 * with a "Dupe Set" column in place of "Label" so the user can see which
 * items were collapsed together and the origins of each member. Opened as a
 * child modal from the Dataset Stats modal's "Duplicate groups" row.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-duplicates-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, ClipboardCopyComponent],
  templateUrl: './duplicates-modal.component.html',
  styleUrl: './duplicates-modal.component.scss',
})
export class DuplicatesModalComponent implements OnInit {
  private readonly datasetsRegistryApi = inject(DatasetsRegistryApiService);

  readonly datasetId = input('');
  readonly datasetName = input('');
  readonly closed = output<void>();

  readonly loading = signal(true);
  readonly error = signal('');
  private readonly sets = signal<DuplicateSet[]>([]);

  /** Preview row cap, mirroring the Export window's preview truncation. */
  private static readonly PREVIEW_LIMIT = 50;

  /** Column toggles; `enabled` is checkbox-bound, so a mutable field. */
  columns: DupeColumnDef[] = [
    { key: 'dupe_set', label: 'Dupe Set', enabled: true },
    { key: 'md5', label: 'MD5', enabled: true },
    { key: 'filename', label: 'Filename', enabled: true },
    { key: 'category', label: 'Category', enabled: true },
    { key: 'origin_name', label: 'Origin', enabled: true },
    { key: 'importer', label: 'Importer', enabled: true },
  ];

  ngOnInit(): void {
    this.datasetsRegistryApi.getDatasetDuplicates(this.datasetId()).subscribe({
      next: (data) => {
        this.sets.set(data.duplicate_sets);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(apiErrorMessage(err, 'Failed to load duplicates'));
        this.loading.set(false);
      },
    });
  }

  /** All members of all sets, flattened to one row per member. */
  readonly rows = computed<DuplicateRow[]>(() =>
    this.sets().flatMap((set, setIndex) =>
      set.members.map((m) => ({
        dupe_set: String(setIndex + 1),
        md5: m.md5,
        filename: m.filename,
        category: m.category,
        origin_name: m.origin_name,
        importer: m.importer,
      })),
    ),
  );

  get enabledColumns(): DupeColumnDef[] {
    return this.columns.filter((c) => c.enabled);
  }

  get numSets(): number {
    return this.sets().length;
  }

  get previewRows(): DuplicateRow[] {
    return this.rows().slice(0, DuplicatesModalComponent.PREVIEW_LIMIT);
  }

  get truncatedCount(): number {
    return Math.max(0, this.rows().length - DuplicatesModalComponent.PREVIEW_LIMIT);
  }

  /** True on the first row of every set after the first, so the preview can
   *  draw a separator line between consecutive sets. */
  isSetStart(index: number): boolean {
    const rows = this.previewRows;
    return index > 0 && rows[index - 1].dupe_set !== rows[index].dupe_set;
  }

  getCellValue(row: DuplicateRow, col: DupeColumnDef): string {
    return row[col.key];
  }

  get clipboardColumns(): ClipboardColumn[] {
    return this.enabledColumns.map((c) => ({ key: c.key, label: c.label }));
  }

  get clipboardRows(): Record<string, string>[] {
    return this.rows() as unknown as Record<string, string>[];
  }

  close(): void {
    this.closed.emit();
  }
}
