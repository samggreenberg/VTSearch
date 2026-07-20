import { ChangeDetectionStrategy, Component, inject, input, OnInit, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { DatasetsCrudApiService } from '../../../services/datasets-crud-api.service';
import { DatasetsListingsApiService } from '../../../services/datasets-listings-api.service';
import { DatasetRegistryEntry, MediaTypeInfo } from '../../../models/api.models';
import { apiErrorMessage } from '../../../utils/api-error';

interface CombineRow {
  id: string;
  name: string;
  media_type: string;
  num_items: number;
  pkl_path: string;
  /** One concrete embedder per type this dataset binds. */
  embeddersByType: Record<string, string>;
}

/**
 * One embedder-type conflict across the datasets being combined: the datasets
 * don't all bind the same concrete embedder of `type` (either different
 * embedders, or some bind it and some don't). The user settles each by
 * re-embedding every source to one winner, or dropping the type entirely.
 */
interface EmbedderConflict {
  /** Embedder type key: "semantic" | "patch_semantic" | "structural". */
  type: string;
  /** Human label, e.g. "Semantic". */
  label: string;
  /** Distinct concrete embedders bound for this type, in first-seen order. */
  options: string[];
  /** Whether the conflict is (also) partial coverage — some datasets lack it. */
  partial: boolean;
}

/** Wire shape of one resolution sent to the combine endpoint. */
interface CombineResolution {
  action: 'reembed' | 'drop';
  embedder?: string;
}

/** Types in classification precedence order (structural ▸ patch ▸ semantic). */
const EMBEDDER_TYPE_ORDER = ['structural', 'patch_semantic', 'semantic'];

const EMBEDDER_TYPE_LABELS: Record<string, string> = {
  semantic: 'Semantic',
  patch_semantic: 'Patch',
  structural: 'Structural',
};

/**
 * Payload emitted when a combine kicks off. Carries the pre-dedup source
 * counts alongside the task id so the dashboard can compute a post-combine
 * summary toast ("N unique kept, M duplicates dropped") once the background
 * task settles — the unique count is only knowable from the resulting
 * dataset, but the duplicate count is `totalItems - uniqueKept`.
 */
export interface CombineStartedInfo {
  taskId: string;
  numSources: number;
  totalItems: number;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-combine-datasets-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent],
  templateUrl: './combine-datasets-modal.component.html',
  styleUrl: './combine-datasets-modal.component.scss',
})
export class CombineDatasetsModalComponent implements OnInit {
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);

  /** Datasets pre-selected on the dashboard when the modal was opened. */
  readonly datasets = input<DatasetRegistryEntry[]>([]);

  readonly closed = output<void>();
  readonly combineStarted = output<CombineStartedInfo>();

  rows: CombineRow[] = [];
  // Signals: written from the media-types / combine subscribes (async, not a
  // zoneless CD trigger) yet read in the template, so they must repaint on emit.
  readonly mediaTypes = signal<MediaTypeInfo[]>([]);
  readonly embedderLabels = signal<Record<string, string>>({});
  readonly submitting = signal(false);
  readonly error = signal('');
  name = '';

  /**
   * Per-type resolution choice, keyed by embedder type. The value is the raw
   * `<select>` value: `''` (unchosen), `drop`, or `reembed:<embedder>`. Signal
   * so a selection repaints the gated Combine button.
   */
  readonly resolutionChoices = signal<Record<string, string>>({});

  ngOnInit(): void {
    this.rows = this.datasets()
      .map((d) => ({
        id: d.id,
        name: d.name,
        media_type: d.media_type,
        num_items: Number(d['num_items'] ?? 0),
        pkl_path: String(d['pkl_path'] ?? ''),
        embeddersByType: (d.embedders_by_type as Record<string, string>) ?? {},
      }))
      .filter((r) => !!r.pkl_path);

    this.name = this.defaultName();

    this.datasetsListingsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypes.set(res.media_types || []);
      },
    });

    // Friendly embedder display names for the resolution dropdowns; falls back
    // to the raw name if the lookup hasn't arrived (or the embedder is unknown).
    this.datasetsListingsApi.getEmbedders().subscribe({
      next: (embs) => {
        const map: Record<string, string> = {};
        for (const e of embs) {
          map[e.name] = e.display_name || e.name;
        }
        this.embedderLabels.set(map);
      },
    });
  }

  /** Default combined-dataset name: source names joined with " + ". */
  private defaultName(): string {
    return this.rows.map((r) => r.name).filter((n) => !!n).join(' + ');
  }

  /** Total media items across all selected datasets, before deduplication. */
  get totalItems(): number {
    return this.rows.reduce((sum, r) => sum + (r.num_items || 0), 0);
  }

  get distinctMediaTypes(): string[] {
    return Array.from(new Set(this.rows.map((r) => r.media_type)));
  }

  get sharedMediaType(): string {
    return this.distinctMediaTypes.length === 1 ? this.distinctMediaTypes[0] : '';
  }

  /**
   * Embedder-type conflicts across the selected datasets. A type is conflicted
   * when the datasets don't all bind the same concrete embedder for it — either
   * two bind different embedders, or some bind it and others don't. Empty when
   * every bound type agrees (the common single-embedder case).
   */
  get conflicts(): EmbedderConflict[] {
    const out: EmbedderConflict[] = [];
    if (this.rows.length < 2) return out;
    for (const type of EMBEDDER_TYPE_ORDER) {
      const reps = this.rows.map((r) => r.embeddersByType[type] || '');
      const present = reps.filter((n) => !!n);
      if (present.length === 0) continue;
      const options: string[] = [];
      for (const n of present) {
        if (!options.includes(n)) options.push(n);
      }
      const partial = present.length < this.rows.length;
      if (options.length > 1 || partial) {
        out.push({ type, label: EMBEDDER_TYPE_LABELS[type] || type, options, partial });
      }
    }
    return out;
  }

  get hasConflicts(): boolean {
    return this.conflicts.length > 0;
  }

  embedderLabel(name: string): string {
    return this.embedderLabels()[name] || name;
  }

  /** Comma-joined friendly names of a conflict's distinct embedders. */
  conflictOptionsText(conflict: EmbedderConflict): string {
    return conflict.options.map((n) => this.embedderLabel(n)).join(', ');
  }

  /** How many embedders the combined dataset would keep given current choices. */
  private keptCount(): number {
    // Non-conflicted present types are auto-kept; conflicted ones are kept only
    // when resolved to "re-embed" (dropped ones contribute nothing).
    const conflictTypes = new Set(this.conflicts.map((c) => c.type));
    let kept = 0;
    for (const type of EMBEDDER_TYPE_ORDER) {
      const present = this.rows.some((r) => !!r.embeddersByType[type]);
      if (!present) continue;
      if (!conflictTypes.has(type)) {
        kept += 1;
      } else if ((this.resolutionChoices()[type] || '').startsWith('reembed:')) {
        kept += 1;
      }
    }
    return kept;
  }

  /** Whether every detected conflict has a chosen resolution. */
  get allConflictsResolved(): boolean {
    return this.conflicts.every((c) => !!this.resolutionChoices()[c.type]);
  }

  setResolution(type: string, value: string): void {
    this.resolutionChoices.set({ ...this.resolutionChoices(), [type]: value });
  }

  get canCombine(): boolean {
    if (this.rows.length < 2 || this.distinctMediaTypes.length !== 1) return false;
    if (!this.allConflictsResolved) return false;
    if (this.hasConflicts && this.keptCount() === 0) return false;
    return true;
  }

  /** Tooltip / inline reason describing why the Combine button is disabled. */
  get disabledReason(): string {
    if (this.rows.length < 2) {
      return 'Need at least two datasets to combine.';
    }
    if (this.distinctMediaTypes.length > 1) {
      return `All datasets must share a media type (got ${this.distinctMediaTypes.join(', ')}).`;
    }
    if (!this.allConflictsResolved) {
      return 'Resolve each embedder conflict below to continue.';
    }
    if (this.hasConflicts && this.keptCount() === 0) {
      return 'At least one embedder must be kept — you cannot drop them all.';
    }
    return '';
  }

  mediaTypeLabel(typeId: string): string {
    const mt = this.mediaTypes().find((m) => m.type_id === typeId);
    return mt?.name || typeId;
  }

  mediaTypeIcon(typeId: string): string {
    const mt = this.mediaTypes().find((m) => m.type_id === typeId);
    return mt?.icon || '';
  }

  removeRow(id: string): void {
    this.rows = this.rows.filter((r) => r.id !== id);
    // Prune choices for conflicts that no longer exist after the removal.
    const live = new Set(this.conflicts.map((c) => c.type));
    const pruned: Record<string, string> = {};
    for (const [type, value] of Object.entries(this.resolutionChoices())) {
      if (live.has(type)) pruned[type] = value;
    }
    this.resolutionChoices.set(pruned);
  }

  /** Translate the raw `<select>` values into the wire `resolutions` map. */
  private buildResolutions(): Record<string, CombineResolution> {
    const out: Record<string, CombineResolution> = {};
    for (const c of this.conflicts) {
      const value = this.resolutionChoices()[c.type] || '';
      if (value === 'drop') {
        out[c.type] = { action: 'drop' };
      } else if (value.startsWith('reembed:')) {
        out[c.type] = { action: 'reembed', embedder: value.slice('reembed:'.length) };
      }
    }
    return out;
  }

  submit(): void {
    if (!this.canCombine) return;
    this.submitting.set(true);
    this.error.set('');
    const paths = this.rows.map((r) => r.pkl_path);
    const name = (this.name || '').trim() || this.defaultName();
    const numSources = this.rows.length;
    const totalItems = this.totalItems;
    const resolutions = this.hasConflicts ? this.buildResolutions() : undefined;
    this.datasetsCrudApi.combineDatasets({ datasets: paths, name, resolutions }).subscribe({
      next: (res) => {
        this.submitting.set(false);
        this.combineStarted.emit({ taskId: res.task_id, numSources, totalItems });
      },
      error: (err) => {
        this.submitting.set(false);
        this.error.set(apiErrorMessage(err, 'Combine failed'));
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
