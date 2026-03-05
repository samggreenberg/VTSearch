import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import { VtDialogService } from '../../../services/dialog.service';
import {
  AutoDetectHit,
  AutoDetectResultsData,
  ExporterInfo,
  ImporterField,
} from '../../../models/api.models';

@Component({
  selector: 'vt-autodetect-results-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './autodetect-results-modal.component.html',
  styleUrl: './autodetect-results-modal.component.scss',
})
export class AutoDetectResultsModalComponent implements OnInit {
  @Input() data: AutoDetectResultsData = { results: {} };
  @Input() sortOrder: { id: number; score: number }[] = [];
  @Input() threshold: number | null = null;
  @Input() goodVoteIds: Set<number> = new Set();
  @Input() badVoteIds: Set<number> = new Set();
  @Output() closed = new EventEmitter<void>();
  @Output() votesRefreshed = new EventEmitter<void>();

  exportSides: 'good' | 'bad' | 'both' = 'good';
  exporters: ExporterInfo[] = [];
  selectedExporter = '';
  exporterFields: ImporterField[] = [];
  exportFieldValues: Record<string, string> = {};
  exportStatus = '';
  exportStatusColor = '';
  fillFromSort = false;
  fillInfo = '';
  copyColumn = 'origin+name';
  copySeparator = 'newline';
  copyButtonText = 'Copy To Clipboard';

  constructor(
    private exportersApi: ExportersApiService,
    private sortingApi: SortingApiService,
    private dialog: VtDialogService,
  ) {}

  ngOnInit(): void {
    this.exportersApi.getExporters().subscribe({
      next: (list) => {
        this.exporters = list;
        if (list.length > 0) {
          this.selectedExporter = list[0].name;
          this.updateExporterFields();
        }
      },
    });
  }

  get allHits(): AutoDetectHit[] {
    const hits: AutoDetectHit[] = [];
    for (const result of Object.values(this.data.results || {})) {
      for (const hit of result.hits || []) {
        hits.push(hit);
      }
    }
    return hits;
  }

  get goodCount(): number {
    let total = 0;
    for (const result of Object.values(this.data.results || {})) {
      total += (result.hits || []).length;
    }
    return total;
  }

  get badCount(): number {
    let total = 0;
    for (const result of Object.values(this.data.results || {})) {
      total += (result.negative_hits || []).length;
    }
    return total;
  }

  get displayHits(): AutoDetectHit[] {
    const hits: AutoDetectHit[] = [];
    for (const result of Object.values(this.data.results || {})) {
      if (this.exportSides === 'good') {
        hits.push(...(result.hits || []));
      } else if (this.exportSides === 'bad') {
        hits.push(...(result.negative_hits || []));
      } else {
        hits.push(
          ...(result.hits || []).map((h) => ({ ...h, label: 'good' })),
          ...(result.negative_hits || []).map((h) => ({ ...h, label: 'bad' })),
        );
      }
    }
    return hits;
  }

  formatOrigin(hit: AutoDetectHit): string {
    const origin = hit.origin;
    if (!origin) return '';
    if (origin.params) {
      const firstVal = Object.values(origin.params)[0];
      if (firstVal) return `${origin.importer}(${firstVal})`;
    }
    return origin.importer || '';
  }

  onExporterChange(): void {
    this.updateExporterFields();
  }

  private updateExporterFields(): void {
    const exp = this.exporters.find((e) => e.name === this.selectedExporter);
    this.exporterFields = exp?.fields || [];
    this.exportFieldValues = {};
    for (const field of this.exporterFields) {
      if (field.default) {
        this.exportFieldValues[field.key] = field.default;
      }
    }
  }

  onSidesChange(): void {
    this.updateFillInfo();
  }

  onFillToggle(): void {
    this.updateFillInfo();
  }

  private updateFillInfo(): void {
    if (!this.fillFromSort) {
      this.fillInfo = '';
      return;
    }
    if (!this.sortOrder.length || this.threshold === null) {
      this.fillInfo = 'No sort results available. Run a sort first.';
      return;
    }
    let goodCount = 0;
    let badCount = 0;
    for (const entry of this.sortOrder) {
      if (this.goodVoteIds.has(entry.id) || this.badVoteIds.has(entry.id)) continue;
      if (entry.score >= this.threshold!) goodCount++;
      else badCount++;
    }
    if (this.exportSides === 'good') {
      this.fillInfo = `${goodCount} unlabeled element${goodCount !== 1 ? 's' : ''} above threshold will be labeled Good.`;
    } else if (this.exportSides === 'bad') {
      this.fillInfo = `${badCount} unlabeled element${badCount !== 1 ? 's' : ''} below threshold will be labeled Bad.`;
    } else {
      const total = goodCount + badCount;
      this.fillInfo = `${goodCount} Good + ${badCount} Bad unlabeled element${total !== 1 ? 's' : ''} will be labeled.`;
    }
  }

  async copyToClipboard(): Promise<void> {
    const hits = this.displayHits;
    if (hits.length === 0) return;

    const separatorMap: Record<string, string> = {
      ',': ',',
      tab: '\t',
      space: ' ',
      newline: '\n',
    };
    const sep = separatorMap[this.copySeparator] || '\n';

    const values = hits.map((hit) => {
      const origin = this.formatOrigin(hit);
      const name = hit.origin_name || hit.filename || '';
      switch (this.copyColumn) {
        case 'origin+name':
          return origin ? `${origin}  ${name}` : name;
        case 'name':
          return name;
        case 'md5':
          return hit.md5 || '';
        case 'filename':
          return hit.filename || '';
        case 'origin':
          return origin;
        default:
          return name;
      }
    });

    try {
      await navigator.clipboard.writeText(values.join(sep));
      this.copyButtonText = 'Copied!';
    } catch {
      this.copyButtonText = 'Copy failed';
    }
    setTimeout(() => (this.copyButtonText = 'Copy To Clipboard'), 2000);
  }

  async runExport(): Promise<void> {
    if (!this.selectedExporter) {
      this.setStatus('Select an exporter.', 'var(--text-muted)');
      return;
    }

    if (this.fillFromSort) {
      await this.runFillFromSortExport();
    } else {
      await this.runStandardExport();
    }
  }

  private async runFillFromSortExport(): Promise<void> {
    if (!this.sortOrder.length || this.threshold === null) {
      await this.dialog.alert('No sort results available. Run a sort first.', 'warning');
      return;
    }

    // Dry run
    this.sortingApi
      .fillFromSort({
        sort_results: this.sortOrder,
        threshold: this.threshold!,
        sides: this.exportSides,
        confirm: false,
      })
      .subscribe({
        next: async (counts: any) => {
          const total = (counts.good_count || 0) + (counts.bad_count || 0);
          if (total === 0) {
            await this.dialog.alert('No unlabeled elements to fill.', 'info');
            return;
          }

          let desc: string;
          if (this.exportSides === 'good') desc = `${counts.good_count} Good label${counts.good_count !== 1 ? 's' : ''}`;
          else if (this.exportSides === 'bad') desc = `${counts.bad_count} Bad label${counts.bad_count !== 1 ? 's' : ''}`;
          else desc = `${counts.good_count} Good + ${counts.bad_count} Bad labels`;

          const confirmed = await this.dialog.confirm(`This will add ${desc} to the LabelSet and export. Continue?`);
          if (!confirmed) return;

          this.setStatus('Filling labels...', 'var(--text-muted)');
          this.sortingApi
            .fillFromSort({
              sort_results: this.sortOrder,
              threshold: this.threshold!,
              sides: this.exportSides,
              confirm: true,
            })
            .subscribe({
              next: (fillData: any) => {
                this.exportWithResults(fillData.results);
                this.votesRefreshed.emit();
              },
              error: () => this.setStatus('Failed to fill labels.', 'var(--color-bad)'),
            });
        },
        error: () => this.setStatus('Failed to compute fill counts.', 'var(--color-bad)'),
      });
  }

  private async runStandardExport(): Promise<void> {
    const filteredResults = this.buildFilteredResults();
    this.exportWithResults(filteredResults);
  }

  private exportWithResults(results: unknown): void {
    this.setStatus('Exporting...', 'var(--text-muted)');
    this.exportersApi
      .runExport({
        exporter_name: this.selectedExporter,
        field_values: this.exportFieldValues,
        results,
      })
      .subscribe({
        next: (res: any) => {
          if (res.success) {
            this.setStatus(res.message || 'Export complete.', 'var(--color-good)');
          } else {
            this.setStatus(res.error || 'Export failed.', 'var(--color-bad)');
          }
        },
        error: (err) => {
          this.setStatus(err.error?.error || 'Export error.', 'var(--color-bad)');
        },
      });
  }

  private buildFilteredResults(): unknown {
    const filtered: Record<string, unknown> = {};
    for (const [detName, detResult] of Object.entries(this.data.results || {})) {
      const entry: Record<string, unknown> = { ...detResult };
      if (this.exportSides === 'good') {
        entry['hits'] = detResult.hits || [];
        delete entry['negative_hits'];
      } else if (this.exportSides === 'bad') {
        entry['hits'] = detResult.negative_hits || [];
        entry['total_hits'] = (entry['hits'] as unknown[]).length;
        delete entry['negative_hits'];
      } else {
        const good = (detResult.hits || []).map((h) => ({ ...h, label: 'good' }));
        const bad = (detResult.negative_hits || []).map((h) => ({ ...h, label: 'bad' }));
        entry['hits'] = [...good, ...bad];
        entry['total_hits'] = (entry['hits'] as unknown[]).length;
        delete entry['negative_hits'];
      }
      filtered[detName] = entry;
    }
    return { ...this.data, results: filtered };
  }

  private setStatus(msg: string, color: string): void {
    this.exportStatus = msg;
    this.exportStatusColor = color;
  }

  close(): void {
    this.closed.emit();
  }
}
