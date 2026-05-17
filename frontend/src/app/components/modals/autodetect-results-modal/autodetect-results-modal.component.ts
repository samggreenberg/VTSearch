import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { ExportersApiService } from '../../../services/exporters-api.service';
import {
  AutoDetectHit,
  AutoDetectResultsData,
  ImporterField,
} from '../../../models/api.models';
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';

@Component({
  selector: 'vt-autodetect-results-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './autodetect-results-modal.component.html',
  styleUrl: './autodetect-results-modal.component.scss',
})
export class AutoDetectResultsModalComponent implements OnInit, OnDestroy {
  @Input() data: AutoDetectResultsData = { results: {} };
  @Output() closed = new EventEmitter<void>();

  exportSides: 'good' | 'bad' | 'both' = 'good';
  exporters: ExporterEntry[] = [];
  selectedExporter = '';
  exporterFields: ImporterField[] = [];
  exportFieldValues: Record<string, string> = {};
  copyColumn = 'origin+name';
  copySeparator = 'newline';
  copyButtonText = 'Copy To Clipboard';

  private destroy$ = new Subject<void>();

  constructor(private exportersApi: ExportersApiService) {}

  ngOnInit(): void {
    this.exportersApi.getExporters().pipe(takeUntil(this.destroy$)).subscribe({
      next: (list) => {
        this.exporters = list;
        if (list.length > 0) {
          this.selectedExporter = list[0].name;
          this.updateExporterFields();
        }
      },
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
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
    this.exporterFields = (exp?.fields ?? []) as ImporterField[];
    this.exportFieldValues = {};
    for (const field of this.exporterFields) {
      if (field.default) {
        this.exportFieldValues[field.key] = field.default;
      }
    }
  }

  onSidesChange(): void {}

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

  close(): void {
    this.closed.emit();
  }
}
