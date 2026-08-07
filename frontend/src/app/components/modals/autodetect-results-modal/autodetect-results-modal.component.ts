import { ChangeDetectionStrategy, Component, inject, input, OnDestroy, OnInit, output, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import {
  ClipboardColumn,
  ClipboardCopyComponent,
} from '../../clipboard-copy/clipboard-copy.component';
import { ExportersApiService } from '../../../services/exporters-api.service';
import {
  AutoDetectHit,
  AutoDetectResultsData,
  ImporterField,
} from '../../../models/api.models';
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';
import { IconComponent } from '../../icon/icon.component';
import { openExternalUrl, safeExternalUrl } from '../../../utils/external-url';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-autodetect-results-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, ClipboardCopyComponent, IconComponent],
  templateUrl: './autodetect-results-modal.component.html',
  styleUrl: './autodetect-results-modal.component.scss',
})
export class AutoDetectResultsModalComponent implements OnInit, OnDestroy {
  private exportersApi = inject(ExportersApiService);

  readonly data = input<AutoDetectResultsData>({ results: {} });
  readonly closed = output<void>();

  exportSides: 'good' | 'bad' | 'both' = 'good';
  // Signals: these are written from the getExporters() subscribe callback (not a
  // zoneless CD trigger) yet read in the template, so they must repaint on emit.
  readonly exporters = signal<ExporterEntry[]>([]);
  readonly selectedExporter = signal('');
  readonly exporterFields = signal<ImporterField[]>([]);
  exportFieldValues: Record<string, string> = {};

  /** Columns offered by the shared clipboard control (single-column list mode). */
  readonly clipboardColumns: ClipboardColumn[] = [
    { key: 'origin+name', label: 'Origin + Name' },
    { key: 'name', label: 'Name' },
    { key: 'md5', label: 'MD5' },
    { key: 'filename', label: 'Filename' },
    { key: 'origin', label: 'Origin' },
  ];

  private destroy$ = new Subject<void>();

  ngOnInit(): void {
    this.exportersApi.getExporters().pipe(takeUntil(this.destroy$)).subscribe({
      next: (list) => {
        // Drop exporters the plugin author flagged hidden_from_picker so they
        // never surface in this destination picker (matches the export modal).
        const visible = list.filter((exp) => !exp.hidden_from_picker);
        this.exporters.set(visible);
        if (visible.length > 0) {
          this.selectedExporter.set(visible[0].name);
          this.updateExporterFields();
        }
      },
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * The Auto-Find auto-export's `open_url`, if it returned an openable one.
   *
   * An exporter can format the run's results into a third-party site's URL
   * instead of (or as well as) delivering them somewhere; the same key drives
   * the Export modal. Surfaced as an "Open" button rather than opened on
   * arrival, because these results land from an async response and a popup
   * blocker would swallow an unprompted `window.open()`.
   */
  autoExportUrl(): string | null {
    const status = this.data().auto_export;
    return status?.success ? safeExternalUrl(status.open_url) : null;
  }

  /** Open the auto-export's URL in a new tab (the click is the user gesture). */
  openExternal(url: string): void {
    openExternalUrl(url);
  }

  get allHits(): AutoDetectHit[] {
    const hits: AutoDetectHit[] = [];
    for (const result of Object.values(this.data().results || {})) {
      for (const hit of result.hits || []) {
        hits.push(hit);
      }
    }
    return hits;
  }

  get goodCount(): number {
    let total = 0;
    for (const result of Object.values(this.data().results || {})) {
      total += (result.hits || []).length;
    }
    return total;
  }

  get badCount(): number {
    let total = 0;
    for (const result of Object.values(this.data().results || {})) {
      total += (result.negative_hits || []).length;
    }
    return total;
  }

  get displayHits(): AutoDetectHit[] {
    const hits: AutoDetectHit[] = [];
    for (const result of Object.values(this.data().results || {})) {
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
    const exp = this.exporters().find((e) => e.name === this.selectedExporter());
    const fields = (exp?.fields ?? []) as ImporterField[];
    this.exporterFields.set(fields);
    this.exportFieldValues = {};
    for (const field of fields) {
      if (field.default) {
        this.exportFieldValues[field.key] = field.default;
      }
    }
  }

  onSidesChange(): void {}

  /** Displayed hits flattened to `{ columnKey: value }` rows for the
   *  shared clipboard control. */
  get clipboardRows(): Record<string, string>[] {
    return this.displayHits.map((hit) => {
      const origin = this.formatOrigin(hit);
      const name = hit.origin_name || hit.filename || '';
      return {
        'origin+name': origin ? `${origin}  ${name}` : name,
        name,
        md5: hit.md5 || '',
        filename: hit.filename || '',
        origin,
      };
    });
  }

  close(): void {
    this.closed.emit();
  }
}
