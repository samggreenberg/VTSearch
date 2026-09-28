import { ChangeDetectionStrategy, Component, DestroyRef, inject, input, OnInit, output, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import {
  ClipboardColumn,
  ClipboardCopyComponent,
} from '../../clipboard-copy/clipboard-copy.component';
import { ExportersApiService } from '../../../services/exporters-api.service';
import { ToastService } from '../../../services/toast.service';
import { PluginTemplateVarsService } from '../../../services/plugin-template-vars.service';
import {
  AutoDetectDetectorResult,
  AutoDetectHit,
  AutoDetectResultsData,
  ImporterField,
} from '../../../models/api.models';
import type { ExporterEntry } from '../../../generated/api-client/models/exporter-entry';
import { IconComponent } from '../../icon/icon.component';
import { openBlankTab, openExternalUrl, safeExternalUrl } from '../../../utils/external-url';
import { visibleFields } from '../../../utils/plugin-fields';
import { PluginCheckboxComponent } from '../../plugin-checkbox/plugin-checkbox.component';

/** The AutoRun Results dialog: one AutoRun run's hits on one dataset, with
 *  the good / bad / both filter, copy-to-clipboard, and an Export button that
 *  sends the listed rows to any exporter that reads a scored run. Mounted once
 *  in `AppComponent`, fed by `AutoRunService`. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-autodetect-results-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, ClipboardCopyComponent, IconComponent, PluginCheckboxComponent],
  templateUrl: './autodetect-results-modal.component.html',
  styleUrl: './autodetect-results-modal.component.scss',
})
export class AutoDetectResultsModalComponent implements OnInit {
  private exportersApi = inject(ExportersApiService);
  private templateVars = inject(PluginTemplateVarsService);
  private toast = inject(ToastService);

  readonly data = input<AutoDetectResultsData>({ results: {} });
  readonly closed = output<void>();

  exportSides: 'good' | 'bad' | 'both' = 'good';
  // Signals: these are written from the getExporters() subscribe callback (not a
  // zoneless CD trigger) yet read in the template, so they must repaint on emit.
  readonly exporters = signal<ExporterEntry[]>([]);
  readonly selectedExporter = signal('');
  /** The active exporter's *renderable* fields. ``hidden`` fields are left
   *  out (their values are the plugin author's, fixed via ``default``) but
   *  are still seeded into ``exportFieldValues``. */
  readonly exporterFields = signal<ImporterField[]>([]);
  exportFieldValues: Record<string, string> = {};
  /** An Export click is in flight (disables the button). */
  readonly exporting = signal(false);

  /** Columns offered by the shared clipboard control (single-column list mode). */
  readonly clipboardColumns: ClipboardColumn[] = [
    { key: 'origin+name', label: 'Origin + Name' },
    { key: 'name', label: 'Name' },
    { key: 'md5', label: 'MD5' },
    { key: 'filename', label: 'Filename' },
    { key: 'origin', label: 'Origin' },
  ];

  private readonly destroyRef = inject(DestroyRef);

  ngOnInit(): void {
    this.exportersApi.getExporters().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (list) => {
        // Drop exporters the plugin author flagged hidden_from_picker, and
        // those that can't read a scored run - this picker's destination is
        // always find results (matches the export modal, which filters on
        // `labelset` instead).
        const visible = list.filter(
          (exp) => !exp.hidden_from_picker && (exp.supported_payloads ?? []).includes('find_results'),
        );
        this.exporters.set(visible);
        if (visible.length > 0) {
          this.selectedExporter.set(visible[0].name);
          this.updateExporterFields();
        }
      },
    });
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
    this.exporterFields.set(visibleFields(fields));
    this.exportFieldValues = {};
    for (const field of fields) {
      if (field.default) {
        // Resolve the field's declared `template_vars` so a default like
        // `"{detector_name}"` shows the real name instead of the placeholder
        // (issue #3199); the server substitutes again, idempotently, on run.
        this.exportFieldValues[field.key] = this.templateVars.resolveDefault(field);
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

  /** The exporter the Export button sends to: the picked one, or the only one. */
  get activeExporter(): ExporterEntry | undefined {
    return this.exporters().find((e) => e.name === this.selectedExporter());
  }

  /**
   * The run reshaped to hold exactly the rows the table lists.
   *
   * Exporters write a detector's `hits` and conventionally ignore
   * `negative_hits` (see `ResultsExporter.export_find_results`), so the chosen
   * side is moved into `hits`: Bad exports the below-threshold rows, Both
   * exports every row with its `label` stamped, and `total_hits` counts what
   * is being sent.
   */
  exportPayload(): AutoDetectResultsData {
    const data = this.data();
    const results: Record<string, AutoDetectDetectorResult> = {};
    for (const [name, result] of Object.entries(data.results || {})) {
      const good = result.hits || [];
      const bad = result.negative_hits || [];
      const hits =
        this.exportSides === 'good'
          ? good
          : this.exportSides === 'bad'
            ? bad
            : [...good.map((h) => ({ ...h, label: 'good' })), ...bad.map((h) => ({ ...h, label: 'bad' }))];
      results[name] = { ...result, hits, negative_hits: [], total_hits: hits.length };
    }
    return {
      media_type: data.media_type,
      detectors_run: data.detectors_run,
      results,
      missing_detectors: data.missing_detectors ?? [],
    };
  }

  /** Send the listed rows to the active exporter. */
  exportResults(): void {
    const exporter = this.activeExporter;
    if (!exporter || this.exporting()) return;
    const exporterLabel = exporter.display_name || exporter.name;
    const rowCount = this.displayHits.length;
    const plural = rowCount === 1 ? '' : 's';
    // Claim the tab now, while this click still counts as user activation;
    // see `openBlankTab` (#2898).
    const pendingTab = exporter.opens_url ? openBlankTab() : null;
    this.exporting.set(true);
    this.exportersApi
      .runExport({
        exporter_name: exporter.name,
        field_values: { ...this.exportFieldValues },
        results: this.exportPayload() as unknown as Record<string, unknown>,
        payload_kind: 'find_results',
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (response) => {
          this.exporting.set(false);
          const openUrl = safeExternalUrl(response?.open_url);
          let opened = false;
          if (openUrl) {
            opened = pendingTab ? pendingTab.navigate(openUrl) : openExternalUrl(openUrl);
          } else {
            pendingTab?.close();
          }
          this.toast.success({
            message: opened
              ? `Opened ${rowCount.toLocaleString()} result${plural} in ${exporterLabel}`
              : `Exported ${rowCount.toLocaleString()} result${plural} to ${exporterLabel}`,
            detail: openUrl && !opened ? 'Your browser blocked the new tab.' : response?.message,
            action: openUrl ? { label: 'Open', title: openUrl, onClick: () => openExternalUrl(openUrl) } : undefined,
            autoDismissMs: openUrl && !opened ? 0 : undefined,
            dedupKey: 'autorun-results-export',
          });
        },
        error: () => {
          // The error interceptor toasts the server's reason.
          pendingTab?.close();
          this.exporting.set(false);
        },
      });
  }

  close(): void {
    this.closed.emit();
  }
}
