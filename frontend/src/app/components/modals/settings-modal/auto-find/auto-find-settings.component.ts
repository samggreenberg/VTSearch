import {
  Component,
  EventEmitter,
  Input,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

import { IconComponent } from '../../../icon/icon.component';
import { ImporterField } from '../../../../models/api.models';
import { DetectorsRegistryApiService } from '../../../../services/detectors-registry-api.service';
import { ExportersApiService } from '../../../../services/exporters-api.service';
import type { ExporterEntry } from '../../../../generated/api-client/models/exporter-entry';

/** One registered detector, narrowed to what the autorun checklist needs. */
interface AutorunDetectorEntry {
  id: string;
  name: string;
  autorun: boolean;
  media_type: string;
}

/** The selected exporter + its per-exporter field-value map, emitted to the
 *  parent so it can persist both onto the settings object. */
export interface AutoFindExporterChange {
  exporter: string;
  fieldValues: Record<string, Record<string, string>>;
}

/**
 * Settings tab body for "Auto-Find".
 *
 * Two sections:
 *  1. **Auto-Find detectors** - an editable checklist of every registered
 *     detector. Each checkbox drives the existing per-detector autorun toggle
 *     (`PUT /api/detectors/registry/<id>/autorun`), which persists into the
 *     shared `autorun_detectors` list. (Moved here from the read-only Server
 *     tab.)
 *  2. **Results Exporter** - a tab strip with one tab per pickable exporter
 *     (plus "None"). The active tab is the exporter that runs automatically
 *     after an Auto-Find; its tab body renders that exporter's own fields.
 *     Field values are kept per-exporter so switching tabs keeps each
 *     exporter's config. Edits are emitted to the parent, which saves them as
 *     `autofind_exporter` / `autofind_exporter_field_values`.
 */
@Component({
  selector: 'vt-auto-find-settings',
  standalone: true,
  imports: [CommonModule, FormsModule, IconComponent],
  templateUrl: './auto-find-settings.component.html',
  styleUrl: './auto-find-settings.component.scss',
})
export class AutoFindSettingsComponent implements OnInit, OnChanges, OnDestroy {
  /** Currently configured results-exporter name ('' = no auto-export). */
  @Input() autofindExporter = '';
  /** Per-exporter saved field values: `{exporter_name: {field_key: value}}`. */
  @Input() autofindExporterFieldValues: Record<string, Record<string, string>> = {};

  /** Emits the new exporter selection + field-value map for the parent to
   *  persist whenever the user changes the exporter or any of its fields. */
  @Output() exporterChange = new EventEmitter<AutoFindExporterChange>();

  detectors: AutorunDetectorEntry[] = [];
  exporters: ExporterEntry[] = [];
  loadingDetectors = true;
  loadingExporters = true;
  detectorError = '';

  /** Active exporter tab ('' = the "None" tab). Mirrors ``autofindExporter``
   *  but is the single source of truth for which tab body is shown. */
  activeExporter = '';
  /** Working copy of the per-exporter field values (cloned from the input so
   *  edits don't mutate the parent's object before it persists them). */
  fieldValues: Record<string, Record<string, string>> = {};

  private destroy$ = new Subject<void>();

  constructor(
    private detectorsRegistryApi: DetectorsRegistryApiService,
    private exportersApi: ExportersApiService,
  ) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['autofindExporter']) {
      this.activeExporter = this.autofindExporter || '';
    }
    if (changes['autofindExporterFieldValues']) {
      this.fieldValues = this.cloneFieldValues(this.autofindExporterFieldValues);
    }
  }

  ngOnInit(): void {
    this.activeExporter = this.autofindExporter || '';
    this.fieldValues = this.cloneFieldValues(this.autofindExporterFieldValues);

    this.detectorsRegistryApi
      .getRegistry()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (res) => {
          this.detectors = (res.detectors || []).map((d) => ({
            id: String((d as Record<string, unknown>)['id'] ?? ''),
            name: String((d as Record<string, unknown>)['name'] ?? ''),
            autorun: Boolean((d as Record<string, unknown>)['autorun']),
            media_type: String((d as Record<string, unknown>)['media_type'] ?? ''),
          }));
          this.loadingDetectors = false;
        },
        error: () => {
          this.detectorError = 'Failed to load detectors';
          this.loadingDetectors = false;
        },
      });

    this.exportersApi
      .getExporters()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (list) => {
          this.exporters = (list || []).filter((exp) => !exp.hidden_from_picker);
          this.loadingExporters = false;
        },
        error: () => {
          this.loadingExporters = false;
        },
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  // --- Auto-Find detectors -------------------------------------------------

  /** Toggle a detector's autorun flag. Persists immediately via the existing
   *  registry endpoint; the shared ``autorun_detectors`` list is the backing
   *  store, so this affects every user (autorun is a deployment-level knob). */
  toggleDetector(detector: AutorunDetectorEntry, checked: boolean): void {
    const prev = detector.autorun;
    detector.autorun = checked;
    this.detectorsRegistryApi
      .setAutorun(detector.id, checked)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        error: () => {
          detector.autorun = prev; // revert on failure
          this.detectorError = `Failed to update autorun for "${detector.name}"`;
        },
      });
  }

  // --- Results Exporter ----------------------------------------------------

  /** Select an exporter tab. ``''`` is the "None" tab (auto-export off). When
   *  an exporter is chosen for the first time its fields are seeded from their
   *  defaults so the form is never blank. Emits the change for persistence. */
  selectExporter(name: string): void {
    this.activeExporter = name;
    if (name && !this.fieldValues[name]) {
      this.fieldValues[name] = this.defaultFieldValues(name);
    }
    this.emitChange();
  }

  /** Fields of the active exporter, or ``[]`` for the "None" tab. */
  get activeFields(): ImporterField[] {
    if (!this.activeExporter) return [];
    const exp = this.exporters.find((e) => e.name === this.activeExporter);
    return ((exp?.fields ?? []) as ImporterField[]) || [];
  }

  /** Current value for a field of the active exporter. */
  fieldValue(key: string): string {
    return this.fieldValues[this.activeExporter]?.[key] ?? '';
  }

  /** Write a field value for the active exporter and emit the change. */
  setFieldValue(key: string, value: string): void {
    if (!this.activeExporter) return;
    const current = { ...(this.fieldValues[this.activeExporter] || {}) };
    current[key] = value;
    this.fieldValues = { ...this.fieldValues, [this.activeExporter]: current };
    this.emitChange();
  }

  /** Concrete `<input type>` for a plugin field, mirroring the auto-detect
   *  results modal's field rendering. */
  inputType(field: ImporterField): string {
    if (field.field_type === 'password') return 'password';
    if (field.field_type === 'email') return 'email';
    if (field.field_type === 'url') return 'url';
    return 'text';
  }

  private defaultFieldValues(name: string): Record<string, string> {
    const exp = this.exporters.find((e) => e.name === name);
    const out: Record<string, string> = {};
    for (const field of (exp?.fields ?? []) as ImporterField[]) {
      if (field.default) out[field.key] = field.default;
    }
    return out;
  }

  private cloneFieldValues(
    src: Record<string, Record<string, string>>,
  ): Record<string, Record<string, string>> {
    const out: Record<string, Record<string, string>> = {};
    for (const [name, vals] of Object.entries(src || {})) {
      out[name] = { ...vals };
    }
    return out;
  }

  private emitChange(): void {
    this.exporterChange.emit({
      exporter: this.activeExporter,
      fieldValues: this.cloneFieldValues(this.fieldValues),
    });
  }
}
