import { ChangeDetectionStrategy, Component, effect, inject, input, OnDestroy, OnInit, output, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';

import { IconComponent } from '../../../icon/icon.component';
import { ImporterField } from '../../../../models/api.models';
import { ExportersApiService } from '../../../../services/exporters-api.service';
import type { ExporterEntry } from '../../../../generated/api-client/models/exporter-entry';

/** The selected exporter + its per-exporter field-value map, emitted to the
 *  parent so it can persist both onto the settings object. */
export interface AutoFindExporterChange {
  exporter: string;
  fieldValues: Record<string, Record<string, string>>;
}

/**
 * Settings tab body for "Auto-Find".
 *
 * Configures the **Results Exporter** - a tab strip with one tab per pickable
 * exporter (plus "None"). The active tab is the exporter that runs
 * automatically after an Auto-Find; its tab body renders that exporter's own
 * fields. Field values are kept per-exporter so switching tabs keeps each
 * exporter's config. Edits are emitted to the parent, which saves them as
 * `autofind_exporter` / `autofind_exporter_field_values`.
 *
 * *Which* detectors auto-run is not chosen here: that's the Dashboard's
 * Drafts/AutoRun detector tabs (each detector's ⋯ menu moves it between
 * them, driving `PUT /api/detectors/registry/<id>/autofind` and the caller's
 * per-user `autofind_detectors` list).
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-auto-find-settings',
  standalone: true,
  imports: [FormsModule, IconComponent],
  templateUrl: './auto-find-settings.component.html',
  styleUrl: './auto-find-settings.component.scss',
})
export class AutoFindSettingsComponent implements OnInit, OnDestroy {
  private exportersApi = inject(ExportersApiService);

  constructor() {
    // Seed the active tab + working field-values from the inputs, and re-seed
    // whenever the parent pushes new values. Replaces the old `ngOnInit`
    // seeding + `ngOnChanges` (signal inputs don't fire `ngOnChanges`). As a
    // component effect this runs before the first template check, so the
    // initial tab/values are set with no flash.
    effect(() => {
      this.activeExporter = this.autofindExporter() || '';
      this.fieldValues = this.cloneFieldValues(this.autofindExporterFieldValues());
    });
  }

  /** Currently configured results-exporter name ('' = no auto-export). */
  readonly autofindExporter = input('');
  /** Per-exporter saved field values: `{exporter_name: {field_key: value}}`. */
  readonly autofindExporterFieldValues = input<Record<string, Record<string, string>>>({});

  /** Emits the new exporter selection + field-value map for the parent to
   *  persist whenever the user changes the exporter or any of its fields. */
  readonly exporterChange = output<AutoFindExporterChange>();

  // Written from the async load subscribe below (not a zoneless CD trigger)
  // yet read in the template, so they must be signals to repaint on emit;
  // plain-property writes inside a subscribe callback never schedule change
  // detection under zoneless, which is what left "Loading…" stuck forever.
  readonly exporters = signal<ExporterEntry[]>([]);
  readonly loadingExporters = signal(true);

  /** Active exporter tab ('' = the "None" tab). Mirrors ``autofindExporter``
   *  but is the single source of truth for which tab body is shown. */
  activeExporter = '';
  /** Working copy of the per-exporter field values (cloned from the input so
   *  edits don't mutate the parent's object before it persists them). */
  fieldValues: Record<string, Record<string, string>> = {};

  private destroy$ = new Subject<void>();

  ngOnInit(): void {
    this.exportersApi
      .getExporters()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (list) => {
          // Auto-Find runs its export on a scored run, so only exporters that
        // implement that payload belong in this picker.
        this.exporters.set(
          (list || []).filter(
            (exp) => !exp.hidden_from_picker && (exp.supported_payloads ?? []).includes('find_results'),
          ),
        );
          this.loadingExporters.set(false);
        },
        error: () => {
          this.loadingExporters.set(false);
        },
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
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
    const exp = this.exporters().find((e) => e.name === this.activeExporter);
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
    const exp = this.exporters().find((e) => e.name === name);
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
