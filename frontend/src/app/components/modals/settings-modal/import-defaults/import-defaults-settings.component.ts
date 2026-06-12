import {
  Component,
  EventEmitter,
  Input,
  OnChanges,
  OnInit,
  Output,
  SimpleChanges,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import {
  ClipperInfo,
  ConverterInfo,
  EmbedderInfo,
  ImportDefaultsByMediaType,
  ImportDefaultsForMediaType,
  MediaTypeInfo,
  SourceSpec,
} from '../../../../models/api.models';
import { DatasetsListingsApiService } from '../../../../services/datasets-listings-api.service';
import { IconComponent } from '../../../icon/icon.component';
import { SourceSpecsPickerComponent } from '../../../dashboard/dataset-importer-modal/source-specs-picker/source-specs-picker.component';
import {
  ClipperChooserComponent,
  ClipperSelection,
} from '../../../dashboard/clipper-chooser/clipper-chooser.component';

/** Settings tab body for "Data Imports": lets the user pick a
 *  per-mediaType default embedder, clipper, and converter-row set.
 *  Mirrors the Add Dataset modal's "Advanced ▾" block but standalone,
 *  always expanded, and not tied to a specific importer.  Whatever the
 *  user picks here is silently auto-filled the next time they open an
 *  importer whose output is the matching mediaType.
 *
 *  The component owns the per-mediaType caches for embedders, clippers,
 *  and converters so switching tabs is instant after the first fetch.
 *  Edits are emitted as a full :type:`ImportDefaultsByMediaType` map so
 *  the parent can persist them with the rest of the settings dict. */
@Component({
  selector: 'vt-import-defaults-settings',
  standalone: true,
  imports: [CommonModule, FormsModule, IconComponent, SourceSpecsPickerComponent, ClipperChooserComponent],
  templateUrl: './import-defaults-settings.component.html',
  styleUrl: './import-defaults-settings.component.scss',
})
export class ImportDefaultsSettingsComponent implements OnInit, OnChanges {
  /** All registered media types; drives the per-mediaType tab strip. */
  @Input() mediaTypes: MediaTypeInfo[] = [];
  /** Current per-mediaType defaults map (the saved value). */
  @Input() defaults: ImportDefaultsByMediaType = {};
  /** When set (solo-mediaType streamlining), the tab strip collapses to
   *  just that one mediaType so the user only configures what they'll
   *  actually use. */
  @Input() effectiveSoloMediaType: string | null = null;
  /** Emits a full updated defaults map whenever the user changes any
   *  per-mediaType setting.  The parent merges it back into its
   *  ``settings`` object and persists. */
  @Output() defaultsChange = new EventEmitter<ImportDefaultsByMediaType>();

  activeType = '';
  clipperChooserOpen = false;

  // Per-mediaType caches, populated on first tab switch.
  embeddersByType: Record<string, EmbedderInfo[]> = {};
  clippersByType: Record<string, ClipperInfo[]> = {};
  convertersByType: Record<string, ConverterInfo[]> = {};
  loadingByType: Record<string, boolean> = {};

  constructor(private datasetsListingsApi: DatasetsListingsApiService) {}

  ngOnInit(): void {
    this.pickInitialTab();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['mediaTypes'] || changes['effectiveSoloMediaType']) {
      this.pickInitialTab();
    }
  }

  private pickInitialTab(): void {
    const visible = this.visibleTypes;
    if (visible.length === 0) return;
    if (this.activeType && visible.some((mt) => mt.type_id === this.activeType)) {
      this.loadForType(this.activeType);
      return;
    }
    const soloMatch = this.effectiveSoloMediaType
      ? visible.find((mt) => mt.type_id === this.effectiveSoloMediaType)
      : null;
    this.selectTab((soloMatch || visible[0]).type_id);
  }

  get visibleTypes(): MediaTypeInfo[] {
    if (this.effectiveSoloMediaType) {
      return this.mediaTypes.filter((mt) => mt.type_id === this.effectiveSoloMediaType);
    }
    return this.mediaTypes;
  }

  get typeLabels(): Record<string, string> {
    const out: Record<string, string> = {};
    for (const mt of this.mediaTypes) out[mt.type_id] = mt.name;
    return out;
  }

  selectTab(typeId: string): void {
    this.activeType = typeId;
    this.loadForType(typeId);
  }

  private loadForType(typeId: string): void {
    if (!typeId) return;
    const need =
      this.embeddersByType[typeId] === undefined ||
      this.clippersByType[typeId] === undefined ||
      this.convertersByType[typeId] === undefined;
    if (!need) return;
    this.loadingByType[typeId] = true;
    let pending = 3;
    const done = () => {
      if (--pending === 0) this.loadingByType[typeId] = false;
    };
    if (this.embeddersByType[typeId] === undefined) {
      this.datasetsListingsApi.getEmbedders(typeId).subscribe({
        next: (e) => {
          this.embeddersByType[typeId] = e;
          done();
        },
        error: () => {
          this.embeddersByType[typeId] = [];
          done();
        },
      });
    } else {
      done();
    }
    if (this.clippersByType[typeId] === undefined) {
      this.datasetsListingsApi.getClippers(typeId).subscribe({
        next: (c) => {
          this.clippersByType[typeId] = c;
          done();
        },
        error: () => {
          this.clippersByType[typeId] = [];
          done();
        },
      });
    } else {
      done();
    }
    if (this.convertersByType[typeId] === undefined) {
      this.datasetsListingsApi.getConverters(typeId).subscribe({
        next: (c) => {
          this.convertersByType[typeId] = c;
          done();
        },
        error: () => {
          this.convertersByType[typeId] = [];
          done();
        },
      });
    } else {
      done();
    }
  }

  get activeEmbedders(): EmbedderInfo[] {
    return this.embeddersByType[this.activeType] || [];
  }

  get activeClippers(): ClipperInfo[] {
    return this.clippersByType[this.activeType] || [];
  }

  get activeConverters(): ConverterInfo[] {
    return this.convertersByType[this.activeType] || [];
  }

  get isLoading(): boolean {
    return !!this.loadingByType[this.activeType];
  }

  get currentDefaults(): ImportDefaultsForMediaType {
    return this.defaults[this.activeType] || {};
  }

  get currentEmbedder(): string {
    return this.currentDefaults.embedder || '';
  }

  get currentClipperName(): string {
    return this.currentDefaults.clipper || '';
  }

  get currentClipperParams(): Record<string, string | number> {
    return this.currentDefaults.clipper_params || {};
  }

  /** Specs list passed to the source-specs picker.  We synthesise an
   *  implicit native "include directly" row at the top so the picker has
   *  a row to show even when the user has no saved non-native rows. */
  get currentSourceSpecsForPicker(): SourceSpec[] {
    const saved = this.currentDefaults.source_specs || [];
    const hasNative = saved.some(
      (s) => s.source_type === this.activeType && s.converter === null,
    );
    if (hasNative) return saved;
    return [{ source_type: this.activeType, converter: null, params: {} }, ...saved];
  }

  get recommendedEmbedders(): EmbedderInfo[] {
    return this.activeEmbedders.filter((e) => e.is_default);
  }

  get advancedEmbedderOptions(): EmbedderInfo[] {
    return this.activeEmbedders.filter((e) => !e.is_default);
  }

  /** Built-in default embedder for the active mediaType: the first
   *  ``is_default`` option, falling back to the first available embedder.
   *  Used to seed the dropdown when the user has no saved override. */
  get defaultEmbedderName(): string {
    const recommended = this.recommendedEmbedders[0];
    if (recommended) return recommended.name;
    const first = this.activeEmbedders[0];
    return first ? first.name : '';
  }

  /** What the dropdown actually shows as selected: the user's override if
   *  set, otherwise the built-in default.  Picking this value back from
   *  the dropdown clears the override (see ``onEmbedderChange``). */
  get displayedEmbedder(): string {
    return this.currentEmbedder || this.defaultEmbedderName;
  }

  embedderLabel(embedder: EmbedderInfo): string {
    return embedder.display_name || embedder.name;
  }

  get licenseNotice(): string | null {
    const found = this.activeEmbedders.find((e) => e.name === this.currentEmbedder);
    return found?.license_notice ?? null;
  }

  /** True when the user has at least one saved override for the active
   *  mediaType, used to gate the "Reset" button so it doesn't appear
   *  when there's nothing to reset. */
  get hasOverridesForActiveType(): boolean {
    const d = this.defaults[this.activeType];
    if (!d) return false;
    if (d.embedder) return true;
    if (d.clipper) return true;
    if (d.clipper_params && Object.keys(d.clipper_params).length > 0) return true;
    if (d.source_specs && d.source_specs.length > 0) return true;
    return false;
  }

  get activeTypeLabel(): string {
    return this.typeLabels[this.activeType] || this.activeType;
  }

  // -----------------------------------------------------------------
  // Mutators
  // -----------------------------------------------------------------

  private updateActive(patch: Partial<ImportDefaultsForMediaType>): void {
    const next: ImportDefaultsByMediaType = { ...this.defaults };
    const merged: ImportDefaultsForMediaType = { ...this.currentDefaults, ...patch };
    // Strip empty values so the persisted dict stays compact.
    for (const k of Object.keys(merged) as (keyof ImportDefaultsForMediaType)[]) {
      const v = merged[k];
      if (v === '' || v === undefined || v === null) {
        delete merged[k];
      } else if (Array.isArray(v) && v.length === 0) {
        delete merged[k];
      } else if (
        typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0
      ) {
        delete merged[k];
      }
    }
    if (Object.keys(merged).length === 0) {
      delete next[this.activeType];
    } else {
      next[this.activeType] = merged;
    }
    this.defaultsChange.emit(next);
  }

  onEmbedderChange(name: string): void {
    // Re-picking the built-in default clears the override so the saved
    // settings stay compact and the Reset button doesn't activate for a
    // no-op change.
    const next = name === this.defaultEmbedderName ? '' : name;
    this.updateActive({ embedder: next });
  }

  onSourceSpecsChange(specs: SourceSpec[]): void {
    // The picker hands back the full list (native row + non-native rows).
    // Persist it as-is so the importer can apply it without a second
    // round of native-row synthesis.
    this.updateActive({ source_specs: specs });
  }

  openClipperChooser(): void {
    if (this.activeClippers.length <= 1) return;
    this.clipperChooserOpen = true;
  }

  onClipperSelected(sel: ClipperSelection): void {
    this.clipperChooserOpen = false;
    this.updateActive({ clipper: sel.name, clipper_params: sel.params });
  }

  onClipperChooserCancelled(): void {
    this.clipperChooserOpen = false;
  }

  resetActiveType(): void {
    const next: ImportDefaultsByMediaType = { ...this.defaults };
    delete next[this.activeType];
    this.defaultsChange.emit(next);
  }
}
