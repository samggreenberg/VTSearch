import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ClipperInfo, ConverterInfo, EmbedderInfo, SourceSpec } from '../../../../models/api.models';
import { SourceSpecsPickerComponent } from '../source-specs-picker/source-specs-picker.component';

/** "Advanced ▾" block of the Add Dataset modal - Include media (source
 *  specs), Embedder, Clipper.  Pulled out of the parent component
 *  because the same block was inlined three times (server-folder,
 *  local-folder/files, generic form), with `sf*` / `lf*` / `form*`
 *  prefixed parallel state and a slew of context-dispatching helpers.
 *
 *  The block lives behind an "Advanced ▾" toggle but the embedder and
 *  clipper pickers can stay visible even when the toggle is collapsed
 *  if the user has chosen a non-default value; otherwise their
 *  override would be hidden until they opened Advanced again.  The
 *  Include media picker is gated strictly by the toggle.
 *
 *  The clipper chooser modal lives at the parent level (one instance
 *  shared across all flows), so clicking the clipper "Details" button
 *  (either the native row's button inside the source-specs column, or
 *  the standalone fallback button rendered below the Advanced block)
 *  emits :prop:`clipperChooserRequested` and the parent opens it.
 */
@Component({
  selector: 'vt-import-advanced',
  standalone: true,
  imports: [CommonModule, FormsModule, SourceSpecsPickerComponent],
  templateUrl: './import-advanced.component.html',
  styleUrl: './import-advanced.component.scss',
})
export class ImportAdvancedComponent {
  /** Converters available for the current native type; feeds the
   *  source-specs picker.  Computed by the parent from the importer's
   *  ``available_converters_by_media_type``. */
  @Input() availableConverters: ConverterInfo[] = [];
  /** Native (output) type id of the dataset (e.g. ``"image"``).  Drives
   *  the "include directly" row inside the source-specs picker. */
  @Input() nativeType = '';
  /** Map of ``type_id`` → human label for the source-specs picker. */
  @Input() typeLabels: Record<string, string> = {};
  /** Embedders available for the current media type.  When fewer than
   *  two are available the picker stays hidden; there is nothing to
   *  choose between. */
  @Input() embedders: EmbedderInfo[] = [];
  /** Clippers available for the current media type.  Same single-option
   *  hide rule as :prop:`embedders`. */
  @Input() clippers: ClipperInfo[] = [];
  /** Whether the Include-media (source-specs) block should be offered
   *  at all.  False for non-``multi_media`` generic-form importers; true
   *  for the server-folder and local-folder/files flows. */
  @Input() showSourceSpecs = false;

  /** Two-way bound source-spec list. */
  @Input() sourceSpecs: SourceSpec[] = [];
  @Output() sourceSpecsChange = new EventEmitter<SourceSpec[]>();

  /** Two-way bound embedder selection. */
  @Input() selectedEmbedder = '';
  @Output() selectedEmbedderChange = new EventEmitter<string>();

  /** When non-empty, the embedder picker is hidden because the user has
   *  set a Solo mediaEmbedder for the current mediaType in settings (or
   *  via ``--solo-embedder``). The parent is responsible for resolving
   *  the lock against the live embedder list and only passing a value
   *  here when the locked embedder actually exists for the type; a
   *  stale or removed embedder falls back to the normal picker. */
  @Input() lockedEmbedder = '';

  /** Current clipper selection (one-way).  Changes flow through the
   *  parent's clipper chooser modal; clicking either the native row's
   *  Details button or the standalone fallback Details button emits
   *  :prop:`clipperChooserRequested` and the parent updates this input
   *  after the chooser settles. */
  @Input() selectedClipper = '';

  /** Current parameter values for the selected clipper, keyed by the
   *  clipper's parameter ``key``.  Forwarded to the source-specs
   *  picker so the native row can render a live preview of the active
   *  settings. */
  @Input() selectedClipperParams: Record<string, string | number> = {};

  /** Fired when the user clicks either the native row's Details
   *  button (inside the source-specs column) or the standalone
   *  Details fallback below the Advanced block; parent opens its
   *  shared clipper chooser modal. */
  @Output() clipperChooserRequested = new EventEmitter<void>();

  /** Whether the Advanced section is currently expanded.  Local state
   *  per instance; opening Advanced in one flow does not carry over
   *  to another (the user only ever sees one flow at a time). */
  advancedOpen = false;

  toggleAdvanced(): void {
    this.advancedOpen = !this.advancedOpen;
  }

  /** True when the user has not overridden the embedder, or when the
   *  current selection is one the registry marks ``is_default``. */
  get isDefaultEmbedderSelected(): boolean {
    if (!this.selectedEmbedder) return true;
    return !!this.embedders.find((e) => e.name === this.selectedEmbedder)?.is_default;
  }

  /** True when the first clipper (the registry's recommended default
   *  for this media type) is currently selected. */
  get isDefaultClipperSelected(): boolean {
    return this.clippers.length > 0 && this.clippers[0].name === this.selectedClipper;
  }

  /** The Advanced toggle is shown either when the Include-media block
   *  is available (it lives strictly behind the toggle, so the toggle
   *  must be reachable) or when neither embedder nor clipper has been
   *  overridden (otherwise their pickers stay visible anyway and the
   *  toggle would be redundant). */
  get showAdvancedToggle(): boolean {
    if (this.showSourceSpecs) return true;
    return this.isDefaultEmbedderSelected && this.isDefaultClipperSelected;
  }

  /** Embedder picker is visible when Advanced is open OR when the user
   *  has picked a non-default embedder, and never when a Solo
   *  mediaEmbedder is locked for the current mediaType (then the user
   *  has explicitly opted out of seeing the picker for this type). */
  get showEmbedderPicker(): boolean {
    if (this.lockedEmbedder) return false;
    return this.advancedOpen || !this.isDefaultEmbedderSelected;
  }

  /** Clipper picker is visible when Advanced is open OR when the user
   *  has picked a non-default clipper. */
  get showClipperPicker(): boolean {
    return this.advancedOpen || !this.isDefaultClipperSelected;
  }

  /** Whether to render the standalone clipper "Details" button below
   *  the Advanced block.  When the source-specs column is visible
   *  (Advanced open AND ``showSourceSpecs`` true), the native row in
   *  the column hosts its own Details button, so the standalone one
   *  would be redundant.  In every other case where the user should be
   *  able to reach the clipper chooser (Advanced collapsed but a
   *  non-default clipper is in effect, or importers with no
   *  source-specs column at all such as demo / non-multi-media form), we
   *  fall back to this standalone button so the override stays visible
   *  and the chooser stays reachable. */
  get showStandaloneClipperButton(): boolean {
    if (this.clippers.length <= 1) return false;
    if (!this.showClipperPicker) return false;
    return !(this.advancedOpen && this.showSourceSpecs);
  }

  /** Embedders flagged ``is_default`` for the active media type, shown
   *  in the Recommended optgroup. */
  get recommendedEmbedders(): EmbedderInfo[] {
    return this.embedders.filter((e) => e.is_default);
  }

  /** Non-default embedders, shown in the Advanced optgroup inside the
   *  embedder select. */
  get advancedEmbedderOptions(): EmbedderInfo[] {
    return this.embedders.filter((e) => !e.is_default);
  }

  /** Human label for an embedder's option element. */
  embedderLabel(embedder: EmbedderInfo): string {
    return embedder.display_name || embedder.name;
  }

  /** License-warning string for the current embedder, or null when the
   *  embedder has no special licensing concerns. */
  get licenseNotice(): string | null {
    if (!this.selectedEmbedder) return null;
    const found = this.embedders.find((e) => e.name === this.selectedEmbedder);
    return found?.license_notice ?? null;
  }

  /** Display name for the currently selected clipper.  Defaults to
   *  ``"None"`` for unset or ``*_default`` clippers (which mean "no
   *  pre-processing"). */
  get clipperDisplayName(): string {
    const clipper = this.clippers.find((c) => c.name === this.selectedClipper);
    if (!clipper) return 'None';
    if (clipper.name.endsWith('_default')) return 'None';
    return clipper.display_name || clipper.name;
  }

  /** Fired by the source-specs picker when its specs list changes. */
  onSourceSpecsChange(specs: SourceSpec[]): void {
    this.sourceSpecsChange.emit(specs);
  }

  /** Fired by the embedder select on user pick. */
  onEmbedderChange(name: string): void {
    this.selectedEmbedderChange.emit(name);
  }

  /** Fired by the clipper button to request the parent's chooser. */
  onClipperClick(): void {
    this.clipperChooserRequested.emit();
  }
}
