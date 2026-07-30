import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

import { FormsModule } from '@angular/forms';

import {
  CleanerInfo,
  CleanerSelection,
  ClipperInfo,
  ClipperParameter,
  ConverterInfo,
  EmbedderInfo,
  SourceSpec,
} from '../../../../models/api.models';
import { SourceSpecsPickerComponent } from '../source-specs-picker/source-specs-picker.component';

/** "Advanced ▾" block of the Add Dataset modal: Include media (source
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
 *  The clipper chooser modal lives one level up (each of the four
 *  Add Dataset picker views - generic form / server-folder /
 *  local-folder / demo - owns its own instance), so clicking the
 *  clipper "Details" button (either the native row's button inside the
 *  source-specs column, or the standalone fallback button rendered
 *  below the Advanced block) emits :prop:`clipperChooserRequested` and
 *  that owner opens it.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-import-advanced',
  standalone: true,
  imports: [FormsModule, SourceSpecsPickerComponent],
  templateUrl: './import-advanced.component.html',
  styleUrl: './import-advanced.component.scss',
})
export class ImportAdvancedComponent {
  /** Converters available for the current native type; feeds the
   *  source-specs picker.  Computed by the parent from the importer's
   *  ``available_converters_by_media_type``. */
  readonly availableConverters = input<ConverterInfo[]>([]);
  /** Native (output) type id of the dataset (e.g. ``"image"``). Drives
   *  the "include directly" row inside the source-specs picker. */
  readonly nativeType = input('');
  /** Map of ``type_id`` → human label for the source-specs picker. */
  readonly typeLabels = input<Record<string, string>>({});
  /** Embedders available for the current media type.  When fewer than
   *  two are available the picker stays hidden; there is nothing to
   *  choose between. */
  readonly embedders = input<EmbedderInfo[]>([]);
  /** Clippers available for the current media type.  Same single-option
   *  hide rule as :prop:`embedders`. */
  readonly clippers = input<ClipperInfo[]>([]);
  /** Cleanup gates registered for the current media type.  Unlike clippers
   *  these are not mutually exclusive: every enabled one runs, in registry
   *  order, on each finished unit before it is embedded.  An empty list
   *  hides the Cleanup block entirely. */
  readonly cleaners = input<CleanerInfo[]>([]);
  /** Whether the Include-media (source-specs) block should be offered
   *  at all.  True for every importer that participates in the dataset
   *  modal's form / server-folder / local-folder/files flows. */
  readonly showSourceSpecs = input(false);

  /** Two-way bound source-spec list. */
  readonly sourceSpecs = input<SourceSpec[]>([]);
  readonly sourceSpecsChange = output<SourceSpec[]>();

  /** Two-way bound embedder selection (the primary embedder; recorded as each
   *  media's primary and the lead of the v3 trio's embed order). */
  readonly selectedEmbedder = input('');
  readonly selectedEmbedderChange = output<string>();

  /** Optional second/third embedders of the v3 trio
   *  (``docs/plans/patch-embedder.md`` → "V3").  The primary picker above can be
   *  any embedder (including a single-vector one); these two *add* a
   *  patch-capable embedder for region search/voting and a structural embedder
   *  for instance matching, so a dataset can bind one of each role.  Both
   *  default to ``''`` (None / not bound) and only appear under Advanced when
   *  the media type actually offers an embedder of that role. */
  readonly selectedPatchEmbedder = input('');
  readonly selectedPatchEmbedderChange = output<string>();
  readonly selectedStructuralEmbedder = input('');
  readonly selectedStructuralEmbedderChange = output<string>();

  /** When non-empty, the embedder picker is hidden because the user has
   *  set a Solo mediaEmbedder for the current mediaType in settings (or
   *  via ``--solo-embedder``). The parent is responsible for resolving
   *  the lock against the live embedder list and only passing a value
   *  here when the locked embedder actually exists for the type; a
   *  stale or removed embedder falls back to the normal picker. */
  readonly lockedEmbedder = input('');

  /** Current clipper selection (one-way).  Changes flow through the
   *  parent's clipper chooser modal; clicking either the native row's
   *  Details button or the standalone fallback Details button emits
   *  :prop:`clipperChooserRequested` and the parent updates this input
   *  after the chooser settles. */
  readonly selectedClipper = input('');

  /** Current parameter values for the selected clipper, keyed by the
   *  clipper's parameter ``key``.  Forwarded to the source-specs
   *  picker so the native row can render a live preview of the active
   *  settings. */
  readonly selectedClipperParams = input<Record<string, string | number>>({});

  /** Fired when the user clicks either the native row's Details
   *  button (inside the source-specs column) or the standalone
   *  Details fallback below the Advanced block and the parent opens its
   *  shared clipper chooser modal. */
  readonly clipperChooserRequested = output<void>();

  /** Two-way bound cleanup selection: one entry per *enabled* cleaner, with
   *  that cleaner's parameter overrides.  Seeded by the parent from each
   *  cleaner's ``default_enabled`` flag, so the checkboxes come up matching
   *  the registry's recommendation. */
  readonly selectedCleaners = input<CleanerSelection[]>([]);
  readonly selectedCleanersChange = output<CleanerSelection[]>();

  /** Two-way bound "compute the 2-D Browse projection at ingest" toggle.
   *  Defaults off: building the UMAP layout + hex-tile pyramid up front
   *  costs compute the user may not want to spend (Browse builds it lazily
   *  on first visit otherwise).  Lives in the Advanced block because it is
   *  a cost/latency tradeoff, not a routine import setting. */
  readonly buildProjection = input(false);
  readonly buildProjectionChange = output<boolean>();

  /** Two-way bound "merge near-duplicates" toggle.  When on, the load
   *  pipeline runs an extra near-duplicate collapse (images + text) after
   *  exact MD5 dedup: visually/textually near-identical items are merged into
   *  one representative (the largest copy), keeping every member's provenance
   *  so exporting one exports the whole set.  Defaults off; lives in Advanced
   *  because it changes which items appear in the dataset. */
  readonly mergeNearDuplicates = input(false);
  readonly mergeNearDuplicatesChange = output<boolean>();

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
    if (!this.selectedEmbedder()) return true;
    return !!this.embedders().find((e) => e.name === this.selectedEmbedder())?.is_default;
  }

  /** True when the first clipper (the registry's recommended default
   *  for this media type) is currently selected. */
  get isDefaultClipperSelected(): boolean {
    return this.clippers().length > 0 && this.clippers()[0].name === this.selectedClipper();
  }

  /** Whether the source-specs "Include media" column has anything worth
   *  showing: the importer participates in the source-specs flow AND at
   *  least one non-native converter feeds the native type.  When a media
   *  type has no converters into it (e.g. video, document), the column
   *  would render only the forced, always-checked native row - a
   *  question with no real answer - so it is suppressed entirely.  This
   *  matches the Import Defaults settings, which gates the same block on
   *  ``activeConverters.length > 0``. */
  get hasSourceSpecsColumn(): boolean {
    return this.showSourceSpecs() && this.availableConverters().length > 0;
  }

  /** True when the cleanup checkboxes still match the registry's own
   *  recommendation (each cleaner's ``default_enabled``) with no parameter
   *  overrides.  A user who changed them keeps the block visible even with
   *  Advanced collapsed, so their override never hides. */
  get isDefaultCleanupSelected(): boolean {
    const wanted = this.cleaners().filter((c) => c.default_enabled);
    const selected = this.selectedCleaners();
    if (selected.length !== wanted.length) return false;
    return wanted.every((c) => {
      const entry = selected.find((s) => s.name === c.name);
      return !!entry && Object.keys(entry.params || {}).length === 0;
    });
  }

  /** The Advanced toggle is shown either when the Include-media block
   *  is available (it lives strictly behind the toggle, so the toggle
   *  must be reachable) or when no picker in the block has been
   *  overridden (otherwise those pickers stay visible anyway and the
   *  toggle would be redundant). */
  get showAdvancedToggle(): boolean {
    if (this.showSourceSpecs()) return true;
    return this.isDefaultEmbedderSelected && this.isDefaultClipperSelected && this.isDefaultCleanupSelected;
  }

  /** Embedder picker is visible when Advanced is open OR when the user
   *  has picked a non-default embedder, and never when a Solo
   *  mediaEmbedder is locked for the current mediaType (then the user
   *  has explicitly opted out of seeing the picker for this type). */
  get showEmbedderPicker(): boolean {
    if (this.lockedEmbedder()) return false;
    return this.advancedOpen || !this.isDefaultEmbedderSelected;
  }

  /** Clipper picker is visible when Advanced is open OR when the user
   *  has picked a non-default clipper. */
  get showClipperPicker(): boolean {
    return this.advancedOpen || !this.isDefaultClipperSelected;
  }

  /** Cleanup block is visible when Advanced is open OR when the user has
   *  changed the cleanup selection away from the registry defaults. */
  get showCleanupSection(): boolean {
    return this.cleaners().length > 0 && (this.advancedOpen || !this.isDefaultCleanupSelected);
  }

  /** Human label for a cleaner's checkbox row. */
  cleanerLabel(cleaner: CleanerInfo): string {
    return cleaner.display_name || cleaner.name;
  }

  /** Parameter descriptors to render under a checked cleaner.  Cleaners
   *  without parameters render the checkbox alone. */
  cleanerParameters(cleaner: CleanerInfo): ClipperParameter[] {
    return cleaner.parameters || [];
  }

  isCleanerEnabled(name: string): boolean {
    return this.selectedCleaners().some((c) => c.name === name);
  }

  /** Current value for one parameter of an enabled cleaner, falling back to
   *  the descriptor's default when the user has not overridden it. */
  cleanerParamValue(cleaner: CleanerInfo, param: ClipperParameter): number | string {
    const entry = this.selectedCleaners().find((c) => c.name === cleaner.name);
    const override = entry?.params?.[param.key];
    return override === undefined ? param.default : override;
  }

  /** Add or remove a cleaner from the selection.  The emitted list is kept in
   *  registry order so the summary reads the same way the gates will run,
   *  even though the server ignores the order it is sent. */
  onCleanerToggle(cleaner: CleanerInfo, enabled: boolean): void {
    const current = this.selectedCleaners();
    const next = enabled
      ? [...current, { name: cleaner.name, params: {} }]
      : current.filter((c) => c.name !== cleaner.name);
    const order = this.cleaners().map((c) => c.name);
    next.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
    this.selectedCleanersChange.emit(next);
  }

  /** Record a parameter override on an already-enabled cleaner.  A value equal
   *  to the descriptor's default is dropped rather than stored, so an untouched
   *  cleaner keeps an empty ``params`` and still counts as default. */
  onCleanerParamChange(cleaner: CleanerInfo, param: ClipperParameter, value: number | string): void {
    this.selectedCleanersChange.emit(
      this.selectedCleaners().map((c) => {
        if (c.name !== cleaner.name) return c;
        const params = { ...(c.params || {}) };
        if (value === param.default || value === '' || value === null || value === undefined) {
          delete params[param.key];
        } else {
          params[param.key] = value;
        }
        return { ...c, params };
      }),
    );
  }

  /** Whether to render the standalone clipper "Details" button below
   *  the Advanced block.  When the source-specs column is visible
   *  (Advanced open AND :prop:`hasSourceSpecsColumn`), the native row in
   *  the column hosts its own Details button, so the standalone one
   *  would be redundant.  In every other case where the user should be
   *  able to reach the clipper chooser (Advanced collapsed but a
   *  non-default clipper is in effect, or importers with no
   *  source-specs column at all (demo form)), we
   *  fall back to this standalone button so the override stays visible
   *  and the chooser stays reachable. */
  get showStandaloneClipperButton(): boolean {
    if (this.clippers().length <= 1) return false;
    if (!this.showClipperPicker) return false;
    return !(this.advancedOpen && this.hasSourceSpecsColumn);
  }

  /** Embedders flagged ``is_default`` for the active media type; shown
   *  in the Recommended optgroup. */
  get recommendedEmbedders(): EmbedderInfo[] {
    return this.embedders().filter((e) => e.is_default);
  }

  /** Non-default embedders; shown in the Advanced optgroup inside the
   *  embedder select. */
  get advancedEmbedderOptions(): EmbedderInfo[] {
    return this.embedders().filter((e) => !e.is_default);
  }

  /** Human label for an embedder's option element. */
  embedderLabel(embedder: EmbedderInfo): string {
    return embedder.display_name || embedder.name;
  }

  /** License-warning string for the current embedder, or null when the
   *  embedder has no special licensing concerns. */
  get licenseNotice(): string | null {
    return this.noticeFor(this.selectedEmbedder());
  }

  /** Patch-capable embedders (region search / voting); the optional v3 "Region
   *  embedder" picker's options. */
  get patchEmbedderOptions(): EmbedderInfo[] {
    return this.embedders().filter((e) => e.supports_patch_regions);
  }

  /** Structural embedders (instance matching); the optional v3 "Instance
   *  embedder" picker's options. */
  get structuralEmbedderOptions(): EmbedderInfo[] {
    return this.embedders().filter((e) => e.supports_geometric_verification);
  }

  /** License notice for the chosen patch / structural embedder, or null. */
  get patchLicenseNotice(): string | null {
    return this.noticeFor(this.selectedPatchEmbedder());
  }

  get structuralLicenseNotice(): string | null {
    return this.noticeFor(this.selectedStructuralEmbedder());
  }

  private noticeFor(name: string): string | null {
    if (!name) return null;
    return this.embedders().find((e) => e.name === name)?.license_notice ?? null;
  }

  /** Display name for the currently selected clipper.  Defaults to
   *  ``"None"`` for unset or ``*_default`` clippers (which mean "no
   *  pre-processing"). */
  get clipperDisplayName(): string {
    const clipper = this.clippers().find((c) => c.name === this.selectedClipper());
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

  /** Fired by the optional region-embedder select on user pick. */
  onPatchEmbedderChange(name: string): void {
    this.selectedPatchEmbedderChange.emit(name);
  }

  /** Fired by the optional instance-embedder select on user pick. */
  onStructuralEmbedderChange(name: string): void {
    this.selectedStructuralEmbedderChange.emit(name);
  }

  /** Fired by the clipper button to request the parent's chooser. */
  onClipperClick(): void {
    this.clipperChooserRequested.emit();
  }

  /** Fired by the projection checkbox on user toggle. */
  onBuildProjectionChange(value: boolean): void {
    this.buildProjectionChange.emit(value);
  }

  /** Fired by the merge-near-duplicates checkbox on user toggle. */
  onMergeNearDuplicatesChange(value: boolean): void {
    this.mergeNearDuplicatesChange.emit(value);
  }
}
