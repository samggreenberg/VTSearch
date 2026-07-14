import { ChangeDetectionStrategy, Component, inject, Input, OnDestroy, OnInit, output, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { forkJoin, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { SettingsImporterModalComponent } from '../settings-importer-modal/settings-importer-modal.component';
import { SettingsExporterModalComponent } from '../settings-exporter-modal/settings-exporter-modal.component';
import { ImportDefaultsSettingsComponent } from './import-defaults/import-defaults-settings.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import {
  AutoFindExporterChange,
  AutoFindSettingsComponent,
} from './auto-find/auto-find-settings.component';
import { ImportDefaultsByMediaType } from '../../../models/api.models';
import { SettingsApiService } from '../../../services/settings-api.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { HuggingFaceAuthService } from '../../../services/huggingface-auth.service';
import { DatasetsListingsApiService } from '../../../services/datasets-listings-api.service';
import type { AppSettings } from '../../../generated/api-client/models/app-settings';
import { EmbedderInfo, MediaTypeInfo } from '../../../models/api.models';
import { Theme, ThemeService } from '../../../services/theme.service';
import { formatVersion } from '../../../utils/format-date';
import { VtDialogService } from '../../../services/dialog.service';
import {
  browserPrefersReducedMotion,
  onBrowserReducedMotionChange,
} from '../../../utils/reduced-motion';
import {
  DEFAULT_THUMBNAIL_BORDER,
  MAX_THUMBNAIL_BORDER,
} from '../../browse-canvas/hex-render.util';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-settings-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, IconComponent, SettingsImporterModalComponent, SettingsExporterModalComponent, ImportDefaultsSettingsComponent, FieldHintIconComponent, AutoFindSettingsComponent],
  templateUrl: './settings-modal.component.html',
  styleUrl: './settings-modal.component.scss',
})
export class SettingsModalComponent implements OnInit, OnDestroy {
  private settingsApi = inject(SettingsApiService);
  private settingsState = inject(SettingsStateService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private themeService = inject(ThemeService);
  private dialog = inject(VtDialogService);
  private hfAuth = inject(HuggingFaceAuthService);

  /** HuggingFace sign-in state, surfaced to the "HuggingFace" settings tab. */
  readonly hfStatus = this.hfAuth.status;

  @Input() preselectedViewTab = '';
  readonly closed = output<void>();

  readonly settings = signal<AppSettings>({ volume: 50 });
  readonly mediaTypes = signal<MediaTypeInfo[]>([]);
  /** All registered embedders, keyed by media-type id, used to populate
   *  the per-mediaType "Solo embedder" dropdowns under Appearance. */
  readonly embeddersByType = signal<Record<string, EmbedderInfo[]>>({});
  readonly activeSettingsTab = signal('appearance');
  readonly activeViewTab = signal('');
  /** Active media-type tab within the Browser settings tab. */
  readonly activeBrowseTab = signal('');
  /** Upper bound for the pile-thumbnail border number input (template-bound). */
  readonly maxThumbnailBorder = MAX_THUMBNAIL_BORDER;
  readonly loading = signal(true);
  readonly error = signal('');
  showImporterModal = false;
  showExporterModal = false;
  readonly version = signal('');
  readonly savedVisible = signal(false);
  private savedTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   * Whether the browser/OS is suppressing motion via `prefers-reduced-motion`.
   * Surfaced as the "Browser motion" status line under the Show Animations
   * toggle so a user whose animations vanished can see the block is coming from
   * their OS/browser (which overrides the toggle), not from VTSearch. Tracks
   * live changes to the OS setting via `onBrowserReducedMotionChange`.
   */
  readonly browserBlocksMotion = signal(browserPrefersReducedMotion());
  private reducedMotionCleanup: (() => void) | null = null;

  private destroy$ = new Subject<void>();

  ngOnDestroy(): void {
    if (this.savedTimer !== null) clearTimeout(this.savedTimer);
    this.reducedMotionCleanup?.();
    this.destroy$.next();
    this.destroy$.complete();
  }

  ngOnInit(): void {
    this.hfAuth.refresh();

    // Keep the "Browser motion" status line live if the user flips their OS
    // reduce-motion setting while the modal is open.
    this.browserBlocksMotion.set(browserPrefersReducedMotion());
    this.reducedMotionCleanup = onBrowserReducedMotionChange((reduced) =>
      this.browserBlocksMotion.set(reduced),
    );

    forkJoin({
      settings: this.settingsApi.getSettings(),
      mediaTypes: this.datasetsListingsApi.getMediaTypes(),
      embedders: this.datasetsListingsApi.getEmbedders(),
      version: this.settingsApi.getVersion(),
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
      next: (res) => {
        this.settings.set(res.settings);
        this.version.set(formatVersion(res.version.version));
        this.mediaTypes.set(res.mediaTypes.media_types || []);
        const allEmbedders = res.embedders || [];
        // Group embedders by media_type_id with defaults first, matching
        // the order ``embedders_for_type`` returns from the API.
        const byType: Record<string, EmbedderInfo[]> = {};
        for (const emb of allEmbedders) {
          const tid = emb.media_type_id || '';
          if (!tid) continue;
          (byType[tid] ||= []).push(emb);
        }
        for (const list of Object.values(byType)) {
          list.sort((a, b) => Number(!!b.is_default) - Number(!!a.is_default));
        }
        this.embeddersByType.set(byType);
        const mediaTypes = this.mediaTypes();
        if (mediaTypes.length > 0) {
          const preselected = this.preselectedViewTab;
          if (preselected && mediaTypes.some((mt) => mt.type_id === preselected)) {
            this.activeViewTab.set(preselected);
            this.activeSettingsTab.set('appearance');
          } else {
            this.activeViewTab.set(mediaTypes[0].type_id);
          }
          this.activeBrowseTab.set(mediaTypes[0].type_id);
        }
        this.loading.set(false);
      },
      error: () => {
        this.loading.set(false);
        this.error.set('Failed to load settings');
      },
    });
  }

  /** Start the "Sign in with HuggingFace" OAuth handshake. */
  hfLogin(): void {
    this.hfAuth.login();
  }

  /** Sign out of HuggingFace (drops the server-held token). */
  hfLogout(): void {
    this.hfAuth.logout();
  }

  onThemeChange(theme: string): void {
    const t = theme as Theme;
    this.settings.update((s) => ({ ...s, theme: t }));
    this.themeService.setTheme(t);
    this.save();
  }

  /**
   * Persist the "Show Animations" pulldown. "Show" forces decorative motion on
   * (even against an OS reduce-motion request), "Hide" always suppresses it,
   * and "OS Setting" defers to the platform preference; `SettingsStateService`
   * mirrors the choice onto `<html>` for the global stylesheet.
   */
  onAnimationModeChange(mode: string): void {
    const m = mode as AppSettings['show_animations'];
    this.settings.update((s) => ({ ...s, show_animations: m }));
    this.save();
  }

  /** Value shown in the "Solo media type" select. Empty string means
   *  "Show everything"; otherwise it's the type_id. We display the
   *  user's explicit choice when set, falling back to the CLI's
   *  effective value so a fresh user sees what the streamlined mode is
   *  currently locking them to (rather than a misleading empty state). */
  get soloMediaTypeSelectValue(): string {
    const settings = this.settings();
    const explicit = settings.solo_media_type_explicit;
    if (explicit) {
      return settings.solo_media_type || '';
    }
    return settings.effective_solo_media_type || '';
  }

  /** Hint text under the solo-mediaType select. Surfaces "from
   *  ``--solo-media-type``" when the value comes from the CLI fallback
   *  so the user understands why the picker is non-empty without ever
   *  having touched it. */
  get soloMediaTypeNote(): string {
    const settings = this.settings();
    const explicit = settings.solo_media_type_explicit;
    const effective = settings.effective_solo_media_type || '';
    if (!explicit && effective) {
      return `Currently set to ${effective} by the --solo-media-type CLI flag. ` +
        'Choose any value here to override it.';
    }
    return '';
  }

  /** Always-present help for the solo-mediaType select, spelling out
   *  exactly what the lock constrains (so it isn't a mystery toggle), plus
   *  the CLI-fallback note when one applies. */
  get soloMediaTypeHint(): string {
    const base =
      'Streamlines the whole app to one media type. The Add Dataset picker ' +
      'hides importers and tabs that can’t produce it and preselects it ' +
      'on the ones that can; the New Detector media picker and the Import ' +
      'Defaults tab collapse to just this type; and converters that output ' +
      'other types are filtered out. Pick "Show everything" to turn it off.';
    const note = this.soloMediaTypeNote;
    return note ? `${base} ${note}` : base;
  }

  onSoloMediaTypeChange(value: string): void {
    // Empty string = "Show everything"; the backend stores it as null
    // and still flips the explicit flag so the choice survives a CLI
    // fallback on the next launch.
    const next = value || null;
    this.settings.update((s) => ({
      ...(s as Record<string, unknown>),
      solo_media_type: next,
      solo_media_type_explicit: true,
      effective_solo_media_type: next,
    }) as AppSettings);
    this.save();
  }

  /** Embedder options for a given media type, used by the per-type
   *  "Solo embedder" dropdowns. Returns an empty list when the registry
   *  hasn't loaded yet or the type has no embedders. */
  embeddersForType(typeId: string): EmbedderInfo[] {
    return this.embeddersByType()[typeId] || [];
  }

  /** Currently selected solo embedder name for *typeId*. Reads from the
   *  effective map (user explicit overlaid on the CLI fallback) so the
   *  picker shows what the user will actually get when they open the
   *  importer (including a CLI-only lock that has not been overridden). */
  soloEmbedderSelectValue(typeId: string): string {
    const map = this.settings().effective_solo_embedder_per_media_type || {};
    const value = map[typeId];
    if (!value) return '';
    // If the stored embedder no longer exists for this type, surface
    // "Ask each time" rather than a broken-looking option.
    const valid = this.embeddersByType()[typeId] || [];
    return valid.find((e) => e.name === value) ? value : '';
  }

  /** Hint text under a solo-embedder dropdown: explains a CLI override
   *  ("from --solo-embedder") or a stale embedder reference so the user
   *  understands why the dropdown shows what it does. Returns ``''`` for
   *  the normal case (no lock, or a user-explicit pick). */
  soloEmbedderNote(typeId: string): string {
    const settings = this.settings();
    const userMap = settings.solo_embedder_per_media_type || {};
    const effectiveMap = settings.effective_solo_embedder_per_media_type || {};
    const userVal = userMap[typeId];
    const effective = effectiveMap[typeId];
    if (!effective && !userVal) return '';
    const valid = this.embeddersByType()[typeId] || [];
    if (effective && !valid.find((e) => e.name === effective)) {
      return `Locked embedder "${effective}" is no longer registered for this type. ` +
        'Pick a different embedder to update the lock.';
    }
    if (effective && !userVal) {
      return `Currently set to ${effective} by --solo-embedder. Pick any value here to override it.`;
    }
    return '';
  }

  onSoloEmbedderChange(typeId: string, value: string): void {
    const settings = this.settings();
    const userMap = { ...(settings.solo_embedder_per_media_type || {}) };
    // Empty value = "Ask each time". Persist it as the opt-out sentinel
    // (an empty-string entry) so it overrides any ``--solo-embedder``
    // CLI fallback for this type; same pattern as solo_media_type's
    // explicit-null override of --solo-media-type.
    userMap[typeId] = value || '';
    // Optimistically update the effective map so the dropdown reflects
    // the new choice immediately; the PUT response will replace it with
    // the authoritative server view including any CLI fallback.
    const effective = { ...(settings.effective_solo_embedder_per_media_type || {}) };
    if (value) {
      effective[typeId] = value;
    } else {
      delete effective[typeId];
    }
    this.settings.update((s) => ({
      ...(s as Record<string, unknown>),
      solo_embedder_per_media_type: userMap,
      effective_solo_embedder_per_media_type: effective,
    }) as AppSettings);
    this.save();
  }

  onToggle(key: string, value: boolean): void {
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), [key]: value }) as AppSettings);
    this.save();
  }

  async onToggleEnableAchievements(value: boolean): Promise<void> {
    if (!value) {
      const ok = await this.dialog.confirmDestructive(
        'Turn off achievements?',
        'All achievement counters, tier progress, and unlocks will be reset to zero. The trophy button and unlock pop-ups will be hidden until you turn this back on.',
        'Turn off',
      );
      if (!ok) {
        // Force-rebind to the previous value so the checkbox snaps back.
        this.settings.update((s) => ({ ...s, enable_achievements: true }));
        return;
      }
    }
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), enable_achievements: value }) as AppSettings);
    this.save();
  }

  onGridIconSizeChange(side: 'grid_icon_size_left' | 'grid_icon_size_right', typeId: string, value: string): void {
    const dict = { ...((this.settings()[side] as Record<string, string>) || {}) };
    dict[typeId] = value;
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), [side]: dict }) as AppSettings);
    this.save();
  }

  getGridIconSize(side: 'grid_icon_size_left' | 'grid_icon_size_right', typeId: string): string {
    const dict = this.settings()[side];
    if (!dict) return 'M';
    return dict[typeId] ?? 'M';
  }

  onFocusModeChange(side: 'focus_mode_left' | 'focus_mode_right', typeId: string, value: string): void {
    const dict = { ...((this.settings()[side] as Record<string, string>) || {}) };
    dict[typeId] = value;
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), [side]: dict }) as AppSettings);
    this.save();
  }

  getFocusMode(side: 'focus_mode_left' | 'focus_mode_right', typeId: string): string {
    const dict = this.settings()[side];
    if (!dict) return 'click';
    return dict[typeId] ?? 'click';
  }

  // --- Browser tab: per-media-type projection-browser preferences ---

  /** Read a per-media-type browser setting for *typeId*, or *fallback*. */
  private getBrowsePref(
    key: 'browse_colormap' | 'browse_icon_size',
    typeId: string,
    fallback: string,
  ): string {
    const dict = this.settings()[key] as Record<string, string> | undefined;
    return (dict && dict[typeId]) || fallback;
  }

  /** Write a per-media-type browser setting and persist. */
  private setBrowsePref(
    key: 'browse_colormap' | 'browse_icon_size',
    typeId: string,
    value: string,
  ): void {
    const dict = { ...((this.settings()[key] as Record<string, string> | undefined) || {}) };
    dict[typeId] = value;
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), [key]: dict }) as AppSettings);
    this.save();
  }

  getBrowseColormap(typeId: string): string {
    return this.getBrowsePref('browse_colormap', typeId, 'auto');
  }

  onBrowseColormapChange(typeId: string, value: string): void {
    this.setBrowsePref('browse_colormap', typeId, value);
  }

  /**
   * True when *typeId* paints cells with thumbnails (image/video). These types
   * are pinned to grayscale, so the colormap picker is hidden for them and the
   * UI shows a fixed read-only value instead. Data-driven from the served
   * ``has_thumbnail`` field on the media types this modal already loaded.
   */
  browseTabUsesThumbnails(typeId: string): boolean {
    return this.mediaTypes().some((mt) => mt.type_id === typeId && mt.has_thumbnail === true);
  }

  getBrowseIconSize(typeId: string): string {
    return this.getBrowsePref('browse_icon_size', typeId, 'M');
  }

  onBrowseIconSizeChange(typeId: string, value: string): void {
    this.setBrowsePref('browse_icon_size', typeId, value);
  }

  /** Pile-thumbnail border width (CSS px) for *typeId*, defaulting to the
   *  feature default when the user hasn't set one for this media type. */
  getBrowseThumbnailBorder(typeId: string): number {
    const dict = this.settings().browse_thumbnail_border as Record<string, number> | undefined;
    const value = dict?.[typeId];
    return value == null ? DEFAULT_THUMBNAIL_BORDER : value;
  }

  /** Write the per-media-type pile-thumbnail border width, clamped to the
   *  shared 0..MAX range, and persist. */
  onBrowseThumbnailBorderChange(typeId: string, value: number | string): void {
    let n = typeof value === 'string' ? parseInt(value, 10) : value;
    if (!Number.isFinite(n)) n = 0;
    n = Math.max(0, Math.min(MAX_THUMBNAIL_BORDER, Math.round(n)));
    const dict = {
      ...((this.settings().browse_thumbnail_border as Record<string, number> | undefined) || {}),
    };
    dict[typeId] = n;
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), browse_thumbnail_border: dict }) as AppSettings);
    this.save();
  }

  /** Mouse-zooms per pyramid level for *typeId* (1..3), defaulting to 2 when
   *  the user hasn't set one for this media type. This is how many wheel notches
   *  / +/- clicks it takes to cross one pyramid level. */
  getBrowseMouseZoomsPerLevel(typeId: string): number {
    const dict = this.settings().browse_mouse_zooms_per_level as Record<string, number> | undefined;
    const value = dict?.[typeId];
    return value == null ? 2 : value;
  }

  /** Write the per-media-type mouse-zooms-per-level (clamped 1..3) and persist. */
  onBrowseMouseZoomsPerLevelChange(typeId: string, value: number | string): void {
    let n = typeof value === 'string' ? parseInt(value, 10) : value;
    if (!Number.isFinite(n)) n = 2;
    n = Math.max(1, Math.min(3, Math.round(n)));
    const dict = {
      ...((this.settings().browse_mouse_zooms_per_level as Record<string, number> | undefined) || {}),
    };
    dict[typeId] = n;
    this.settings.update(
      (s) => ({ ...(s as Record<string, unknown>), browse_mouse_zooms_per_level: dict }) as AppSettings,
    );
    this.save();
  }

  /** Whether the projection is compacted for *typeId* (oceans closed),
   *  defaulting to on when the user hasn't set one for this media type. */
  getBrowseCompact(typeId: string): boolean {
    const dict = this.settings().browse_compact as Record<string, boolean> | undefined;
    const value = dict?.[typeId];
    return value == null ? true : value;
  }

  /** Write the per-media-type compaction toggle and persist. Takes effect on
   *  the next projection build or the Browser's Re-project action, since the
   *  layout coordinates are computed once and frozen. */
  onBrowseCompactChange(typeId: string, value: boolean): void {
    const dict = { ...((this.settings().browse_compact as Record<string, boolean> | undefined) || {}) };
    dict[typeId] = value;
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), browse_compact: dict }) as AppSettings);
    this.save();
  }

  /** Whether the browse canvas draws region signposts (the named "street sign"
   *  labels over the map) for *typeId*, defaulting to on when unset. */
  getBrowseSignposts(typeId: string): boolean {
    const dict = this.settings().browse_signposts as Record<string, boolean> | undefined;
    const value = dict?.[typeId];
    return value == null ? true : value;
  }

  /** Write the per-media-type signposts toggle and persist. The same map the
   *  signpost button on the browse canvas writes, so the two stay in step. */
  onBrowseSignpostsChange(typeId: string, value: boolean): void {
    const dict = { ...((this.settings().browse_signposts as Record<string, boolean> | undefined) || {}) };
    dict[typeId] = value;
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), browse_signposts: dict }) as AppSettings);
    this.save();
  }

  // --- Browser tab: right-click bin-popup thumbnail size ---
  // The bin popup keeps its own per-media-type thumbnail size
  // (``grid_icon_size_popup``), independent of the left/right panels. The
  // popup's in-header size buttons write the same map, so this widget and the
  // live popup stay in step. (The popup is grid-only — it has no list mode.)

  getPopupGridIconSize(typeId: string): string {
    const dict = this.settings().grid_icon_size_popup as Record<string, string> | undefined;
    return (dict && dict[typeId]) || 'M';
  }

  onPopupGridIconSizeChange(typeId: string, value: string): void {
    const dict = { ...((this.settings().grid_icon_size_popup as Record<string, string> | undefined) || {}) };
    dict[typeId] = value;
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), grid_icon_size_popup: dict }) as AppSettings);
    this.save();
  }

  onNumberChange(key: string, value: number): void {
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), [key]: value }) as AppSettings);
    this.save();
  }

  onStringChange(key: string, value: string): void {
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), [key]: value }) as AppSettings);
    this.save();
  }

  /** Current per-mediaType import-defaults map, normalised to a plain
   *  object so the child component never has to defend against ``null``. */
  get importDefaults(): ImportDefaultsByMediaType {
    const raw = (this.settings() as Record<string, unknown>)['import_defaults_by_media_type'];
    return (raw as ImportDefaultsByMediaType | undefined) || {};
  }

  onImportDefaultsChange(value: ImportDefaultsByMediaType): void {
    this.settings.update((s) => ({ ...(s as Record<string, unknown>), import_defaults_by_media_type: value }) as AppSettings);
    this.save();
  }

  /** Effective solo-mediaType for the import-defaults tab: collapses
   *  the per-mediaType picker to a single tab when the user is in solo
   *  mode (so they only configure what they'll actually import). */
  get effectiveSoloMediaType(): string | null {
    const settings = this.settings();
    const explicit = settings.solo_media_type_explicit;
    if (explicit) {
      return settings.solo_media_type || null;
    }
    return settings.effective_solo_media_type || null;
  }

  /** Configured Auto-Find results-exporter name (''=none), read from the
   *  settings object for the Auto-Find tab's child component. */
  get autofindExporter(): string {
    return ((this.settings() as Record<string, unknown>)['autofind_exporter'] as string) || '';
  }

  /** Per-exporter saved field values for the Auto-Find tab's child component. */
  get autofindExporterFieldValues(): Record<string, Record<string, string>> {
    return (
      ((this.settings() as Record<string, unknown>)['autofind_exporter_field_values'] as Record<
        string,
        Record<string, string>
      >) || {}
    );
  }

  /** Persist the user's Auto-Find results-exporter choice + field values. */
  onAutofindExporterChange(change: AutoFindExporterChange): void {
    this.settings.update((s) => ({
      ...(s as Record<string, unknown>),
      autofind_exporter: change.exporter,
      autofind_exporter_field_values: change.fieldValues,
    }) as AppSettings);
    this.save();
  }

  /** Effective hidden-plugins map flattened to ``[{family, names}]`` rows
   *  (sorted by family, empty families dropped) for the read-only Server
   *  tab. ``names`` is the comma-joined plugin names within the family. */
  get hiddenPluginsDisplay(): { family: string; names: string }[] {
    const raw = (this.settings() as Record<string, unknown>)['hidden_plugins'] as
      | Record<string, string[]>
      | undefined;
    if (!raw) return [];
    return Object.keys(raw)
      .filter((family) => (raw[family] || []).length > 0)
      .sort()
      .map((family) => ({ family, names: (raw[family] || []).join(', ') }));
  }

  async resetDefaults(): Promise<void> {
    const choice = await this.dialog.confirmDestructiveWithEscape(
      'Reset all settings to factory defaults?',
      'Your current preferences (appearance, view modes, autopilot, sorting, and other per-user settings) will be overwritten and cannot be recovered.',
      'Reset',
      'Export first…',
    );
    if (choice === 'cancel') return;
    if (choice === 'escape') {
      // Escape hatch: let the user save their current config before resetting.
      // Reset is abandoned; they can re-initiate it after exporting.
      this.openExporter();
      return;
    }
    this.settingsApi
      .getDefaults()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (defaults) => {
          this.settings.set(defaults);
          if (defaults.theme) {
            this.themeService.setTheme(defaults.theme as Theme);
          }
          this.save();
        },
      });
  }


  openImporter(): void {
    this.showImporterModal = true;
  }

  onImportComplete(): void {
    // Reload settings from the server after import
    this.settingsApi
      .getSettings()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (s) => {
          this.settings.set(s);
          if (s.theme) {
            this.themeService.setTheme(s.theme as Theme);
          }
        },
      });
  }

  openExporter(): void {
    this.showExporterModal = true;
  }

  private save(): void {
    this.settingsState
      .update(this.settings())
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.error.set('');
          this.flashSaved();
        },
        error: () => {
          this.error.set('Failed to save settings');
        },
      });
  }

  private flashSaved(): void {
    if (this.savedTimer !== null) clearTimeout(this.savedTimer);
    this.savedVisible.set(true);
    this.savedTimer = setTimeout(() => {
      this.savedVisible.set(false);
      this.savedTimer = null;
    }, 1800);
  }

  close(): void {
    this.closed.emit();
  }
}
