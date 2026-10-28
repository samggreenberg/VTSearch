import { ChangeDetectionStrategy, Component, computed, DestroyRef, inject, input, OnDestroy, OnInit, output, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
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
import { MediaTypeInfo } from '../../../models/api.models';
import { Theme, ThemeService } from '../../../services/theme.service';
import { formatVersion } from '../../../utils/format-date';
import { BuildSkewService } from '../../../services/build-skew.service';
import { VtDialogService } from '../../../services/dialog.service';
import {
  browserPrefersReducedMotion,
  onBrowserReducedMotionChange,
} from '../../../utils/reduced-motion';
import {
  DEFAULT_THUMBNAIL_BORDER,
  MAX_THUMBNAIL_BORDER,
} from '../../browse-canvas/hex-render.util';

/** The Settings rail's panes. */
export type SettingsTabId =
  | 'general'
  | 'panels'
  | 'import'
  | 'browse'
  | 'training'
  | 'autofind'
  | 'huggingface'
  | 'server';

interface SettingsTab {
  id: SettingsTabId;
  label: string;
  icon: string;
  title: string;
  /** Draw the rail's divider above this tab. */
  dividerBefore?: boolean;
}

/**
 * The rail, in the order a user meets the work: the app as a whole, the panels
 * of Train, Tune and Find, then importing, browsing, training and AutoFind.
 * Below the divider sit the panes that are not preferences: the HuggingFace
 * account and the server's read-only values (#5246).
 */
const SETTINGS_TABS: readonly SettingsTab[] = [
  { id: 'general', label: 'General', icon: 'palette', title: "Theme, animations, the Dashboard's usage bars, achievements, and Toasty's hints" },
  { id: 'panels', label: 'Panels', icon: 'dock-left', title: 'Which panels Train, Tune and Find open with, and how their item lists show and select items' },
  { id: 'import', label: 'Import', icon: 'import', title: 'Default embedder, clipper, and converters to apply when importing each kind of dataset' },
  { id: 'browse', label: 'Browse Map', icon: 'eye', title: 'How the Browse map looks per media type: signposts, colormap, cell size, and zoom' },
  { id: 'training', label: 'Training', icon: 'graduation', title: "Autopilot's thresholds, text-sort phrasing, and how training places the line" },
  { id: 'autofind', label: 'AutoFind', icon: 'auto-find', title: 'Detectors that auto-run on import, and what to do with their results' },
  { id: 'huggingface', label: 'HuggingFace', icon: 'globe', title: 'Sign in with HuggingFace to download gated demo datasets and gated AI models', dividerBefore: true },
  { id: 'server', label: 'Server', icon: 'server', title: 'Settings fixed when the server started; shared by everyone and read-only here' },
];

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

  readonly preselectedViewTab = input('');
  /** The rail tab to open on ('' = General), for a caller that opens Settings
   *  for one purpose (#5344's AutoFind prompt). */
  readonly initialTab = input<SettingsTabId | ''>('');
  /** Open the settings importer over the modal straight away. */
  readonly startWithImporter = input(false);
  readonly closed = output<void>();
  /** A settings import was applied. */
  readonly imported = output<void>();

  readonly settings = signal<AppSettings>({ volume: 50 });
  readonly mediaTypes = signal<MediaTypeInfo[]>([]);
  /** The media types the per-type tab bars (Scroll Style, Browse Map) offer: just
   *  the server's one type when it is locked to one (#5106), so a bar with
   *  nothing to switch to drops out, as it does for a lone registered type. */
  readonly tabMediaTypes = computed(() => {
    const solo = this.effectiveSoloMediaType;
    const all = this.mediaTypes();
    const only = solo ? all.filter((mt) => mt.type_id === solo) : [];
    return only.length ? only : all;
  });
  readonly tabs = SETTINGS_TABS;
  readonly activeSettingsTab = signal<SettingsTabId>('general');
  readonly activeViewTab = signal('');
  /** Active media-type tab within the Browse Map pane. */
  readonly activeBrowseTab = signal('');
  /** Upper bound for the pile-thumbnail border number input (template-bound). */
  readonly maxThumbnailBorder = MAX_THUMBNAIL_BORDER;
  readonly loading = signal(true);
  readonly error = signal('');
  showImporterModal = false;
  showExporterModal = false;
  readonly version = signal('');
  private readonly buildSkew = inject(BuildSkewService);
  /**
   * The bundle's own build stamp, shown *only* when it disagrees with the
   * server's. The `version` line above is the server's, so on a stale `static/`
   * build it reads current while the browser runs older code (#2898); this is
   * the line that says so where someone reporting a bug will actually look.
   */
  readonly staleBundleVersion = computed(() =>
    this.buildSkew.skewed() ? formatVersion(this.buildSkew.bundleVersion) : '',
  );
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

  private readonly destroyRef = inject(DestroyRef);

  ngOnDestroy(): void {
    if (this.savedTimer !== null) clearTimeout(this.savedTimer);
    this.reducedMotionCleanup?.();
  }

  ngOnInit(): void {
    if (this.initialTab()) this.activeSettingsTab.set(this.initialTab() as SettingsTabId);
    if (this.startWithImporter()) this.showImporterModal = true;
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
      version: this.settingsApi.getVersion(),
    })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
      next: (res) => {
        this.settings.set(res.settings);
        this.version.set(formatVersion(res.version.version));
        // Feed the same reading to the skew service, so the stale-bundle chip
        // is correct even if the startup check hasn't landed (or failed).
        this.buildSkew.serverVersion.set((res.version.version || '').trim());
        this.mediaTypes.set(res.mediaTypes.media_types || []);
        const mediaTypes = this.tabMediaTypes();
        if (mediaTypes.length > 0) {
          const preselected = this.preselectedViewTab();
          if (preselected && mediaTypes.some((mt) => mt.type_id === preselected)) {
            this.activeViewTab.set(preselected);
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

  /** Persist the "RAM / Disk bars" pulldown: when the Dashboard shows its
   *  server usage bars ("Default" = once a detector exists; see
   *  `DashboardComponent.showUsageBars`). */
  onUsageBarsModeChange(mode: string): void {
    const m = mode as AppSettings['show_usage_bars'];
    this.settings.update((s) => ({ ...s, show_usage_bars: m }));
    this.save();
  }

  /** Display name of the admin-set solo media type for the read-only
   *  Server settings row. Falls back to the raw type_id when the registry
   *  hasn't loaded (or no longer carries the type), and to "Show
   *  everything" when no restriction is in force. */
  get soloMediaTypeDisplay(): string {
    const value = this.effectiveSoloMediaType;
    if (!value) return 'Show everything';
    return this.mediaTypes().find((mt) => mt.type_id === value)?.name || value;
  }

  /** Toasty's hints (#4680): Hide All has nothing to do once "Hide all
   *  hints" is on, and Show All has nothing to do while no hint is hidden,
   *  by hand or by a first use of the feature it teaches (#5360). */
  readonly allHintsHidden = computed(() => this.settings().hide_all_hints === true);
  readonly noHintsHidden = computed(
    () =>
      !this.settings().hide_all_hints &&
      (this.settings().hidden_hints ?? []).length === 0 &&
      (this.settings().used_features ?? []).length === 0,
  );

  /** Hide All: the same switch as a hint's "Hide all hints" box. */
  onHideAllHints(): void {
    this.settings.update((s) => ({ ...s, hide_all_hints: true }));
    this.save();
  }

  /** Show All: every hint back, including ones hidden one at a time and ones
   *  a feature's first use retired. */
  onShowAllHints(): void {
    this.settings.update((s) => ({ ...s, hide_all_hints: false, hidden_hints: [], used_features: [] }));
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

  // --- Browse Map pane: per-media-type projection-browser preferences ---

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

  /**
   * Canvas rendering effort for the Browse map. Unlike the rest of the Browser
   * tab this is one global scalar, not a per-media-type map: it describes the
   * client's rendering capability, not anything about the data. Falls back to
   * ``auto`` (detect per client) for unset or unrecognized values.
   */
  get browseGraphics(): 'auto' | 'full' | 'reduced' {
    const value = this.settings().browse_graphics;
    return value === 'full' || value === 'reduced' ? value : 'auto';
  }

  /**
   * Persist the Graphics pulldown. ``reduced`` keeps every browse animation but
   * drops the effects a CPU rasterizer can't afford (smoothed blits, overscan
   * buffers, shadow blurs); ``full`` always runs the rich pipeline; ``auto``
   * probes the client. The canvas picks the change up live.
   */
  onBrowseGraphicsChange(mode: string): void {
    const m = mode as AppSettings['browse_graphics'];
    this.settings.update((s) => ({ ...s, browse_graphics: m }));
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

  /** Media types that ship a generative signpost captioner (image VLM / audio
   *  captioner). Others only ever use the zero-shot tag texts, so the source
   *  toggle is hidden for them. Mirrors the backend `_CAPTIONERS` registry in
   *  `vtscore/projection/signpost_texts.py`. */
  browseTabHasCaptioner(typeId: string): boolean {
    return typeId === 'image' || typeId === 'audio';
  }

  /** Whether *typeId*'s signpost texts come from the generative captioner
   *  (VLM / audio captioner) instead of the zero-shot tags. Defaults to tags
   *  (false), since the captioner downloads a multi-GB model. */
  getBrowseSignpostCaptioner(typeId: string): boolean {
    const dict = this.settings().browse_signpost_captioner as Record<string, boolean> | undefined;
    return dict?.[typeId] ?? false;
  }

  /** Write the per-media-type signpost-text-source toggle and persist. Takes
   *  effect on the next projection build / Re-project, since signpost texts are
   *  computed and cached at build time. */
  onBrowseSignpostCaptionerChange(typeId: string, value: boolean): void {
    const dict = { ...((this.settings().browse_signpost_captioner as Record<string, boolean> | undefined) || {}) };
    dict[typeId] = value;
    this.settings.update(
      (s) => ({ ...(s as Record<string, unknown>), browse_signpost_captioner: dict }) as AppSettings,
    );
    this.save();
  }

  // --- Browse Map pane: right-click bin-popup thumbnail size ---
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

  // ``null`` is a legal value for nullable numeric settings: clearing the
  // Held-out fraction input writes null, which the server stores as "no
  // explicit split" (the per-embedder automatic default applies).
  onNumberChange(key: string, value: number | null): void {
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

  /** The admin-set solo mediaType in force, or ``null`` when the server
   *  isn't restricting one. The Import pane uses it to collapse
   *  the per-mediaType picker to a single tab (so the user only configures
   *  what they can actually import). */
  get effectiveSoloMediaType(): string | null {
    return this.settings().solo_media_type || null;
  }

  /** Configured AutoFind results-exporter name (''=none), read from the
   *  settings object for the AutoFind tab's child component. */
  get autofindExporter(): string {
    return ((this.settings() as Record<string, unknown>)['autofind_exporter'] as string) || '';
  }

  /** Per-exporter saved field values for the AutoFind tab's child component. */
  get autofindExporterFieldValues(): Record<string, Record<string, string>> {
    return (
      ((this.settings() as Record<string, unknown>)['autofind_exporter_field_values'] as Record<
        string,
        Record<string, string>
      >) || {}
    );
  }

  /** Persist the user's AutoFind results-exporter choice + field values. */
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

  /** How many signpost terms the Server tab lists inline before eliding the
   *  rest. An operator vocabulary can run to hundreds of terms (the backend
   *  caps it at 2000), which would otherwise bury the rest of the tab. */
  private static readonly SIGNPOST_VOCAB_PREVIEW_TERMS = 12;

  /** The server's custom signpost vocabulary flattened to
   *  ``[{mediaType, terms}]`` rows (sorted by media type, empty entries
   *  dropped) for the read-only Server tab. Long lists are elided to the first
   *  {@link SIGNPOST_VOCAB_PREVIEW_TERMS} terms plus a total count. */
  get signpostVocabDisplay(): { mediaType: string; terms: string }[] {
    const raw = (this.settings() as Record<string, unknown>)['browse_signpost_vocab'] as
      | Record<string, string[]>
      | undefined;
    if (!raw) return [];
    const limit = SettingsModalComponent.SIGNPOST_VOCAB_PREVIEW_TERMS;
    return Object.keys(raw)
      .filter((mediaType) => (raw[mediaType] || []).length > 0)
      .sort()
      .map((mediaType) => {
        const all = raw[mediaType] || [];
        const shown = all.slice(0, limit).join(', ');
        return {
          mediaType,
          terms: all.length > limit ? `${shown} … (${all.length} terms)` : shown,
        };
      });
  }

  async resetDefaults(): Promise<void> {
    const choice = await this.dialog.confirmDestructiveWithEscape(
      'Reset all settings to factory defaults?',
      'Your current preferences (theme, panels, Autopilot, training, and other per-user settings) will be overwritten and cannot be recovered.',
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
      .pipe(takeUntilDestroyed(this.destroyRef))
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
    // Applied server-side already, so say so now: closing before the reload
    // below lands would cancel it.
    this.imported.emit();
    // Reload settings from the server after import
    this.settingsApi
      .getSettings()
      .pipe(takeUntilDestroyed(this.destroyRef))
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
      .pipe(takeUntilDestroyed(this.destroyRef))
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
