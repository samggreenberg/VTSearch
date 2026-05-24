import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { SettingsImporterModalComponent } from '../settings-importer-modal/settings-importer-modal.component';
import { SettingsExporterModalComponent } from '../settings-exporter-modal/settings-exporter-modal.component';
import { ImportDefaultsSettingsComponent } from './import-defaults/import-defaults-settings.component';
import { ImportDefaultsByMediaType } from '../../../models/api.models';
import { SettingsApiService } from '../../../services/settings-api.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import type { AppSettings } from '../../../generated/api-client/models/app-settings';
import { EmbedderInfo, MediaTypeInfo } from '../../../models/api.models';
import { Theme, ThemeService } from '../../../services/theme.service';
import { formatVersion } from '../../../utils/format-date';
import { VtDialogService } from '../../../services/dialog.service';

@Component({
  selector: 'vt-settings-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent, SettingsImporterModalComponent, SettingsExporterModalComponent, ImportDefaultsSettingsComponent],
  templateUrl: './settings-modal.component.html',
  styleUrl: './settings-modal.component.scss',
})
export class SettingsModalComponent implements OnInit, OnDestroy {
  @Input() preselectedViewTab = '';
  @Output() closed = new EventEmitter<void>();

  settings: AppSettings = { volume: 50 };
  mediaTypes: MediaTypeInfo[] = [];
  /** All registered embedders, keyed by media-type id, used to populate
   *  the per-mediaType "Solo embedder" dropdowns under Appearance. */
  embeddersByType: Record<string, EmbedderInfo[]> = {};
  activeSettingsTab = 'appearance';
  activeViewTab = '';
  loading = true;
  error = '';
  showImporterModal = false;
  showExporterModal = false;
  version = '';
  savedVisible = false;
  private savedTimer: ReturnType<typeof setTimeout> | null = null;

  private destroy$ = new Subject<void>();

  constructor(
    private settingsApi: SettingsApiService,
    private settingsState: SettingsStateService,
    private datasetsApi: DatasetsApiService,
    private themeService: ThemeService,
    private dialog: VtDialogService,
  ) {}

  ngOnDestroy(): void {
    if (this.savedTimer !== null) clearTimeout(this.savedTimer);
    this.destroy$.next();
    this.destroy$.complete();
  }

  ngOnInit(): void {
    forkJoin({
      settings: this.settingsApi.getSettings(),
      mediaTypes: this.datasetsApi.getMediaTypes(),
      embedders: this.datasetsApi.getEmbedders(),
      version: this.settingsApi.getVersion(),
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
      next: (res) => {
        this.settings = res.settings;
        this.version = formatVersion(res.version.version);
        this.mediaTypes = res.mediaTypes.media_types || [];
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
        this.embeddersByType = byType;
        if (this.mediaTypes.length > 0) {
          const preselected = this.preselectedViewTab;
          if (preselected && this.mediaTypes.some((mt) => mt.type_id === preselected)) {
            this.activeViewTab = preselected;
            this.activeSettingsTab = 'appearance';
          } else {
            this.activeViewTab = this.mediaTypes[0].type_id;
          }
        }
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load settings';
      },
    });
  }

  onThemeChange(theme: string): void {
    const t = theme as Theme;
    this.settings.theme = t;
    this.themeService.setTheme(t);
    this.save();
  }

  /** Value shown in the "Solo media type" select. Empty string means
   *  "Show everything"; otherwise it's the type_id. We display the
   *  user's explicit choice when set, falling back to the CLI's
   *  effective value so a fresh user sees what the streamlined mode is
   *  currently locking them to (rather than a misleading empty state). */
  get soloMediaTypeSelectValue(): string {
    const explicit = this.settings.solo_media_type_explicit;
    if (explicit) {
      return this.settings.solo_media_type || '';
    }
    return this.settings.effective_solo_media_type || '';
  }

  /** Hint text under the solo-mediaType select. Surfaces "from
   *  ``--solo-media-type``" when the value comes from the CLI fallback
   *  so the user understands why the picker is non-empty without ever
   *  having touched it. */
  get soloMediaTypeNote(): string {
    const explicit = this.settings.solo_media_type_explicit;
    const effective = this.settings.effective_solo_media_type || '';
    if (!explicit && effective) {
      return `Currently set to ${effective} by the --solo-media-type CLI flag. ` +
        'Choose any value here to override it.';
    }
    return '';
  }

  onSoloMediaTypeChange(value: string): void {
    // Empty string = "Show everything"; the backend stores it as null
    // and still flips the explicit flag so the choice survives a CLI
    // fallback on the next launch.
    const next = value || null;
    (this.settings as Record<string, unknown>)['solo_media_type'] = next;
    (this.settings as Record<string, unknown>)['solo_media_type_explicit'] = true;
    (this.settings as Record<string, unknown>)['effective_solo_media_type'] = next;
    this.save();
  }

  /** Embedder options for a given media type, used by the per-type
   *  "Solo embedder" dropdowns. Returns an empty list when the registry
   *  hasn't loaded yet or the type has no embedders. */
  embeddersForType(typeId: string): EmbedderInfo[] {
    return this.embeddersByType[typeId] || [];
  }

  /** Currently selected solo embedder name for *typeId*. Reads from the
   *  effective map (user explicit overlaid on the CLI fallback) so the
   *  picker shows what the user will actually get when they open the
   *  importer — including a CLI-only lock that has not been overridden. */
  soloEmbedderSelectValue(typeId: string): string {
    const map = this.settings.effective_solo_embedder_per_media_type || {};
    const value = map[typeId];
    if (!value) return '';
    // If the stored embedder no longer exists for this type, surface
    // "Ask each time" rather than a broken-looking option.
    const valid = this.embeddersByType[typeId] || [];
    return valid.find((e) => e.name === value) ? value : '';
  }

  /** Hint text under a solo-embedder dropdown — explains a CLI override
   *  ("from --solo-embedder") or a stale embedder reference so the user
   *  understands why the dropdown shows what it does. Returns ``''`` for
   *  the normal case (no lock, or a user-explicit pick). */
  soloEmbedderNote(typeId: string): string {
    const userMap = this.settings.solo_embedder_per_media_type || {};
    const effectiveMap = this.settings.effective_solo_embedder_per_media_type || {};
    const userVal = userMap[typeId];
    const effective = effectiveMap[typeId];
    if (!effective && !userVal) return '';
    const valid = this.embeddersByType[typeId] || [];
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
    const userMap = { ...(this.settings.solo_embedder_per_media_type || {}) };
    if (value) {
      userMap[typeId] = value;
    } else {
      // Empty string from the dropdown = "Ask each time" — clear this
      // type's lock. Other types' entries are preserved.
      delete userMap[typeId];
    }
    (this.settings as Record<string, unknown>)['solo_embedder_per_media_type'] = userMap;
    // Optimistically update the effective map so the dropdown reflects
    // the new choice immediately; the PUT response will replace it with
    // the authoritative server view including any CLI fallback.
    const effective = { ...(this.settings.effective_solo_embedder_per_media_type || {}) };
    if (value) {
      effective[typeId] = value;
    } else {
      delete effective[typeId];
    }
    (this.settings as Record<string, unknown>)['effective_solo_embedder_per_media_type'] = effective;
    this.save();
  }

  onToggle(key: string, value: boolean): void {
    (this.settings as Record<string, unknown>)[key] = value;
    this.save();
  }

  async onToggleDisableAchievements(value: boolean): Promise<void> {
    if (value) {
      const ok = await this.dialog.confirmDestructive(
        'Turn off achievements?',
        'All achievement counters, tier progress, and unlocks will be reset to zero. The trophy button and unlock pop-ups will be hidden until you turn this back off.',
        'Turn off',
      );
      if (!ok) {
        // Force-rebind to the previous value so the checkbox snaps back.
        this.settings = { ...this.settings, disable_achievements: false };
        return;
      }
    }
    (this.settings as Record<string, unknown>)['disable_achievements'] = value;
    this.save();
  }

  onViewModeChange(side: 'view_mode_left' | 'view_mode_right', typeId: string, value: string): void {
    const dict = (this.settings[side] as Record<string, string>) || {};
    dict[typeId] = value;
    (this.settings as Record<string, unknown>)[side] = { ...dict };
    this.save();
  }

  getViewMode(side: 'view_mode_left' | 'view_mode_right', typeId: string): string {
    const dict = this.settings[side];
    if (!dict) return side === 'view_mode_left' ? 'list' : 'grid';
    return dict[typeId] ?? (side === 'view_mode_left' ? 'list' : 'grid');
  }

  onGridIconSizeChange(side: 'grid_icon_size_left' | 'grid_icon_size_right', typeId: string, value: string): void {
    const dict = (this.settings[side] as Record<string, string>) || {};
    dict[typeId] = value;
    (this.settings as Record<string, unknown>)[side] = { ...dict };
    this.save();
  }

  getGridIconSize(side: 'grid_icon_size_left' | 'grid_icon_size_right', typeId: string): string {
    const dict = this.settings[side];
    if (!dict) return 'M';
    return dict[typeId] ?? 'M';
  }

  onFocusModeChange(side: 'focus_mode_left' | 'focus_mode_right', typeId: string, value: string): void {
    const dict = (this.settings[side] as Record<string, string>) || {};
    dict[typeId] = value;
    (this.settings as Record<string, unknown>)[side] = { ...dict };
    this.save();
  }

  getFocusMode(side: 'focus_mode_left' | 'focus_mode_right', typeId: string): string {
    const dict = this.settings[side];
    if (!dict) return 'click';
    return dict[typeId] ?? 'click';
  }

  onNumberChange(key: string, value: number): void {
    (this.settings as Record<string, unknown>)[key] = value;
    this.save();
  }

  onStringChange(key: string, value: string): void {
    (this.settings as Record<string, unknown>)[key] = value;
    this.save();
  }

  /** Current per-mediaType import-defaults map, normalised to a plain
   *  object so the child component never has to defend against ``null``. */
  get importDefaults(): ImportDefaultsByMediaType {
    const raw = (this.settings as Record<string, unknown>)['import_defaults_by_media_type'];
    return (raw as ImportDefaultsByMediaType | undefined) || {};
  }

  onImportDefaultsChange(value: ImportDefaultsByMediaType): void {
    (this.settings as Record<string, unknown>)['import_defaults_by_media_type'] = value;
    this.save();
  }

  /** Effective solo-mediaType for the import-defaults tab — collapses
   *  the per-mediaType picker to a single tab when the user is in solo
   *  mode (so they only configure what they'll actually import). */
  get effectiveSoloMediaType(): string | null {
    const explicit = this.settings.solo_media_type_explicit;
    if (explicit) {
      return this.settings.solo_media_type || null;
    }
    return this.settings.effective_solo_media_type || null;
  }

  async resetDefaults(): Promise<void> {
    const ok = await this.dialog.confirmDestructive(
      'Reset all settings to factory defaults?',
      'Your current preferences (appearance, view modes, autopilot, calibration, and other per-user settings) will be overwritten and cannot be recovered.',
      'Reset',
    );
    if (!ok) return;
    this.settingsApi
      .getDefaults()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (defaults) => {
          this.settings = defaults;
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
          this.settings = s;
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
      .update(this.settings)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.error = '';
          this.flashSaved();
        },
        error: () => {
          this.error = 'Failed to save settings';
        },
      });
  }

  private flashSaved(): void {
    if (this.savedTimer !== null) clearTimeout(this.savedTimer);
    this.savedVisible = true;
    this.savedTimer = setTimeout(() => {
      this.savedVisible = false;
      this.savedTimer = null;
    }, 1800);
  }

  close(): void {
    this.closed.emit();
  }
}
