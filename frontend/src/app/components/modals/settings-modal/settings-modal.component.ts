import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { SettingsImporterModalComponent } from '../settings-importer-modal/settings-importer-modal.component';
import { SettingsExporterModalComponent } from '../settings-exporter-modal/settings-exporter-modal.component';
import { SettingsApiService } from '../../../services/settings-api.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import type { AppSettings } from '../../../generated/api-client/models/app-settings';
import { MediaTypeInfo } from '../../../models/api.models';
import { Theme, ThemeService } from '../../../services/theme.service';
import { formatVersion } from '../../../utils/format-date';
import { VtDialogService } from '../../../services/dialog.service';

@Component({
  selector: 'vt-settings-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, IconComponent, SettingsImporterModalComponent, SettingsExporterModalComponent],
  templateUrl: './settings-modal.component.html',
  styleUrl: './settings-modal.component.scss',
})
export class SettingsModalComponent implements OnInit, OnDestroy {
  @Input() preselectedViewTab = '';
  @Output() closed = new EventEmitter<void>();

  settings: AppSettings = { volume: 50 };
  mediaTypes: MediaTypeInfo[] = [];
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
      version: this.settingsApi.getVersion(),
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
      next: (res) => {
        this.settings = res.settings;
        this.version = formatVersion(res.version.version);
        this.mediaTypes = res.mediaTypes.media_types || [];
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
