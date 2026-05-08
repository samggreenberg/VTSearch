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
import { AppSettings, EmbedderInfo, MediaTypeInfo } from '../../../models/api.models';
import { Theme, ThemeService } from '../../../services/theme.service';

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
  embedders: EmbedderInfo[] = [];
  mediaTypes: MediaTypeInfo[] = [];
  activeSettingsTab = 'appearance';
  activeViewTab = '';
  loading = true;
  error = '';
  showImporterModal = false;
  showExporterModal = false;
  version = '';

  private destroy$ = new Subject<void>();

  constructor(
    private settingsApi: SettingsApiService,
    private settingsState: SettingsStateService,
    private datasetsApi: DatasetsApiService,
    private themeService: ThemeService,
  ) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  ngOnInit(): void {
    forkJoin({
      settings: this.settingsApi.getSettings(),
      embedders: this.settingsApi.getEmbedders(),
      mediaTypes: this.datasetsApi.getMediaTypes(),
      version: this.settingsApi.getVersion(),
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
      next: (res) => {
        this.settings = res.settings;
        this.version = res.version.version;
        this.embedders = (res.embedders.embedders || []).sort(
          (a, b) => a.media_type_id.localeCompare(b.media_type_id) || a.name.localeCompare(b.name),
        );
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
    this.settings.theme = theme;
    this.themeService.setTheme(theme as Theme);
    this.save();
  }

  onToggle(key: string, value: boolean): void {
    (this.settings as Record<string, unknown>)[key] = value;
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

  getMediaTypeIcon(typeId: string): string {
    const mt = this.mediaTypes.find((m) => m.type_id === typeId);
    return mt?.icon || '';
  }

  isEmbedderAutoloaded(embedder: EmbedderInfo): boolean {
    return (this.settings.autoload_media_embedders || []).includes(embedder.name);
  }

  toggleEmbedder(embedder: EmbedderInfo): void {
    const current = this.settings.autoload_media_embedders || [];
    if (current.includes(embedder.name)) {
      this.settings.autoload_media_embedders = current.filter((e) => e !== embedder.name);
    } else {
      this.settings.autoload_media_embedders = [...current, embedder.name];
    }
    this.save();
  }

  resetDefaults(): void {
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
        error: () => {
          this.error = 'Failed to save settings';
        },
      });
  }

  close(): void {
    this.closed.emit();
  }
}
