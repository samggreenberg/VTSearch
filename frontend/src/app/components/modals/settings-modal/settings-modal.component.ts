import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import { ModalComponent } from '../../modal/modal.component';
import { SettingsApiService } from '../../../services/settings-api.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { AppSettings, EmbedderInfo, MediaTypeInfo } from '../../../models/api.models';
import { Theme, ThemeService } from '../../../services/theme.service';

@Component({
  selector: 'vt-settings-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './settings-modal.component.html',
  styleUrl: './settings-modal.component.scss',
})
export class SettingsModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();

  settings: AppSettings = { volume: 50 };
  embedders: EmbedderInfo[] = [];
  mediaTypes: MediaTypeInfo[] = [];
  activeViewTab = '';
  loading = true;
  error = '';

  constructor(
    private settingsApi: SettingsApiService,
    private settingsState: SettingsStateService,
    private datasetsApi: DatasetsApiService,
    private themeService: ThemeService,
  ) {}

  ngOnInit(): void {
    forkJoin({
      settings: this.settingsApi.getSettings(),
      embedders: this.settingsApi.getEmbedders(),
      mediaTypes: this.datasetsApi.getMediaTypes(),
    }).subscribe({
      next: (res) => {
        this.settings = res.settings;
        this.embedders = (res.embedders.embedders || []).sort(
          (a, b) => a.media_type_id.localeCompare(b.media_type_id) || a.name.localeCompare(b.name),
        );
        this.mediaTypes = res.mediaTypes.media_types || [];
        if (this.mediaTypes.length > 0) {
          this.activeViewTab = this.mediaTypes[0].type_id;
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

  onGridItemSizeChange(side: 'grid_item_size_left' | 'grid_item_size_right', typeId: string, value: string): void {
    const dict = (this.settings[side] as Record<string, string>) || {};
    dict[typeId] = value;
    (this.settings as Record<string, unknown>)[side] = { ...dict };
    this.save();
  }

  getGridItemSize(side: 'grid_item_size_left' | 'grid_item_size_right', typeId: string): string {
    const dict = this.settings[side];
    if (!dict) return 'medium';
    return dict[typeId] ?? 'medium';
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
    this.settingsApi.getDefaults().subscribe({
      next: (defaults) => {
        this.settings = defaults;
        if (defaults.theme) {
          this.themeService.setTheme(defaults.theme as Theme);
        }
        this.save();
      },
    });
  }


  exportSettings(): void {
    const blob = new Blob([JSON.stringify(this.settings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'settings.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  importSettings(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const imported = JSON.parse(reader.result as string);
        this.settings = imported;
        if (imported.theme) {
          this.themeService.setTheme(imported.theme);
        }
        this.save();
      } catch {
        this.error = 'Invalid settings file';
      }
    };
    reader.readAsText(file);
  }

  private save(): void {
    this.settingsState.update(this.settings).subscribe({
      error: () => {
        this.error = 'Failed to save settings';
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
