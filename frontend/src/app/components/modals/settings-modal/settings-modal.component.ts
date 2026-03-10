import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import { ModalComponent } from '../../modal/modal.component';
import { SettingsApiService } from '../../../services/settings-api.service';
import { AppSettings, EmbedderInfo } from '../../../models/api.models';
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
  loading = true;
  error = '';

  constructor(
    private settingsApi: SettingsApiService,
    private themeService: ThemeService,
  ) {}

  ngOnInit(): void {
    forkJoin({
      settings: this.settingsApi.getSettings(),
      embedders: this.settingsApi.getEmbedders(),
    }).subscribe({
      next: (res) => {
        this.settings = res.settings;
        this.embedders = (res.embedders.embedders || []).sort(
          (a, b) => a.media_type_id.localeCompare(b.media_type_id) || a.name.localeCompare(b.name),
        );
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
    this.settingsApi.updateSettings(this.settings).subscribe({
      error: () => {
        this.error = 'Failed to save settings';
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
