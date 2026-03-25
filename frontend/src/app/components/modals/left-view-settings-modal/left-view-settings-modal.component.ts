import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import { SettingsApiService } from '../../../services/settings-api.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { AppSettings, MediaTypeInfo } from '../../../models/api.models';

@Component({
  selector: 'vt-left-view-settings-modal',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './left-view-settings-modal.component.html',
  styleUrl: './left-view-settings-modal.component.scss',
})
export class LeftViewSettingsModalComponent implements OnInit {
  @Input() currentMediaType = '';
  @Output() closed = new EventEmitter<void>();

  settings: AppSettings = { volume: 50 };
  mediaTypes: MediaTypeInfo[] = [];
  activeViewTab = '';
  loading = true;
  error = '';

  constructor(
    private settingsApi: SettingsApiService,
    private settingsState: SettingsStateService,
    private datasetsApi: DatasetsApiService,
  ) {}

  ngOnInit(): void {
    forkJoin({
      settings: this.settingsApi.getSettings(),
      mediaTypes: this.datasetsApi.getMediaTypes(),
    }).subscribe({
      next: (res) => {
        this.settings = res.settings;
        this.mediaTypes = res.mediaTypes.media_types || [];
        if (this.currentMediaType) {
          this.mediaTypes = this.mediaTypes.filter(mt => mt.type_id === this.currentMediaType);
        }
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

  onViewModeChange(typeId: string, value: string): void {
    const dict = (this.settings.view_mode_left as Record<string, string>) || {};
    dict[typeId] = value;
    const updated = { ...dict };
    (this.settings as Record<string, unknown>)['view_mode_left'] = updated;
    this.saveField({ view_mode_left: updated as Record<string, 'grid' | 'list'> });
  }

  getViewMode(typeId: string): string {
    const dict = this.settings.view_mode_left;
    if (!dict) return 'list';
    return dict[typeId] ?? 'list';
  }

  onGridIconSizeChange(typeId: string, value: string): void {
    const dict = (this.settings.grid_icon_size_left as Record<string, string>) || {};
    dict[typeId] = value;
    const updated = { ...dict };
    (this.settings as Record<string, unknown>)['grid_icon_size_left'] = updated;
    this.saveField({ grid_icon_size_left: updated });
  }

  getGridIconSize(typeId: string): string {
    const dict = this.settings.grid_icon_size_left;
    if (!dict) return 'M';
    return dict[typeId] ?? 'M';
  }

  onFocusModeChange(typeId: string, value: string): void {
    const dict = (this.settings.focus_mode_left as Record<string, string>) || {};
    dict[typeId] = value;
    const updated = { ...dict };
    (this.settings as Record<string, unknown>)['focus_mode_left'] = updated;
    this.saveField({ focus_mode_left: updated as Record<string, 'click' | 'hover'> });
  }

  getFocusMode(typeId: string): string {
    const dict = this.settings.focus_mode_left;
    if (!dict) return 'click';
    return dict[typeId] ?? 'click';
  }

  onPanelPctChange(typeId: string, value: number): void {
    const clamped = Math.max(10, Math.min(90, Math.round(value)));
    const dict = (this.settings.panel_pct_left as Record<string, number>) || {};
    dict[typeId] = clamped;
    const updated = { ...dict };
    (this.settings as Record<string, unknown>)['panel_pct_left'] = updated;
    this.saveField({ panel_pct_left: updated });
  }

  getPanelPct(typeId: string): number {
    const dict = this.settings.panel_pct_left;
    if (!dict) return 50;
    return dict[typeId] ?? 50;
  }

  private saveField(changes: Partial<AppSettings>): void {
    this.settingsState.update(changes).subscribe({
      next: (updated) => {
        this.settings = updated;
      },
      error: () => {
        this.error = 'Failed to save settings';
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
