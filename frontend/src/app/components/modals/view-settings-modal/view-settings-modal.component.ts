import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import { ModalComponent } from '../../modal/modal.component';
import { SettingsApiService } from '../../../services/settings-api.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';
import { AppSettings, MediaTypeInfo } from '../../../models/api.models';

@Component({
  selector: 'vt-view-settings-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './view-settings-modal.component.html',
  styleUrl: './view-settings-modal.component.scss',
})
export class ViewSettingsModalComponent implements OnInit {
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
    const dict = (this.settings.view_mode_right as Record<string, string>) || {};
    dict[typeId] = value;
    (this.settings as Record<string, unknown>)['view_mode_right'] = { ...dict };
    this.save();
  }

  getViewMode(typeId: string): string {
    const dict = this.settings.view_mode_right;
    if (!dict) return 'grid';
    return dict[typeId] ?? 'grid';
  }

  onGridColumnsChange(typeId: string, value: number): void {
    const clamped = Math.max(1, Math.min(6, Math.round(value)));
    const dict = (this.settings.grid_columns_right as Record<string, number>) || {};
    dict[typeId] = clamped;
    (this.settings as Record<string, unknown>)['grid_columns_right'] = { ...dict };
    this.save();
  }

  getGridColumns(typeId: string): number {
    const dict = this.settings.grid_columns_right;
    if (!dict) return 2;
    return dict[typeId] ?? 2;
  }

  onFocusModeChange(typeId: string, value: string): void {
    const dict = (this.settings.focus_mode_right as Record<string, string>) || {};
    dict[typeId] = value;
    (this.settings as Record<string, unknown>)['focus_mode_right'] = { ...dict };
    this.save();
  }

  getFocusMode(typeId: string): string {
    const dict = this.settings.focus_mode_right;
    if (!dict) return 'click';
    return dict[typeId] ?? 'click';
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
