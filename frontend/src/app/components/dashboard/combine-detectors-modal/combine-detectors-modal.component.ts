import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorSwatchComponent } from '../../detector-swatch/detector-swatch.component';
import { DetectorsCrudApiService } from '../../../services/detectors-crud-api.service';
import type { DetectorCombineResponse } from '../../../generated/api-client/models/detector-combine-response';
import { DetectorRegistryEntry } from '../../../models/api.models';

interface SourceRow {
  name: string;
  numLabels: number;
  textQuery: string;
}

@Component({
  selector: 'vt-combine-detectors-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, DetectorSwatchComponent],
  templateUrl: './combine-detectors-modal.component.html',
  styleUrl: './combine-detectors-modal.component.scss',
})
export class CombineDetectorsModalComponent implements OnInit {
  /** Trainable models the user has selected on the dashboard. */
  @Input() sources: DetectorRegistryEntry[] = [];
  /** All registered model names; used for inline name-collision check. */
  @Input() existingNames: string[] = [];

  @Output() closed = new EventEmitter<void>();
  @Output() created = new EventEmitter<string>();

  newName = '';
  conflictPolicy: 'drop' = 'drop';
  submitting = false;
  error = '';

  rows: SourceRow[] = [];
  totalLabels = 0;
  mediaType = '';

  constructor(private detectorsCrudApi: DetectorsCrudApiService) {}

  ngOnInit(): void {
    this.rows = this.sources.map((m) => ({
      name: this.trainableNameOf(m),
      numLabels: (m.num_training as number) ?? 0,
      textQuery: (m.text_query as string) ?? '',
    }));
    this.totalLabels = this.rows.reduce((sum, r) => sum + r.numLabels, 0);
    this.mediaType = this.sources[0]?.media_type ?? '';
  }

  /** The combine API operates on the registry name (= labelset filename). */
  private trainableNameOf(m: DetectorRegistryEntry): string {
    return (m.name || '').trim();
  }

  get nameCollision(): boolean {
    const trimmed = this.newName.trim();
    if (!trimmed) return false;
    return this.existingNames.some((n) => n === trimmed);
  }

  get nameValidationMessage(): string {
    const trimmed = this.newName.trim();
    if (!trimmed) return '';
    if (this.nameCollision) return `A model named "${trimmed}" already exists.`;
    return '';
  }

  get canSubmit(): boolean {
    return (
      !this.submitting &&
      this.rows.length >= 2 &&
      !!this.newName.trim() &&
      !this.nameCollision
    );
  }

  submit(): void {
    if (!this.canSubmit) return;
    const trimmed = this.newName.trim();
    const names = this.rows.map((r) => r.name);
    this.submitting = true;
    this.error = '';

    this.detectorsCrudApi.combine(names, trimmed, this.conflictPolicy).subscribe({
      next: (resp: DetectorCombineResponse) => {
        this.submitting = false;
        this.created.emit(resp?.name || trimmed);
      },
      error: (err) => {
        this.submitting = false;
        const status = err?.status;
        const serverMsg = err?.error?.error || '';
        if (status === 422) {
          this.error =
            serverMsg ||
            'Every label was a conflict; no detector was created. Try fewer or more aligned sources.';
        } else if (status === 409) {
          this.error = serverMsg || `A detector named "${trimmed}" already exists.`;
        } else if (status === 404) {
          this.error = serverMsg || 'A source detector was not found.';
        } else if (status === 400) {
          this.error = serverMsg || 'Invalid combine request.';
        } else {
          this.error = serverMsg || 'Failed to combine detectors.';
        }
      },
    });
  }

  close(): void {
    if (this.submitting) return;
    this.closed.emit();
  }
}
