import { ChangeDetectionStrategy, Component, inject, input, OnInit, output, signal } from '@angular/core';
import { TitleCasePipe } from '@angular/common';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsCrudApiService } from '../../../services/detectors-crud-api.service';
import type { DetectorCombineResponse } from '../../../generated/api-client/models/detector-combine-response';
import { DetectorRegistryEntry } from '../../../generated/api-client/models/detector-registry-entry';
import { apiErrorMessage } from '../../../utils/api-error';

interface SourceRow {
  name: string;
  numLabels: number;
  textQuery: string;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-combine-detectors-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, TitleCasePipe],
  templateUrl: './combine-detectors-modal.component.html',
  styleUrl: './combine-detectors-modal.component.scss',
})
export class CombineDetectorsModalComponent implements OnInit {
  private detectorsCrudApi = inject(DetectorsCrudApiService);

  /** Trainable models the user has selected on the dashboard. */
  readonly sources = input<DetectorRegistryEntry[]>([]);
  /** All registered model names; used for inline name-collision check. */
  readonly existingNames = input<string[]>([]);

  readonly closed = output<void>();
  readonly created = output<string>();

  newName = '';
  conflictPolicy: 'drop' = 'drop';
  // Signals, mirroring the sibling combine-datasets-modal: written from the
  // async combine subscribe, which under zoneless + OnPush schedules no
  // repaint of its own. Plain fields left the modal stuck on "Combining…"
  // with both buttons disabled and the server error hidden.
  readonly submitting = signal(false);
  readonly error = signal('');

  rows: SourceRow[] = [];
  totalLabels = 0;
  mediaType = '';

  ngOnInit(): void {
    this.rows = this.sources().map((m) => ({
      name: this.trainableNameOf(m),
      numLabels: (m.num_training as number) ?? 0,
      textQuery: (m.text_query as string) ?? '',
    }));
    this.totalLabels = this.rows.reduce((sum, r) => sum + r.numLabels, 0);
    this.mediaType = this.sources()[0]?.media_type ?? '';
  }

  /** The combine API operates on the registry name (= labelset filename). */
  private trainableNameOf(m: DetectorRegistryEntry): string {
    return (m.name || '').trim();
  }

  get nameCollision(): boolean {
    const trimmed = this.newName.trim();
    if (!trimmed) return false;
    return this.existingNames().some((n) => n === trimmed);
  }

  get nameValidationMessage(): string {
    const trimmed = this.newName.trim();
    if (!trimmed) return '';
    if (this.nameCollision) return `A model named "${trimmed}" already exists.`;
    return '';
  }

  get canSubmit(): boolean {
    return (
      !this.submitting() &&
      this.rows.length >= 2 &&
      !!this.newName.trim() &&
      !this.nameCollision
    );
  }

  submit(): void {
    if (!this.canSubmit) return;
    const trimmed = this.newName.trim();
    const names = this.rows.map((r) => r.name);
    this.submitting.set(true);
    this.error.set('');

    this.detectorsCrudApi.combine(names, trimmed, this.conflictPolicy).subscribe({
      next: (resp: DetectorCombineResponse) => {
        this.submitting.set(false);
        this.created.emit(resp?.name || trimmed);
      },
      error: (err) => {
        this.submitting.set(false);
        const status = err?.status;
        const serverMsg = apiErrorMessage(err, '');
        if (status === 422) {
          this.error.set(
            serverMsg ||
              'Every label was a conflict; no detector was created. Try fewer or more aligned sources.',
          );
        } else if (status === 409) {
          this.error.set(serverMsg || `A detector named "${trimmed}" already exists.`);
        } else if (status === 404) {
          this.error.set(serverMsg || 'A source detector was not found.');
        } else if (status === 400) {
          this.error.set(serverMsg || 'Invalid combine request.');
        } else {
          this.error.set(serverMsg || 'Failed to combine detectors.');
        }
      },
    });
  }

  close(): void {
    if (this.submitting()) return;
    this.closed.emit();
  }
}
