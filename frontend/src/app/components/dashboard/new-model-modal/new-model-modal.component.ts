import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { TrainableModelsApiService } from '../../../services/trainable-models-api.service';
import { DatasetsApiService } from '../../../services/datasets-api.service';

@Component({
  selector: 'vt-new-model-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './new-model-modal.component.html',
  styleUrl: './new-model-modal.component.scss',
})
export class NewModelModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() created = new EventEmitter<void>();

  name = '';
  mediaType = 'audio';
  textQuery = '';
  mediaTypes: string[] = [];
  submitting = false;
  error = '';

  constructor(
    private modelsApi: TrainableModelsApiService,
    private datasetsApi: DatasetsApiService,
  ) {}

  ngOnInit(): void {
    this.datasetsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypes = (res.media_types || []).map((t) => t.type_id || t.name);
      },
    });
    this.datasetsApi.getRegistry().subscribe({
      next: (res) => {
        const types = new Set(
          (res.datasets || []).map((d) => d['media_type'] as string).filter(Boolean),
        );
        if (types.size === 1) {
          this.mediaType = [...types][0];
        }
      },
    });
  }

  submit(): void {
    const trimmedName = this.name.trim();
    const trimmedQuery = this.textQuery.trim();
    if (!trimmedName) {
      this.error = 'Name is required';
      return;
    }
    if (!trimmedQuery) {
      this.error = 'At least one text query or example is required';
      return;
    }

    this.submitting = true;
    this.error = '';

    this.modelsApi
      .registerModel({
        name: trimmedName,
        media_type: this.mediaType,
        trainable: true,
        text_query: trimmedQuery,
      })
      .subscribe({
        next: () => {
          this.submitting = false;
          this.created.emit();
        },
        error: (err) => {
          this.submitting = false;
          this.error = err.error?.error || 'Failed to create model';
        },
      });
  }

  close(): void {
    this.closed.emit();
  }
}
