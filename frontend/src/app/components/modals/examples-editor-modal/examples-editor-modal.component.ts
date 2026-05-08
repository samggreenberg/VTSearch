import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsApiService } from '../../../services/detectors-api.service';

interface Example {
  type: 'good' | 'bad';
  label?: string;
  data?: unknown;
  [key: string]: unknown;
}

@Component({
  selector: 'vt-examples-editor-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './examples-editor-modal.component.html',
  styleUrl: './examples-editor-modal.component.scss',
})
export class ExamplesEditorModalComponent implements OnInit {
  @Input() modelName = '';
  @Output() closed = new EventEmitter<void>();
  @Output() saved = new EventEmitter<void>();

  examples: Example[] = [];
  loading = true;
  saving = false;
  error = '';
  status = '';

  constructor(private detectorsApi: DetectorsApiService) {}

  ngOnInit(): void {
    if (!this.modelName) {
      this.loading = false;
      return;
    }
    this.detectorsApi.get(this.modelName).subscribe({
      next: (data: any) => {
        this.examples = data.examples || [];
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error = 'Failed to load examples';
      },
    });
  }

  get goodExamples(): Example[] {
    return this.examples.filter((e) => e.type === 'good');
  }

  get badExamples(): Example[] {
    return this.examples.filter((e) => e.type === 'bad');
  }

  removeExample(index: number): void {
    this.examples.splice(index, 1);
  }

  onFileSelected(event: Event, type: 'good' | 'bad'): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    this.examples.push({
      type,
      label: file.name,
    });
    input.value = '';
  }

  save(): void {
    this.saving = true;
    this.error = '';
    this.status = '';
    this.detectorsApi.setExamples(this.modelName, this.examples).subscribe({
      next: () => {
        this.saving = false;
        this.status = 'Saved.';
        this.saved.emit();
        setTimeout(() => this.close(), 600);
      },
      error: (err) => {
        this.saving = false;
        this.error = err.error?.error || 'Failed to save';
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
