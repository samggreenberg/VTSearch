import { Component, Input, OnDestroy, OnInit, inject, output, signal } from '@angular/core';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsCrudApiService } from '../../../services/detectors-crud-api.service';

interface Example {
  type: 'good' | 'bad';
  label?: string;
  data?: unknown;
  [key: string]: unknown;
}

@Component({
  selector: 'vt-examples-editor-modal',
  standalone: true,
  imports: [ModalComponent],
  templateUrl: './examples-editor-modal.component.html',
  styleUrl: './examples-editor-modal.component.scss',
})
export class ExamplesEditorModalComponent implements OnInit, OnDestroy {
  private detectorsCrudApi = inject(DetectorsCrudApiService);

  @Input() modelName = '';
  readonly closed = output<void>();
  readonly saved = output<void>();

  examples: Example[] = [];
  loading = true;
  saving = false;
  // Written from the load/save subscribes (async); template-bound.
  readonly error = signal('');
  readonly status = signal('');
  private closeTimer: ReturnType<typeof setTimeout> | null = null;

  ngOnInit(): void {
    if (!this.modelName) {
      this.loading = false;
      return;
    }
    this.detectorsCrudApi.get(this.modelName).subscribe({
      next: (data: any) => {
        this.examples = data.examples || [];
        this.loading = false;
      },
      error: () => {
        this.loading = false;
        this.error.set('Failed to load examples');
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
    this.error.set('');
    this.status.set('');
    this.detectorsCrudApi.setExamples(this.modelName, this.examples).subscribe({
      next: () => {
        this.saving = false;
        this.status.set('Saved.');
        this.saved.emit();
        this.closeTimer = setTimeout(() => this.close(), 600);
      },
      error: (err) => {
        this.saving = false;
        this.error.set(err.error?.error || 'Failed to save');
      },
    });
  }

  close(): void {
    if (this.closeTimer !== null) {
      clearTimeout(this.closeTimer);
      this.closeTimer = null;
    }
    this.closed.emit();
  }

  ngOnDestroy(): void {
    if (this.closeTimer !== null) {
      clearTimeout(this.closeTimer);
      this.closeTimer = null;
    }
  }
}
