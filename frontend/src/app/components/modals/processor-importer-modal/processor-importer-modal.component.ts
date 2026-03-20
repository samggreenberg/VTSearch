import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { FileBrowserComponent } from '../../file-browser/file-browser.component';
import { ProcessorImportersApiService } from '../../../services/processor-importers-api.service';
import { ImporterField } from '../../../models/api.models';

interface ProcessorImporter {
  name: string;
  label?: string;
  display_name?: string;
  description?: string;
  icon?: string;
  fields?: ImporterField[];
}

type ModalView = 'picker' | 'form';

@Component({
  selector: 'vt-processor-importer-modal',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent, FileBrowserComponent],
  templateUrl: './processor-importer-modal.component.html',
  styleUrl: './processor-importer-modal.component.scss',
})
export class ProcessorImporterModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() imported = new EventEmitter<void>();

  view: ModalView = 'picker';
  importers: ProcessorImporter[] = [];
  selectedImporter: ProcessorImporter | null = null;
  formValues: Record<string, string> = {};
  selectedFile: File | null = null;
  submitting = false;
  error = '';
  successMessage = '';

  constructor(private processorImportersApi: ProcessorImportersApiService) {}

  ngOnInit(): void {
    this.processorImportersApi.list().subscribe({
      next: (list: any[]) => {
        this.importers = list;
      },
    });
  }

  selectImporter(importer: ProcessorImporter): void {
    this.selectedImporter = importer;
    this.formValues = {};
    this.error = '';
    this.successMessage = '';
    if (importer.fields) {
      for (const field of importer.fields) {
        if (field.default) this.formValues[field.key] = field.default;
      }
    }
    this.view = 'form';
  }

  back(): void {
    this.view = 'picker';
    this.selectedImporter = null;
    this.error = '';
    this.successMessage = '';
  }

  onFileSelected(event: Event, fieldName: string): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.formValues[fieldName] = input.files[0].name;
    }
  }

  submit(): void {
    if (!this.selectedImporter) return;
    this.submitting = true;
    this.error = '';
    this.successMessage = '';

    this.processorImportersApi.runImport(this.selectedImporter.name, this.formValues).subscribe({
      next: (res: any) => {
        this.submitting = false;
        this.successMessage = res.message || 'Import successful';
        this.imported.emit();
        setTimeout(() => this.close(), 1500);
      },
      error: (err) => {
        this.submitting = false;
        this.error = err.error?.error || 'Import failed';
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
