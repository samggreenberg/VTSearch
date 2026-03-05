import { Component, EventEmitter, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsApiService } from '../../../services/detectors-api.service';
import { SortingApiService } from '../../../services/sorting-api.service';

@Component({
  selector: 'vt-load-sort-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './load-sort-modal.component.html',
  styleUrl: './load-sort-modal.component.scss',
})
export class LoadSortModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();
  @Output() detectorLoaded = new EventEmitter<unknown>();
  @Output() exampleSortStarted = new EventEmitter<unknown>();

  serverDetectors: string[] = [];
  serverMediaFiles: string[] = [];
  loading = true;
  status = '';
  error = '';

  constructor(
    private detectorsApi: DetectorsApiService,
    private sortingApi: SortingApiService,
  ) {}

  ngOnInit(): void {
    this.detectorsApi.getServerFiles().subscribe({
      next: (res) => {
        this.serverDetectors = res.files || [];
        this.loading = false;
      },
      error: () => {
        this.loading = false;
      },
    });
    this.sortingApi.getServerMediaFiles().subscribe({
      next: (res) => {
        this.serverMediaFiles = res.files || [];
      },
    });
  }

  onDetectorFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string);
        this.detectorLoaded.emit(data);
        this.closed.emit();
      } catch {
        this.error = 'Invalid detector file';
      }
    };
    reader.readAsText(file);
  }

  loadServerDetector(name: string): void {
    this.status = 'Loading server detector...';
    this.detectorsApi.getServerFile(name).subscribe({
      next: (data) => {
        this.status = '';
        this.detectorLoaded.emit(data);
        this.closed.emit();
      },
      error: () => {
        this.status = '';
        this.error = 'Failed to load detector';
      },
    });
  }

  onMediaFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    const file = input.files[0];
    this.status = 'Scoring with example media...';
    this.sortingApi.exampleSort(file).subscribe({
      next: (data) => {
        this.status = '';
        this.exampleSortStarted.emit(data);
        this.closed.emit();
      },
      error: () => {
        this.status = '';
        this.error = 'Example sort failed';
      },
    });
  }

  loadServerMedia(filename: string): void {
    this.status = 'Scoring with example media...';
    this.sortingApi.exampleSortServer({ filename }).subscribe({
      next: (data) => {
        this.status = '';
        this.exampleSortStarted.emit(data);
        this.closed.emit();
      },
      error: () => {
        this.status = '';
        this.error = 'Example sort failed';
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
