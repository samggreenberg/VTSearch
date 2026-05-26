import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { SortMode } from '../left-panel.component';
import { LoadSortModalComponent } from '../../modals/load-sort-modal/load-sort-modal.component';

@Component({
  selector: 'vt-sort-bar',
  standalone: true,
  imports: [CommonModule, FormsModule, LoadSortModalComponent],
  templateUrl: './sort-bar.component.html',
  styleUrl: './sort-bar.component.scss',
})
export class SortBarComponent implements OnInit {
  @Input() sortMode: SortMode = 'text';
  @Input() loadSortLabel = '';
  @Input() initialTextQuery = '';
  /**
   * True when the active detector (or active votes, when no detector is
   * loaded) has at least one good and one bad label available for training.
   * Drives the gating of the "Learned" sort mode.
   */
  @Input() learnedSortAvailable = false;
  /**
   * True when the active dataset's embedder supports text queries. ``false``
   * for vision-only encoders (DINOv3, Perception Encoder) - disables the
   * "Text" sort radio so users can't try a search that will always fail.
   */
  @Input() textSortAvailable = true;

  @Output() sortModeChange = new EventEmitter<SortMode>();
  @Output() textSort = new EventEmitter<string>();
  @Output() learnedSort = new EventEmitter<void>();
  @Output() loadSort = new EventEmitter<void>();
  @Output() modelSelected = new EventEmitter<string>();
  @Output() exampleSortStarted = new EventEmitter<unknown>();

  textQuery = '';
  showLoadSortModal = false;

  ngOnInit(): void {
    if (this.initialTextQuery) {
      this.textQuery = this.initialTextQuery;
    }
  }

  onSortModeChange(mode: SortMode): void {
    this.sortModeChange.emit(mode);
    if (mode === 'learned') {
      this.learnedSort.emit();
    } else if (mode === 'load') {
      this.loadSort.emit();
    }
  }

  onTextInput(value: string): void {
    this.textQuery = value;
  }

  submitTextSort(): void {
    const trimmed = this.textQuery.trim();
    if (trimmed) {
      this.textSort.emit(trimmed);
    }
  }

  get learnedDisabled(): boolean {
    return !this.learnedSortAvailable;
  }

  get textDisabled(): boolean {
    return !this.textSortAvailable;
  }

  get searchDisabled(): boolean {
    return !this.textQuery.trim();
  }

  onAddLoadSort(): void {
    this.showLoadSortModal = true;
  }

  onModelSelected(modelId: string): void {
    this.showLoadSortModal = false;
    this.modelSelected.emit(modelId);
  }

  onExampleSortStarted(data: unknown): void {
    this.showLoadSortModal = false;
    this.exampleSortStarted.emit(data);
  }
}
