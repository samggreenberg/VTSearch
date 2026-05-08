import { Component, Input, Output, EventEmitter, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { debounceTime, takeUntil } from 'rxjs/operators';
import { SortMode } from '../left-panel.component';
import { LoadSortModalComponent } from '../../modals/load-sort-modal/load-sort-modal.component';

@Component({
  selector: 'vt-sort-bar',
  standalone: true,
  imports: [CommonModule, FormsModule, LoadSortModalComponent],
  templateUrl: './sort-bar.component.html',
  styleUrl: './sort-bar.component.scss',
})
export class SortBarComponent implements OnInit, OnDestroy {
  @Input() sortMode: SortMode = 'text';
  @Input() loadSortLabel = '';
  @Input() initialTextQuery = '';
  @Input() hasGoodVotes = false;
  @Input() hasBadVotes = false;

  @Output() sortModeChange = new EventEmitter<SortMode>();
  @Output() textSort = new EventEmitter<string>();
  @Output() learnedSort = new EventEmitter<void>();
  @Output() loadSort = new EventEmitter<void>();
  @Output() modelSelected = new EventEmitter<string>();
  @Output() exampleSortStarted = new EventEmitter<unknown>();

  textQuery = '';
  showLoadSortModal = false;

  private textInput$ = new Subject<string>();
  private destroy$ = new Subject<void>();

  constructor() {
    this.textInput$
      .pipe(debounceTime(400), takeUntil(this.destroy$))
      .subscribe((text) => {
        if (text.trim()) {
          this.textSort.emit(text.trim());
        }
      });
  }

  ngOnInit(): void {
    if (this.initialTextQuery) {
      this.textQuery = this.initialTextQuery;
    }
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
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
    this.textInput$.next(value);
  }

  get learnedDisabled(): boolean {
    return !this.hasGoodVotes || !this.hasBadVotes;
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
