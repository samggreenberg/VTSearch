import { Component, Input, Output, EventEmitter, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { debounceTime, takeUntil } from 'rxjs/operators';
import { SortMode } from '../left-panel.component';

@Component({
  selector: 'vt-sort-bar',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './sort-bar.component.html',
  styleUrl: './sort-bar.component.scss',
})
export class SortBarComponent implements OnDestroy {
  @Input() sortMode: SortMode = 'text';
  @Input() loadSortLabel = '';
  @Input() hasGoodVotes = false;
  @Input() hasBadVotes = false;

  @Output() sortModeChange = new EventEmitter<SortMode>();
  @Output() textSort = new EventEmitter<string>();
  @Output() learnedSort = new EventEmitter<void>();
  @Output() loadSort = new EventEmitter<void>();

  textQuery = '';

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
}
