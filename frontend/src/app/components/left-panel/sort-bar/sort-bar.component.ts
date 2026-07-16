import { ChangeDetectionStrategy, Component, input, OnInit, output } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { SortMode } from '../left-panel.component';
import { LoadSortModalComponent } from '../../modals/load-sort-modal/load-sort-modal.component';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-sort-bar',
  standalone: true,
  imports: [FormsModule, LoadSortModalComponent],
  templateUrl: './sort-bar.component.html',
  styleUrl: './sort-bar.component.scss',
})
export class SortBarComponent implements OnInit {
  readonly sortMode = input<SortMode>('text');
  readonly loadSortLabel = input('');
  readonly initialTextQuery = input('');
  /**
   * True when the active detector (or active votes, when no detector is
   * loaded) has at least one good and one bad label available for training.
   * Drives the gating of the "Learned" sort mode.
   */
  readonly learnedSortAvailable = input(false);
  /**
   * True when the active dataset's embedder supports text queries. ``false``
   * for vision-only encoders (DINOv3, Perception Encoder); disables the
   * "Text" sort radio so users can't try a search that will always fail.
   */
  readonly textSortAvailable = input(true);

  readonly sortModeChange = output<SortMode>();
  readonly textSort = output<string>();
  readonly learnedSort = output<void>();
  readonly loadSort = output<void>();
  readonly modelSelected = output<string>();
  readonly exampleSortStarted = output<unknown>();

  textQuery = '';
  showLoadSortModal = false;

  ngOnInit(): void {
    const initialTextQuery = this.initialTextQuery();
    if (initialTextQuery) {
      this.textQuery = initialTextQuery;
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
    return !this.learnedSortAvailable();
  }

  get textDisabled(): boolean {
    return !this.textSortAvailable();
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
