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

  /**
   * Run the text sort, and drop focus from whatever triggered it.
   *
   * The blur is the point of the ``trigger`` argument, not a nicety:
   * ``KeyboardService`` deliberately ignores every shortcut while focus sits
   * in a text field, so an input that keeps focus after Enter leaves the user
   * unable to vote with the arrow keys on the results they just asked for.
   * Submitting ends the typing task, so hand focus back to the document.
   *
   * Only a submit that actually runs blurs: Enter on an empty (or
   * whitespace-only) query is a no-op, and yanking focus out of the field the
   * user is still filling in would be hostile.
   */
  submitTextSort(trigger?: HTMLElement | null): void {
    const trimmed = this.textQuery.trim();
    if (trimmed) {
      trigger?.blur();
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
