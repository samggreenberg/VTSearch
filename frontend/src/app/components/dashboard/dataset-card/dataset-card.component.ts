import { ChangeDetectionStrategy, Component, ElementRef, HostBinding, HostListener, Input, input, OnChanges, output, SimpleChanges, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoadingTask } from '../../../models/api.models';
import { JobProgressComponent } from '../../job-progress/job-progress.component';
import {
  ProgressBarState,
  ProgressHeader,
  ProgressKind,
  formatProgressHeader,
  progressBarState,
} from '../../../utils/format-progress';
import { formatTimestamp } from '../../../utils/format-date';
import { ContextMenuComponent, ContextMenuItem } from '../../context-menu/context-menu.component';
import { buildDatasetCardMenuItems } from '../card-context-menu-items';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-dataset-card',
  standalone: true,
  imports: [CommonModule, FormsModule, JobProgressComponent, ContextMenuComponent],
  templateUrl: './dataset-card.component.html',
  styleUrl: './dataset-card.component.scss',
})
export class DatasetCardComponent implements OnChanges {
  @Input() dataset: any;
  readonly currentUser = input('');
  readonly isDefaultLogin = input(true);
  @Input() columnOrder: string[] = [];
  @Input() @HostBinding('class.selected') selected = false;
  @Input() @HostBinding('class.dimmed') dimmed = false;
  @Input() loadingTask?: LoadingTask;
  /** Which progress vocabulary to render the inline row with. ``projection``
   *  is used while the Browse button pre-builds the dataset's projection. */
  readonly taskKind = input<ProgressKind>('dataset');

  @HostBinding('class.loading-error')
  get hasLoadingError(): boolean {
    return !!this.loadingTask?.error;
  }
  readonly rowClick = output<MouseEvent>();

  @HostListener('click', ['$event'])
  onClick(event: MouseEvent): void {
    this.rowClick.emit(event);
  }

  @HostListener('contextmenu', ['$event'])
  onContextMenu(event: MouseEvent): void {
    // While renaming, let the native menu serve the text input (paste, etc.).
    if (this.editing) return;
    // A loading/errored row shows a progress bar in place of the actions, so
    // there is nothing to act on.
    if (this.loadingTask) return;
    event.preventDefault();
    this.contextMenuItems = buildDatasetCardMenuItems(this.dataset, {
      isDefaultLogin: this.isDefaultLogin(),
      isOwner: this.isOwner,
    });
    this.contextMenuX = event.clientX;
    this.contextMenuY = event.clientY;
    this.contextMenuOpen = true;
  }
  readonly rename = output<string>();
  readonly stats = output<void>();
  readonly delete = output<void>();
  readonly load = output<void>();
  readonly browse = output<void>();
  readonly security = output<void>();
  readonly cancelTask = output<string>();
  readonly dismissTask = output<string>();
  readonly checkboxToggle = output<void>();

  get isOwner(): boolean {
    return this.dataset?.created_by === this.currentUser();
  }

  @ViewChild('renameInput') renameInput?: ElementRef<HTMLInputElement>;

  readonly statsOpen = input(false);
  readonly deleteConfirmOpen = input(false);

  editing = false;
  wasEditing = false;
  editName = '';
  wasStatsOpen = false;
  wasDeleteOpen = false;

  contextMenuOpen = false;
  contextMenuX = 0;
  contextMenuY = 0;
  contextMenuItems: ContextMenuItem[] = [];

  startRename(event: MouseEvent): void {
    event.stopPropagation();
    this.beginRename();
  }

  /** Enter inline-rename mode (shared by the pencil button and the menu). */
  beginRename(): void {
    this.editing = true;
    this.wasEditing = true;
    this.editName = this.dataset.name;
    setTimeout(() => this.renameInput?.nativeElement.focus());
  }

  onContextMenuAction(id: string): void {
    this.contextMenuOpen = false;
    switch (id) {
      case 'load':
        this.load.emit();
        break;
      case 'browse':
        this.browse.emit();
        break;
      case 'security':
        this.security.emit();
        break;
      case 'rename':
        this.beginRename();
        break;
      case 'stats':
        this.stats.emit();
        break;
      case 'delete':
        this.delete.emit();
        break;
    }
  }

  dismissContextMenu(): void {
    this.contextMenuOpen = false;
  }

  confirmRename(): void {
    const trimmed = this.editName.trim();
    if (trimmed && trimmed !== this.dataset.name) {
      this.rename.emit(trimmed);
    }
    this.editing = false;
  }

  cancelRename(): void {
    this.editing = false;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['statsOpen'] && !changes['statsOpen'].currentValue && changes['statsOpen'].previousValue) {
      this.wasStatsOpen = true;
    }
    if (changes['deleteConfirmOpen'] && !changes['deleteConfirmOpen'].currentValue && changes['deleteConfirmOpen'].previousValue) {
      this.wasDeleteOpen = true;
    }
  }

  onPencilAnimationEnd(): void {
    if (!this.editing) {
      this.wasEditing = false;
    }
  }

  onPieAnimationEnd(): void {
    if (!this.statsOpen()) {
      this.wasStatsOpen = false;
    }
  }

  onTrashAnimationEnd(): void {
    if (!this.deleteConfirmOpen()) {
      this.wasDeleteOpen = false;
    }
  }

  onRenameKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      this.confirmRename();
    } else if (event.key === 'Escape') {
      this.cancelRename();
    }
  }

  onStats(event: MouseEvent): void {
    event.stopPropagation();
    this.stats.emit();
  }

  onSecurity(event: MouseEvent): void {
    event.stopPropagation();
    this.security.emit();
  }

  onLoad(event: MouseEvent): void {
    event.stopPropagation();
    this.load.emit();
  }

  onBrowse(event: MouseEvent): void {
    event.stopPropagation();
    this.browse.emit();
  }

  onDelete(event: MouseEvent): void {
    event.stopPropagation();
    this.delete.emit();
  }

  onCheckboxClick(event: MouseEvent): void {
    event.stopPropagation();
    this.checkboxToggle.emit();
  }

  formatDate(timestamp: number | null): string {
    return formatTimestamp(timestamp);
  }

  capitalizeType(type: string | undefined): string {
    if (!type) return '-';
    return type.charAt(0).toUpperCase() + type.slice(1);
  }

  get taskProgressInfo(): ProgressHeader {
    const task = this.loadingTask;
    if (!task) return { header: '', subtitle: '', detail: '', eta: '' };
    return formatProgressHeader(task, this.taskKind(), task.embedder);
  }

  get taskBar(): ProgressBarState {
    const task = this.loadingTask;
    if (!task) return { value: 0, max: 1, indeterminate: true };
    return progressBarState(task);
  }

  // `vt-job-progress` stops the click before it reaches the row, so no event.
  onCancelTask(): void {
    if (this.loadingTask) {
      this.cancelTask.emit(this.loadingTask.task_id);
    }
  }

  onDismissTask(event: MouseEvent): void {
    event.stopPropagation();
    if (this.loadingTask) {
      this.dismissTask.emit(this.loadingTask.task_id);
    }
  }
}
