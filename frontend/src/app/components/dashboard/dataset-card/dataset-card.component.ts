import { ChangeDetectionStrategy, Component, ElementRef, HostBinding, HostListener, Input, input, output, ViewChild } from '@angular/core';
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
import { buildDatasetCardMenuItems, CARD_MENU_MIN_WIDTH, overflowMenuItems } from '../card-context-menu-items';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-dataset-card',
  standalone: true,
  imports: [CommonModule, FormsModule, JobProgressComponent, ContextMenuComponent],
  templateUrl: './dataset-card.component.html',
  styleUrl: './dataset-card.component.scss',
})
export class DatasetCardComponent {
  @Input() dataset: any;
  readonly currentUser = input('');
  readonly isDefaultLogin = input(true);
  @Input() columnOrder: string[] = [];
  @Input() @HostBinding('class.selected') selected = false;
  @Input() @HostBinding('class.dimmed') dimmed = false;
  @Input() loadingTask?: LoadingTask;

  /** True while this row's delete-confirm dialog is open (driven by the
   *  dashboard's `deletingDatasetId`). Spins the trash icon to 90° while open;
   *  the reverse animation plays back to 0° once the dialog resolves. */
  @Input()
  set deleting(value: boolean) {
    if (value && !this._deleting) this.wasDeleting = true;
    this._deleting = value;
  }
  get deleting(): boolean {
    return this._deleting;
  }
  private _deleting = false;
  wasDeleting = false;
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
    // Right-click gets the complete action list; the ⋯ button gets the overflow
    // subset (see openMenuAt).
    this.openMenuAt(event.clientX, event.clientY, false);
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

  editing = false;
  wasEditing = false;
  editName = '';

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

  /** Open the shared action menu at a viewport point.
   *  ``buildDatasetCardMenuItems`` is the single source of truth for the action
   *  list; ``overflow`` trims the verbs already shown as inline icons (Load,
   *  Browse, Delete) so the ⋯ button reads as "more" while right-click stays
   *  complete. */
  private openMenuAt(x: number, y: number, overflow: boolean): void {
    const items = buildDatasetCardMenuItems(this.dataset, {
      isDefaultLogin: this.isDefaultLogin(),
      isOwner: this.isOwner,
    });
    this.contextMenuItems = overflow ? overflowMenuItems(items) : items;
    this.contextMenuX = x;
    this.contextMenuY = y;
    this.contextMenuOpen = true;
  }

  /** Open the overflow action menu from the ⋯ button, right-aligned under it so
   *  the menu never spills off the viewport's right edge. */
  onOverflow(event: MouseEvent): void {
    event.stopPropagation();
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    this.openMenuAt(Math.max(8, rect.right - CARD_MENU_MIN_WIDTH), rect.bottom + 4, true);
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

  onPencilAnimationEnd(): void {
    if (!this.editing) {
      this.wasEditing = false;
    }
  }

  onDeleteAnimationEnd(): void {
    if (!this._deleting) {
      this.wasDeleting = false;
    }
  }

  onRenameKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      this.confirmRename();
    } else if (event.key === 'Escape') {
      this.cancelRename();
    }
  }

  onLoad(event: MouseEvent): void {
    event.stopPropagation();
    this.load.emit();
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
