import { ChangeDetectionStrategy, Component, ElementRef, HostListener, effect, inject, input, output, viewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoadingTask } from '../../../models/api.models';
import { JobProgressComponent } from '../../job-progress/job-progress.component';
import {
  ProgressBarState,
  ProgressHeader,
  formatProgressHeader,
  progressBarState,
} from '../../../utils/format-progress';
import { formatTimestamp } from '../../../utils/format-date';
import { ContextMenuComponent, ContextMenuItem } from '../../context-menu/context-menu.component';
import { IconComponent } from '../../icon/icon.component';
import { buildDetectorCardMenuItems, CARD_MENU_MIN_WIDTH, overflowMenuItems } from '../card-context-menu-items';
import { DashboardLoadingTasksService } from '../../../services/dashboard-loading-tasks.service';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'tr[vt-detector-card]',
  standalone: true,
  host: {
    class: 'entity-card',
    '[class.selected]': 'selected()',
    '[class.dimmed]': 'dimmed()',
    '[class.loading-error]': 'hasLoadingError',
  },
  imports: [CommonModule, FormsModule, JobProgressComponent, ContextMenuComponent, IconComponent],
  templateUrl: './detector-card.component.html',
  styleUrl: './detector-card.component.scss',
})
export class DetectorCardComponent {
  private readonly loadingTasksSvc = inject(DashboardLoadingTasksService);

  readonly detector = input<any>(undefined);
  readonly currentUser = input('');
  readonly isDefaultLogin = input(true);
  readonly columnOrder = input<string[]>([]);
  readonly selected = input(false);
  readonly dimmed = input(false);
  readonly loadingTask = input<LoadingTask | undefined>(undefined);

  /** True while this row's delete-confirm dialog is open (driven by the
   *  dashboard's `deletingDetectorId`). Spins the trash icon to 90° while open;
   *  the reverse animation plays back to 0° once the dialog resolves. */
  readonly deleting = input(false);
  wasDeleting = false;
  private previousDeleting = false;

  get hasLoadingError(): boolean {
    return !!this.loadingTask()?.error;
  }

  constructor() {
    effect(() => {
      const value = this.deleting();
      if (value && !this.previousDeleting) this.wasDeleting = true;
      this.previousDeleting = value;
    });
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
    if (this.loadingTask()) return;
    event.preventDefault();
    // Right-click gets the complete action list; the ⋯ button gets the overflow
    // subset (see openMenuAt).
    this.openMenuAt(event.clientX, event.clientY, false);
  }
  readonly rename = output<string>();
  readonly delete = output<void>();
  readonly export = output<void>();
  readonly addLabels = output<void>();
  readonly load = output<void>();
  readonly unload = output<void>();
  readonly browse = output<void>();
  readonly stats = output<void>();
  readonly cancelTask = output<string>();
  readonly dismissTask = output<string>();
  readonly checkboxToggle = output<void>();
  readonly security = output<void>();
  /** Emits the new AutoRun membership: true for "Move to AutoRun", false for
   *  "Move to Drafts". */
  readonly setAutorun = output<boolean>();

  /** True for an AutoRun detector (``autofind``): the row is frozen, so the
   *  editing affordances (rename pencil, inline Delete) are hidden and the
   *  menu omits the editing verbs — Move to Drafts first to edit. */
  get frozen(): boolean {
    return !!this.detector()?.autofind;
  }

  /** True for a just-created detector that has never been trained (no labels
   *  yet). Drives the "Empty" hint so a fresh detector reads as new rather than
   *  broken. */
  get isUntrained(): boolean {
    return (this.detector()?.num_training ?? 0) === 0;
  }

  /** True when the current user created this detector (only the creator may
   *  rename/delete it or edit its access list). */
  get isOwner(): boolean {
    return this.detector()?.created_by === this.currentUser();
  }

  readonly renameInput = viewChild<ElementRef<HTMLInputElement>>('renameInput');

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
    this.editName = this.detector().name;
    setTimeout(() => this.renameInput()?.nativeElement.focus());
  }

  /** Open the shared action menu at a viewport point.
   *  ``buildDetectorCardMenuItems`` is the single source of truth for the action
   *  list; ``overflow`` trims the verbs already shown as inline icons (Load,
   *  Browse, Delete) so the ⋯ button reads as "more" while right-click stays
   *  complete. */
  private openMenuAt(x: number, y: number, overflow: boolean): void {
    const items = buildDetectorCardMenuItems(this.detector(), {
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
      case 'add-labels':
        this.addLabels.emit();
        break;
      case 'export':
        this.export.emit();
        break;
      case 'stats':
        this.stats.emit();
        break;
      case 'move-to-autorun':
        this.setAutorun.emit(true);
        break;
      case 'move-to-drafts':
        this.setAutorun.emit(false);
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
    if (trimmed && trimmed !== this.detector().name) {
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
    if (!this.deleting()) {
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

  onUnload(event: MouseEvent): void {
    event.stopPropagation();
    this.unload.emit();
  }

  // `vt-job-progress` stops the click before it reaches the row, so no event.
  onCancelTask(): void {
    const task = this.loadingTask();
    if (task) {
      this.cancelTask.emit(task.task_id);
    }
  }

  onDismissTask(event: MouseEvent): void {
    event.stopPropagation();
    const task = this.loadingTask();
    if (task) {
      this.dismissTask.emit(task.task_id);
    }
  }

  taskBar(): ProgressBarState {
    const t = this.loadingTask();
    if (!t) return { value: 0, max: 1, indeterminate: true };
    return progressBarState(t);
  }

  get taskProgressInfo(): ProgressHeader {
    const task = this.loadingTask();
    if (!task) return { header: '', subtitle: '', detail: '', eta: '' };
    return formatProgressHeader(task, 'detector', task.embedder);
  }

  /** True once Cancel has been clicked on this row's task and the backend is
   *  still unwinding it; swaps the Cancel button for a "Cancelling…" badge. */
  get taskCancelling(): boolean {
    const task = this.loadingTask();
    return !!task && this.loadingTasksSvc.isCancelling(task.task_id);
  }
}
