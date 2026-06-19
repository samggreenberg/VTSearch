import { Component, ElementRef, HostBinding, HostListener, Input, OnChanges, SimpleChanges, ViewChild, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoadingTask } from '../../../models/api.models';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
import {
  ProgressHeader,
  formatProgressHeader,
  isProgressIndeterminate,
} from '../../../utils/format-progress';
import { formatTimestamp } from '../../../utils/format-date';
import { ContextMenuComponent, ContextMenuItem } from '../../context-menu/context-menu.component';
import { buildDetectorCardMenuItems } from '../card-context-menu-items';

@Component({
  selector: 'vt-detector-card',
  standalone: true,
  imports: [CommonModule, FormsModule, ProgressBarComponent, ContextMenuComponent],
  templateUrl: './detector-card.component.html',
  styleUrl: './detector-card.component.scss',
})
export class DetectorCardComponent implements OnChanges {
  @Input() detector: any;
  readonly currentUser = input('');
  readonly isDefaultLogin = input(true);
  @Input() columnOrder: string[] = [];
  @Input() @HostBinding('class.selected') selected = false;
  @Input() @HostBinding('class.dimmed') dimmed = false;
  @Input() loadingTask?: LoadingTask;

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
    this.contextMenuItems = buildDetectorCardMenuItems(this.detector, {
      isDefaultLogin: this.isDefaultLogin(),
      isOwner: this.isOwner,
    });
    this.contextMenuX = event.clientX;
    this.contextMenuY = event.clientY;
    this.contextMenuOpen = true;
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

  /** True when the current user created this detector (only the creator may
   *  rename/delete it or edit its access list). */
  get isOwner(): boolean {
    return this.detector?.created_by === this.currentUser();
  }

  onSecurity(event: MouseEvent): void {
    event.stopPropagation();
    this.security.emit();
  }

  @ViewChild('renameInput') renameInput?: ElementRef<HTMLInputElement>;

  readonly deleteConfirmOpen = input(false);
  readonly addLabelsOpen = input(false);
  readonly exportOpen = input(false);
  readonly statsOpen = input(false);

  editing = false;
  wasEditing = false;
  editName = '';
  wasDeleteOpen = false;
  wasAddLabelsOpen = false;
  wasExportOpen = false;
  wasStatsOpen = false;

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
    this.editName = this.detector.name;
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
      case 'add-labels':
        this.addLabels.emit();
        break;
      case 'export':
        this.export.emit();
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
    if (trimmed && trimmed !== this.detector.name) {
      this.rename.emit(trimmed);
    }
    this.editing = false;
  }

  cancelRename(): void {
    this.editing = false;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['deleteConfirmOpen'] && !changes['deleteConfirmOpen'].currentValue && changes['deleteConfirmOpen'].previousValue) {
      this.wasDeleteOpen = true;
    }
    if (changes['addLabelsOpen'] && !changes['addLabelsOpen'].currentValue && changes['addLabelsOpen'].previousValue) {
      this.wasAddLabelsOpen = true;
    }
    if (changes['exportOpen'] && !changes['exportOpen'].currentValue && changes['exportOpen'].previousValue) {
      this.wasExportOpen = true;
    }
    if (changes['statsOpen'] && !changes['statsOpen'].currentValue && changes['statsOpen'].previousValue) {
      this.wasStatsOpen = true;
    }
  }

  onPencilAnimationEnd(): void {
    if (!this.editing) {
      this.wasEditing = false;
    }
  }

  onTrashAnimationEnd(): void {
    if (!this.deleteConfirmOpen()) {
      this.wasDeleteOpen = false;
    }
  }

  onCapAnimationEnd(): void {
    if (!this.addLabelsOpen()) {
      this.wasAddLabelsOpen = false;
    }
  }

  onExportAnimationEnd(): void {
    if (!this.exportOpen()) {
      this.wasExportOpen = false;
    }
  }

  onPieAnimationEnd(): void {
    if (!this.statsOpen()) {
      this.wasStatsOpen = false;
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

  onBrowse(event: MouseEvent): void {
    event.stopPropagation();
    this.browse.emit();
  }

  onStats(event: MouseEvent): void {
    event.stopPropagation();
    this.stats.emit();
  }

  onDelete(event: MouseEvent): void {
    event.stopPropagation();
    this.delete.emit();
  }

  onAddLabels(event: MouseEvent): void {
    event.stopPropagation();
    this.addLabels.emit();
  }

  onExport(event: MouseEvent): void {
    event.stopPropagation();
    this.export.emit();
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

  onCancelTask(event: MouseEvent): void {
    event.stopPropagation();
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

  taskIsIndeterminate(): boolean {
    const t = this.loadingTask;
    if (!t) return true;
    return isProgressIndeterminate(t);
  }

  get taskProgressInfo(): ProgressHeader {
    const task = this.loadingTask;
    if (!task) return { header: '', subtitle: '', detail: '', eta: '' };
    return formatProgressHeader(task, 'detector', task.embedder);
  }
}
