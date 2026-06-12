import { Component, ElementRef, EventEmitter, HostBinding, HostListener, Input, OnChanges, Output, SimpleChanges, ViewChild } from '@angular/core';
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

@Component({
  selector: 'vt-detector-card',
  standalone: true,
  imports: [CommonModule, FormsModule, ProgressBarComponent],
  templateUrl: './detector-card.component.html',
  styleUrl: './detector-card.component.scss',
})
export class DetectorCardComponent implements OnChanges {
  @Input() detector: any;
  @Input() currentUser = '';
  @Input() isDefaultLogin = true;
  @Input() columnOrder: string[] = [];
  @Input() @HostBinding('class.selected') selected = false;
  @Input() @HostBinding('class.dimmed') dimmed = false;
  @Input() loadingTask?: LoadingTask;

  @HostBinding('class.loading-error')
  get hasLoadingError(): boolean {
    return !!this.loadingTask?.error;
  }

  @Output() rowClick = new EventEmitter<MouseEvent>();

  @HostListener('click', ['$event'])
  onClick(event: MouseEvent): void {
    this.rowClick.emit(event);
  }
  @Output() rename = new EventEmitter<string>();
  @Output() delete = new EventEmitter<void>();
  @Output() export = new EventEmitter<void>();
  @Output() addLabels = new EventEmitter<void>();
  @Output() load = new EventEmitter<void>();
  @Output() unload = new EventEmitter<void>();
  @Output() browse = new EventEmitter<void>();
  @Output() stats = new EventEmitter<void>();
  @Output() cancelTask = new EventEmitter<string>();
  @Output() dismissTask = new EventEmitter<string>();
  @Output() checkboxToggle = new EventEmitter<void>();
  @Output() security = new EventEmitter<void>();

  /** True when the current user created this detector (only the creator may
   *  rename/delete it or edit its access list). */
  get isOwner(): boolean {
    return this.detector?.created_by === this.currentUser;
  }

  onSecurity(event: MouseEvent): void {
    event.stopPropagation();
    this.security.emit();
  }

  @ViewChild('renameInput') renameInput?: ElementRef<HTMLInputElement>;

  @Input() deleteConfirmOpen = false;
  @Input() addLabelsOpen = false;
  @Input() exportOpen = false;
  @Input() statsOpen = false;

  editing = false;
  wasEditing = false;
  editName = '';
  wasDeleteOpen = false;
  wasAddLabelsOpen = false;
  wasExportOpen = false;
  wasStatsOpen = false;

  startRename(event: MouseEvent): void {
    event.stopPropagation();
    this.editing = true;
    this.wasEditing = true;
    this.editName = this.detector.name;
    setTimeout(() => this.renameInput?.nativeElement.focus());
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
    if (!this.deleteConfirmOpen) {
      this.wasDeleteOpen = false;
    }
  }

  onCapAnimationEnd(): void {
    if (!this.addLabelsOpen) {
      this.wasAddLabelsOpen = false;
    }
  }

  onExportAnimationEnd(): void {
    if (!this.exportOpen) {
      this.wasExportOpen = false;
    }
  }

  onPieAnimationEnd(): void {
    if (!this.statsOpen) {
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
