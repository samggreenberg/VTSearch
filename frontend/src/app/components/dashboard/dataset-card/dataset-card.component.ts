import { Component, ElementRef, EventEmitter, HostBinding, HostListener, Input, OnChanges, Output, SimpleChanges, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoadingTask } from '../../../models/api.models';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
import { formatProgressFraction } from '../../../utils/format-progress';
import { formatTimestamp } from '../../../utils/format-date';

@Component({
  selector: 'vt-dataset-card',
  standalone: true,
  imports: [CommonModule, FormsModule, ProgressBarComponent],
  templateUrl: './dataset-card.component.html',
  styleUrl: './dataset-card.component.scss',
})
export class DatasetCardComponent implements OnChanges {
  @Input() dataset: any;
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
  @Output() stats = new EventEmitter<void>();
  @Output() delete = new EventEmitter<void>();
  @Output() load = new EventEmitter<void>();
  @Output() security = new EventEmitter<void>();
  @Output() cancelTask = new EventEmitter<string>();
  @Output() dismissTask = new EventEmitter<string>();
  @Output() checkboxToggle = new EventEmitter<void>();

  get isOwner(): boolean {
    return this.dataset?.created_by === this.currentUser;
  }

  @ViewChild('renameInput') renameInput?: ElementRef<HTMLInputElement>;

  @Input() statsOpen = false;
  @Input() deleteConfirmOpen = false;

  editing = false;
  wasEditing = false;
  editName = '';
  wasStatsOpen = false;
  wasDeleteOpen = false;

  startRename(event: MouseEvent): void {
    event.stopPropagation();
    this.editing = true;
    this.wasEditing = true;
    this.editName = this.dataset.name;
    setTimeout(() => this.renameInput?.nativeElement.focus());
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
    if (!this.statsOpen) {
      this.wasStatsOpen = false;
    }
  }

  onTrashAnimationEnd(): void {
    if (!this.deleteConfirmOpen) {
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

  get taskProgressMessage(): string {
    const task = this.loadingTask;
    if (!task) return '';
    let msg = task.message || 'Loading...';
    if (task.step != null && task.total_steps != null && task.total_steps > 1) {
      msg = `[Step ${task.step}/${task.total_steps}] ${msg}`;
    }
    if (task.current != null && task.total != null && task.total > 0) {
      const fraction = `(${formatProgressFraction(task.current, task.total)})`;
      const stepEnd = msg.indexOf('] ');
      if (stepEnd !== -1) {
        msg = msg.slice(0, stepEnd + 2) + fraction + ' ' + msg.slice(stepEnd + 2);
      } else {
        msg = fraction + ' ' + msg;
      }
    }
    return msg;
  }

  get taskIsIndeterminate(): boolean {
    const task = this.loadingTask;
    return !(task && task.current != null && task.total != null && task.total > 0);
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
}
