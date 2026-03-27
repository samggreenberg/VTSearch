import { Component, ElementRef, EventEmitter, HostBinding, HostListener, Input, Output, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoadingTask } from '../../../models/api.models';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
import { formatProgressFraction } from '../../../utils/format-progress';

@Component({
  selector: 'vt-dataset-card',
  standalone: true,
  imports: [CommonModule, FormsModule, ProgressBarComponent],
  templateUrl: './dataset-card.component.html',
  styleUrl: './dataset-card.component.scss',
})
export class DatasetCardComponent {
  @Input() dataset: any;
  @Input() currentUser = '';
  @Input() isDefaultLogin = true;
  @Input() @HostBinding('class.selected') selected = false;
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

  get isOwner(): boolean {
    return this.dataset?.created_by === this.currentUser;
  }

  @ViewChild('renameInput') renameInput?: ElementRef<HTMLInputElement>;

  editing = false;
  editName = '';

  startRename(event: MouseEvent): void {
    event.stopPropagation();
    this.editing = true;
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

  formatDate(timestamp: number | null): string {
    if (!timestamp) return '-';
    const d = new Date(timestamp * 1000);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
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
