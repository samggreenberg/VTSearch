import { Component, ElementRef, EventEmitter, HostBinding, HostListener, Input, Output, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoadingTask } from '../../../models/api.models';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
import { formatProgressFraction } from '../../../utils/format-progress';

@Component({
  selector: 'vt-model-card',
  standalone: true,
  imports: [CommonModule, FormsModule, ProgressBarComponent],
  templateUrl: './model-card.component.html',
  styleUrl: './model-card.component.scss',
})
export class ModelCardComponent {
  @Input() model: any;
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
  @Output() delete = new EventEmitter<void>();
  @Output() export = new EventEmitter<void>();
  @Output() addLabels = new EventEmitter<void>();
  @Output() load = new EventEmitter<void>();
  @Output() unload = new EventEmitter<void>();
  @Output() cancelTask = new EventEmitter<string>();
  @Output() dismissTask = new EventEmitter<string>();
  @Output() autorunToggle = new EventEmitter<boolean>();

  @ViewChild('renameInput') renameInput?: ElementRef<HTMLInputElement>;

  editing = false;
  wasEditing = false;
  editName = '';

  startRename(event: MouseEvent): void {
    event.stopPropagation();
    this.editing = true;
    this.wasEditing = true;
    this.editName = this.model.name;
    setTimeout(() => this.renameInput?.nativeElement.focus());
  }

  confirmRename(): void {
    const trimmed = this.editName.trim();
    if (trimmed && trimmed !== this.model.name) {
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

  onAddLabels(event: MouseEvent): void {
    event.stopPropagation();
    this.addLabels.emit();
  }

  onExport(event: MouseEvent): void {
    event.stopPropagation();
    this.export.emit();
  }

  onAutorunToggle(event: Event): void {
    event.stopPropagation();
    const checked = (event.target as HTMLInputElement).checked;
    this.autorunToggle.emit(checked);
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
    return !(t && t.current != null && t.total != null && t.total > 0);
  }

  formatFraction(current: number, total: number): string {
    return formatProgressFraction(current, total);
  }
}
