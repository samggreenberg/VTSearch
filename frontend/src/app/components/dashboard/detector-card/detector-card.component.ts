import { Component, ElementRef, EventEmitter, HostBinding, HostListener, Input, OnChanges, Output, SimpleChanges, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoadingTask } from '../../../models/api.models';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
import { DetectorSwatchComponent } from '../../detector-swatch/detector-swatch.component';
import {
  ProgressHeader,
  formatProgressHeader,
  isProgressIndeterminate,
} from '../../../utils/format-progress';
import { formatTimestamp } from '../../../utils/format-date';
import { detectorHue } from '../../../utils/detector-color';

@Component({
  selector: 'vt-detector-card',
  standalone: true,
  imports: [CommonModule, FormsModule, ProgressBarComponent, DetectorSwatchComponent],
  templateUrl: './detector-card.component.html',
  styleUrl: './detector-card.component.scss',
})
export class DetectorCardComponent implements OnChanges {
  @Input() detector: any;
  @Input() columnOrder: string[] = [];
  @Input() @HostBinding('class.selected') selected = false;
  @Input() @HostBinding('class.dimmed') dimmed = false;
  @Input() loadingTask?: LoadingTask;

  @HostBinding('style.--detector-hue')
  get hueStyle(): string {
    return String(detectorHue(this.detector?.name || ''));
  }

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
  @Output() checkboxToggle = new EventEmitter<void>();

  @ViewChild('renameInput') renameInput?: ElementRef<HTMLInputElement>;

  @Input() deleteConfirmOpen = false;
  @Input() addLabelsOpen = false;

  editing = false;
  wasEditing = false;
  editName = '';
  wasDeleteOpen = false;
  wasAddLabelsOpen = false;

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

  onAutorunToggle(event: MouseEvent): void {
    event.stopPropagation();
    this.autorunToggle.emit(!this.detector.autorun);
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
    if (!task) return { header: '', subtitle: '', detail: '' };
    return formatProgressHeader(task, 'detector', task.embedder);
  }
}
