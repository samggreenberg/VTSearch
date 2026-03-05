import { Component, EventEmitter, HostBinding, HostListener, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'vt-dataset-card',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './dataset-card.component.html',
  styleUrl: './dataset-card.component.scss',
})
export class DatasetCardComponent {
  @Input() dataset: any;
  @Input() @HostBinding('class.selected') selected = false;
  @Output() rowClick = new EventEmitter<MouseEvent>();

  @HostListener('click', ['$event'])
  onClick(event: MouseEvent): void {
    this.rowClick.emit(event);
  }
  @Output() rename = new EventEmitter<string>();
  @Output() delete = new EventEmitter<void>();
  @Output() load = new EventEmitter<void>();

  editing = false;
  editName = '';

  startRename(event: MouseEvent): void {
    event.stopPropagation();
    this.editing = true;
    this.editName = this.dataset.name;
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

  onDelete(event: MouseEvent): void {
    event.stopPropagation();
    this.delete.emit();
  }

  onLoad(event: MouseEvent): void {
    event.stopPropagation();
    this.load.emit();
  }

  formatDate(timestamp: number | null): string {
    if (!timestamp) return '-';
    return new Date(timestamp * 1000).toLocaleDateString();
  }
}
