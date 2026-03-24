import { Component, ElementRef, EventEmitter, HostBinding, HostListener, Input, Output, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'vt-model-card',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './model-card.component.html',
  styleUrl: './model-card.component.scss',
})
export class ModelCardComponent {
  @Input() model: any;
  @Input() @HostBinding('class.selected') selected = false;
  @Output() rowClick = new EventEmitter<MouseEvent>();

  @HostListener('click', ['$event'])
  onClick(event: MouseEvent): void {
    this.rowClick.emit(event);
  }
  @Output() rename = new EventEmitter<string>();
  @Output() delete = new EventEmitter<void>();
  @Output() export = new EventEmitter<void>();
  @Output() load = new EventEmitter<void>();
  @Output() autorunToggle = new EventEmitter<boolean>();

  @ViewChild('renameInput') renameInput?: ElementRef<HTMLInputElement>;

  editing = false;
  editName = '';

  startRename(event: MouseEvent): void {
    event.stopPropagation();
    this.editing = true;
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
}
