import { Component, ElementRef, EventEmitter, HostBinding, HostListener, Input, Output, ViewChild } from '@angular/core';
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
  @Input() currentUser = '';
  @Input() isDefaultLogin = true;
  @Input() @HostBinding('class.selected') selected = false;
  @Output() rowClick = new EventEmitter<MouseEvent>();

  @HostListener('click', ['$event'])
  onClick(event: MouseEvent): void {
    this.rowClick.emit(event);
  }
  @Output() rename = new EventEmitter<string>();
  @Output() delete = new EventEmitter<void>();
  @Output() security = new EventEmitter<void>();

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

  onSecurity(event: MouseEvent): void {
    event.stopPropagation();
    this.security.emit();
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
}
