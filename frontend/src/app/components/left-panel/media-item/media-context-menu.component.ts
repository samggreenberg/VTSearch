import { CommonModule } from '@angular/common';
import {
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  Input,
  Output,
} from '@angular/core';

export interface MediaContextMenuItem {
  id: string;
  label: string;
  title?: string;
  disabled?: boolean;
}

@Component({
  selector: 'vt-media-context-menu',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './media-context-menu.component.html',
  styleUrl: './media-context-menu.component.scss',
})
export class MediaContextMenuComponent {
  @Input({ required: true }) x = 0;
  @Input({ required: true }) y = 0;
  @Input() items: MediaContextMenuItem[] = [];

  @Output() actionSelected = new EventEmitter<string>();
  @Output() dismissed = new EventEmitter<void>();

  constructor(private host: ElementRef<HTMLElement>) {}

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    if (!this.host.nativeElement.contains(event.target as Node)) {
      this.dismissed.emit();
    }
  }

  @HostListener('document:contextmenu', ['$event'])
  onDocumentContextMenu(event: MouseEvent): void {
    if (!this.host.nativeElement.contains(event.target as Node)) {
      this.dismissed.emit();
    }
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    this.dismissed.emit();
  }

  onItemClick(event: MouseEvent, item: MediaContextMenuItem): void {
    event.stopPropagation();
    if (item.disabled) return;
    this.actionSelected.emit(item.id);
  }
}
