import { Component, ElementRef, Input, ViewChild, output } from '@angular/core';

import { FormsModule } from '@angular/forms';

@Component({
  selector: 'vt-detector-context-bar',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './detector-context-bar.component.html',
  styleUrl: './detector-context-bar.component.scss',
})
export class DetectorContextBarComponent {
  @Input() detectorName = '';
  @Input() visible = false;
  readonly renamed = output<string>();

  editing = false;
  editValue = '';

  @ViewChild('renameInput') renameInput!: ElementRef<HTMLInputElement>;

  startRename(): void {
    if (!this.detectorName) return;
    this.editValue = this.detectorName;
    this.editing = true;
    setTimeout(() => {
      this.renameInput?.nativeElement.focus();
      this.renameInput?.nativeElement.select();
    });
  }

  finishRename(): void {
    if (!this.editing) return;
    const newName = this.editValue.trim();
    this.editing = false;
    if (newName && newName !== this.detectorName) {
      this.renamed.emit(newName);
    }
  }

  cancelRename(): void {
    this.editing = false;
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter') {
      event.preventDefault();
      this.finishRename();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      this.cancelRename();
    }
  }
}
