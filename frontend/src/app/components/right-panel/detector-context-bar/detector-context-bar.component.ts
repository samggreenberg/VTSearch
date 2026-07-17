import { ChangeDetectionStrategy, Component, ElementRef, input, output, viewChild } from '@angular/core';

import { FormsModule } from '@angular/forms';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-detector-context-bar',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './detector-context-bar.component.html',
  styleUrl: './detector-context-bar.component.scss',
})
export class DetectorContextBarComponent {
  readonly detectorName = input('');
  readonly visible = input(false);
  readonly renamed = output<string>();

  editing = false;
  editValue = '';

  readonly renameInput = viewChild<ElementRef<HTMLInputElement>>('renameInput');

  startRename(): void {
    if (!this.detectorName()) return;
    this.editValue = this.detectorName();
    this.editing = true;
    setTimeout(() => {
      const input = this.renameInput();
      input?.nativeElement.focus();
      input?.nativeElement.select();
    });
  }

  finishRename(): void {
    if (!this.editing) return;
    const newName = this.editValue.trim();
    this.editing = false;
    if (newName && newName !== this.detectorName()) {
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
