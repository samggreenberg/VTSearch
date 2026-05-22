import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';

interface Shortcut {
  keys: string[];
  description: string;
}

interface ShortcutGroup {
  title: string;
  shortcuts: Shortcut[];
}

interface ShortcutSection {
  header?: string;
  groups: ShortcutGroup[];
}

@Component({
  selector: 'vt-keyboard-help-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './keyboard-help-modal.component.html',
  styleUrl: './keyboard-help-modal.component.scss',
})
export class KeyboardHelpModalComponent {
  @Output() closed = new EventEmitter<void>();

  readonly sections: ShortcutSection[] = [
    {
      header: 'In the Train / Find window only',
      groups: [
        {
          title: 'Voting',
          shortcuts: [
            { keys: ['→'], description: 'Vote good' },
            { keys: ['←'], description: 'Vote bad' },
          ],
        },
        {
          title: 'Playback',
          shortcuts: [
            { keys: ['Space'], description: 'Play / pause audio or video' },
            { keys: ['↑'], description: 'Volume up' },
            { keys: ['↓'], description: 'Volume down' },
          ],
        },
        {
          title: 'Image viewer',
          shortcuts: [
            { keys: ['+'], description: 'Zoom in' },
            { keys: ['-'], description: 'Zoom out' },
            { keys: ['['], description: 'Rotate left' },
            { keys: [']'], description: 'Rotate right' },
            { keys: ['Shift', 'drag'], description: 'Draw region box (or use the Marquee button)' },
            { keys: ['Esc'], description: 'Cancel armed vote / clear region box' },
          ],
        },
      ],
    },
    {
      header: 'Anywhere',
      groups: [
        {
          title: 'General',
          shortcuts: [
            { keys: ['?'], description: 'Show this help' },
            { keys: ['Esc'], description: 'Close modal or dropdown' },
          ],
        },
      ],
    },
  ];

  close(): void {
    this.closed.emit();
  }
}
