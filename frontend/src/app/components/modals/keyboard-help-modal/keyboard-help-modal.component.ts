import { Component, EventEmitter, Output, inject, signal, OnInit, SecurityContext } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { marked } from 'marked';
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

type Tab = 'shortcuts' | 'guide';

@Component({
  selector: 'vt-keyboard-help-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './keyboard-help-modal.component.html',
  styleUrl: './keyboard-help-modal.component.scss',
})
export class KeyboardHelpModalComponent implements OnInit {
  @Output() closed = new EventEmitter<void>();

  private readonly http = inject(HttpClient);
  private readonly sanitizer = inject(DomSanitizer);

  readonly activeTab = signal<Tab>('shortcuts');
  readonly guideHtml = signal<SafeHtml | null>(null);
  readonly guideError = signal<string | null>(null);
  private guideLoaded = false;

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

  ngOnInit(): void {
    // Defer guide fetch until the user opens that tab.
  }

  selectTab(tab: Tab): void {
    this.activeTab.set(tab);
    if (tab === 'guide' && !this.guideLoaded) {
      this.loadGuide();
    }
  }

  private loadGuide(): void {
    this.guideLoaded = true;
    this.http.get('assets/docs/USER_GUIDE.md', { responseType: 'text' }).subscribe({
      next: (md) => {
        const rendered = marked.parse(md, { async: false }) as string;
        const safe = this.sanitizer.sanitize(SecurityContext.HTML, rendered) ?? '';
        this.guideHtml.set(this.sanitizer.bypassSecurityTrustHtml(safe));
      },
      error: (err) => {
        this.guideError.set(`Failed to load user guide: ${err?.message ?? err}`);
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
