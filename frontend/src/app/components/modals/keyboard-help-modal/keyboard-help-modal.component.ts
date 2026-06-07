import { Component, EventEmitter, Output, inject, signal, OnInit, SecurityContext } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { marked } from 'marked';
import { ModalComponent } from '../../modal/modal.component';
import { ThemeService, EffectiveTheme } from '../../../services/theme.service';

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

/**
 * Directory the user guide is served from. Image paths inside
 * USER_GUIDE.md are written relative to the doc's repo location
 * (``docs/user/USER_GUIDE.md``), e.g. ``assets/foo.png`` -> on disk
 * ``docs/user/assets/foo.png``. The Angular build copies both the doc and
 * its ``assets/`` folder under ``/assets/docs``, so a relative src must be
 * resolved against this base to load in the app.
 */
const GUIDE_ASSET_BASE = 'assets/docs/';

/** Matches a theme-suffixed screenshot filename, e.g. ``foo.light.png``. */
const THEME_VARIANT_RE = /\.(light|dark)\.(png|jpe?g|webp|gif|avif)$/i;

/** True for an absolute URL or root-relative path (left untouched). */
const ABSOLUTE_SRC_RE = /^([a-z]+:)?\/\//i;

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
  private readonly themeService = inject(ThemeService);

  readonly activeTab = signal<Tab>('shortcuts');
  readonly guideHtml = signal<SafeHtml | null>(null);
  readonly guideError = signal<string | null>(null);
  private guideLoaded = false;
  /** Raw markdown, cached so a theme switch re-renders without re-fetching. */
  private rawGuide: string | null = null;

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

  constructor() {
    // Re-render the guide whenever the effective theme changes so embedded
    // screenshots track the user's current theme (no side-by-side, no extra
    // control). No-op until the guide has been loaded once.
    this.themeService.theme$.pipe(takeUntilDestroyed()).subscribe(() => {
      if (this.rawGuide !== null) {
        this.renderGuide(this.rawGuide);
      }
    });
  }

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
        this.rawGuide = md;
        this.renderGuide(md);
      },
      error: (err) => {
        this.guideError.set(`Failed to load user guide: ${err?.message ?? err}`);
      },
    });
  }

  /** Parse markdown, theme-match + resolve its images, sanitize, and show. */
  private renderGuide(md: string): void {
    const rendered = marked.parse(md, { async: false }) as string;
    const themed = this.applyImagePolicy(rendered, this.themeService.resolveEffectiveTheme(this.themeService.currentTheme));
    const safe = this.sanitizer.sanitize(SecurityContext.HTML, themed) ?? '';
    this.guideHtml.set(this.sanitizer.bypassSecurityTrustHtml(safe));
  }

  /**
   * Rewrite the rendered guide's images for in-app display:
   *
   * - Collapse each ``<picture>`` (used so GitHub/GitLab honour
   *   ``prefers-color-scheme``) to its inner ``<img>``; in the app we pick
   *   the theme ourselves rather than relying on the OS preference.
   * - Swap any ``*.light.*`` / ``*.dark.*`` screenshot to the variant
   *   matching the app's current effective theme (``light`` -> light,
   *   everything else -> dark; there are no high-viz screenshot variants).
   * - Resolve relative ``src`` paths against the guide's served directory.
   */
  private applyImagePolicy(html: string, theme: EffectiveTheme): string {
    if (typeof DOMParser === 'undefined') {
      return html;
    }
    const doc = new DOMParser().parseFromString(html, 'text/html');

    doc.querySelectorAll('picture').forEach((pic) => {
      const img = pic.querySelector('img');
      if (img) {
        pic.replaceWith(img);
      } else {
        pic.remove();
      }
    });

    const wantLight = theme === 'light';
    doc.querySelectorAll('img').forEach((img) => {
      img.removeAttribute('srcset');
      let src = img.getAttribute('src') ?? '';
      if (!src) {
        return;
      }
      if (THEME_VARIANT_RE.test(src)) {
        src = src.replace(THEME_VARIANT_RE, `.${wantLight ? 'light' : 'dark'}.$2`);
      }
      if (!ABSOLUTE_SRC_RE.test(src) && !src.startsWith('/')) {
        src = GUIDE_ASSET_BASE + src;
      }
      img.setAttribute('src', src);
      img.setAttribute('loading', 'lazy');
    });

    return doc.body.innerHTML;
  }

  close(): void {
    this.closed.emit();
  }
}
