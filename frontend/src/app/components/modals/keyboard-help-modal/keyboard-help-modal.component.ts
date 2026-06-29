import { ChangeDetectionStrategy, Component, inject, OnInit, output, SecurityContext, signal } from '@angular/core';

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

/** A keyboard-shortcuts context: one sub-tab under the "Keyboard shortcuts" tab.
 *  Splitting the (growing) shortcut list by where the keys apply keeps each
 *  panel short instead of one long scroll. */
interface ShortcutContext {
  id: string;
  label: string;
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
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-keyboard-help-modal',
  standalone: true,
  imports: [ModalComponent],
  templateUrl: './keyboard-help-modal.component.html',
  styleUrl: './keyboard-help-modal.component.scss',
})
export class KeyboardHelpModalComponent implements OnInit {
  readonly closed = output<void>();

  private readonly http = inject(HttpClient);
  private readonly sanitizer = inject(DomSanitizer);
  private readonly themeService = inject(ThemeService);

  readonly activeTab = signal<Tab>('shortcuts');
  /** Which shortcut context's panel is shown under the "Keyboard shortcuts" tab. */
  readonly activeContext = signal<string>('find');
  readonly guideHtml = signal<SafeHtml | null>(null);
  readonly guideError = signal<string | null>(null);
  private guideLoaded = false;
  /** Raw markdown, cached so a theme switch re-renders without re-fetching. */
  private rawGuide: string | null = null;

  /**
   * Shortcuts grouped by the context they apply in. Each entry becomes a sub-tab
   * under the "Keyboard shortcuts" tab so the sheet stays scannable as the list
   * grows; the "General" context holds the keys that work anywhere.
   */
  readonly contexts: ShortcutContext[] = [
    {
      id: 'find',
      label: 'Train / Find',
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
      id: 'browse',
      label: 'Browser',
      groups: [
        {
          title: 'Navigating the map',
          shortcuts: [
            { keys: ['↑ ↓ ← →'], description: 'Pan the view' },
            { keys: ['+'], description: 'Zoom in' },
            { keys: ['-'], description: 'Zoom out' },
            { keys: ['Ctrl', 'A'], description: 'Select every bin fully in view' },
          ],
        },
        {
          title: 'Bin details window',
          shortcuts: [
            { keys: ['↑ ↓ ← →'], description: 'Move the viewed item within the grid' },
            { keys: ['+'], description: 'Make the detail image bigger' },
            { keys: ['-'], description: 'Make the detail image smaller' },
            { keys: ['Ctrl', 'A'], description: 'Select all items in this bin' },
          ],
        },
      ],
    },
    {
      id: 'general',
      label: 'General',
      groups: [
        {
          title: 'Anywhere',
          shortcuts: [
            { keys: ['?'], description: 'Show this help' },
            { keys: ['Esc'], description: 'Close modal or dropdown' },
          ],
        },
      ],
    },
  ];

  selectContext(id: string): void {
    this.activeContext.set(id);
  }

  /** Groups of the currently-selected context (the visible shortcut panel). */
  get activeGroups(): ShortcutGroup[] {
    return this.contexts.find((c) => c.id === this.activeContext())?.groups ?? [];
  }

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
