import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  OnInit,
  output,
  SecurityContext,
  signal,
} from '@angular/core';

import { HttpClient } from '@angular/common/http';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { marked } from 'marked';
import { ModalComponent } from '../../modal/modal.component';
import { ThemeService, EffectiveTheme } from '../../../services/theme.service';
import { SettingsStateService } from '../../../services/settings-state.service';

/** Fallback "Email us" recipient used until server settings resolve. Matches
 *  the backend default (``vtsearch.settings_models.DEFAULT_SUPPORT_EMAIL``). */
const DEFAULT_SUPPORT_EMAIL = 'sam.greenberg@gmail.com';

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

/**
 * GitHub's heading-anchor slug, so the guide's own table of contents (written
 * against GitHub's rendering) resolves in-app too: lowercase, drop everything
 * that isn't a word character / hyphen / space, then turn each remaining space
 * into a hyphen. Space-by-space rather than run-collapsing, because that is
 * what GitHub does and the guide's links are written to match it.
 */
export function headingSlug(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\w\- ]+/g, '')
    .replace(/ /g, '-');
}

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
  private readonly settingsState = inject(SettingsStateService);

  /** ``mailto:`` href for the "Email us" footer link, pre-addressed to the
   *  server's configured support address (``--support-email`` /
   *  ``VTSEARCH_SUPPORT_EMAIL`` / the persisted ``support_email`` setting),
   *  falling back to the built-in default until settings load. */
  readonly mailtoHref = computed(() => {
    const email = this.settingsState.settingsSignal()?.support_email?.trim() || DEFAULT_SUPPORT_EMAIL;
    return `mailto:${encodeURIComponent(email)}?subject=VTSearch%20Issue%3A`;
  });

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
            { keys: ['Space'], description: 'Select / deselect the viewed item' },
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
    // Heading ids go on *after* sanitization: Angular's HTML sanitizer strips
    // `id` (DOM-clobbering defence), so ids added before it would not survive.
    // Safe to add here because every id is derived by `headingSlug`, which
    // keeps only word characters and hyphens.
    this.guideHtml.set(this.sanitizer.bypassSecurityTrustHtml(this.addHeadingIds(safe)));
  }

  /**
   * Give every heading a GitHub-compatible `id`.
   *
   * `marked` v14 emits bare `<h2>` elements, so without this the guide's
   * table of contents (and its body cross-references) would link to anchors
   * that exist only on GitHub, leaving the in-app copy's whole TOC dead.
   * Collisions get GitHub's `-1`, `-2`, … suffix for the same reason.
   */
  private addHeadingIds(html: string): string {
    if (typeof DOMParser === 'undefined') {
      return html;
    }
    const doc = new DOMParser().parseFromString(html, 'text/html');
    const seen = new Map<string, number>();
    doc.querySelectorAll('h1, h2, h3, h4, h5, h6').forEach((heading) => {
      const base = headingSlug(heading.textContent ?? '');
      if (!base) {
        return;
      }
      const count = seen.get(base) ?? 0;
      seen.set(base, count + 1);
      heading.setAttribute('id', count === 0 ? base : `${base}-${count}`);
    });
    return doc.body.innerHTML;
  }

  /**
   * Follow an in-guide anchor link by scrolling, not by navigating.
   *
   * A bare `href="#..."` would push a fragment onto the SPA's URL (and, in a
   * modal, scroll a container the browser picks rather than the guide pane),
   * so intercept the click and scroll the matching heading into view here.
   * Links to anything else are left alone.
   */
  onGuideClick(event: MouseEvent): void {
    const anchor = (event.target as Element | null)?.closest?.('a');
    const href = anchor?.getAttribute('href') ?? '';
    if (!href.startsWith('#') || href.length < 2) {
      return;
    }
    event.preventDefault();
    const id = decodeURIComponent(href.slice(1));
    const host = event.currentTarget as Element;
    for (const heading of Array.from(host.querySelectorAll('h1, h2, h3, h4, h5, h6'))) {
      if (heading.id === id) {
        // `scrollIntoView` is absent under jsdom; the scroll is cosmetic, so
        // skipping it there costs nothing.
        heading.scrollIntoView?.({ block: 'start' });
        return;
      }
    }
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
