import {
  afterNextRender,
  ChangeDetectionStrategy,
  Component,
  computed,
  ElementRef,
  inject,
  Injector,
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
 * Directory the user docs are served from. Paths inside a doc are written
 * relative to the doc's repo location under ``docs/user/`` (e.g.
 * ``assets/foo.png`` in ``USER_GUIDE.md``, ``../assets/foo.png`` in
 * ``howto/bar.md``). The Angular build copies every ``docs/user/**.md`` and
 * the ``assets/`` folder under ``/assets/docs``, so a relative path must be
 * resolved against the current doc, then against this base, to load in the app.
 */
const GUIDE_ASSET_BASE = 'assets/docs/';

/** The doc the "User guide" tab opens on, relative to {@link GUIDE_ASSET_BASE}. */
const GUIDE_DOC = 'USER_GUIDE.md';

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

/**
 * Resolve *href*, written in the doc at *from* (both relative to
 * ``docs/user/``), to a path relative to ``docs/user/``.
 *
 * Returns ``null`` for anything the Help panel cannot serve itself: an
 * absolute URL, a root-relative path, a bare fragment, or a relative path
 * that climbs out of ``docs/user/`` (``../SETUP.md`` from the guide). Those
 * are left to the browser. Any ``#fragment`` or ``?query`` is dropped; the
 * caller reads the fragment off *href* separately.
 */
export function resolveDocPath(from: string, href: string): string | null {
  if (!href || href.startsWith('#') || href.startsWith('/') || ABSOLUTE_SRC_RE.test(href) || /^[a-z]+:/i.test(href)) {
    return null;
  }
  const path = href.split(/[?#]/, 1)[0];
  if (!path) {
    return null;
  }
  const segments = from.split('/').slice(0, -1);
  for (const part of path.split('/')) {
    if (part === '..') {
      if (!segments.length) {
        return null;
      }
      segments.pop();
    } else if (part !== '.' && part !== '') {
      segments.push(part);
    }
  }
  return segments.join('/');
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
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly injector = inject(Injector);

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
  /**
   * The doc the guide pane is showing, relative to ``docs/user/``. The User
   * guide opens on USER_GUIDE.md; a link to another doc in ``docs/user/``
   * (the how-to pages under ``howto/``) opens it in the same pane.
   */
  readonly docPath = signal<string>(GUIDE_DOC);
  /** Docs the pane came from, most recent last, for the Back button. */
  private readonly history = signal<string[]>([]);
  readonly canGoBack = computed(() => this.history().length > 0);
  /** Raw markdown per doc, cached so a theme switch or Back re-renders without re-fetching. */
  private readonly rawDocs = new Map<string, string>();

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
            { keys: ['↓'], description: 'Back to the item you just voted on (again to step further back)' },
            { keys: ['↑'], description: 'Forward to the next unlabeled item' },
          ],
        },
        {
          title: 'Playback',
          shortcuts: [
            { keys: ['Space'], description: 'Play / pause audio or video' },
            { keys: ['Shift', '↑'], description: 'Volume up' },
            { keys: ['Shift', '↓'], description: 'Volume down' },
          ],
        },
        {
          title: 'Image viewer',
          shortcuts: [
            { keys: ['+'], description: 'Zoom in' },
            { keys: ['-'], description: 'Zoom out' },
            { keys: ['['], description: 'Rotate left' },
            { keys: [']'], description: 'Rotate right' },
            { keys: ['Double-click'], description: 'Zoom in on the spot you clicked (again at 5x returns to fit)' },
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
      const raw = this.rawDocs.get(this.docPath());
      if (raw !== undefined) {
        this.renderGuide(raw);
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
    this.showDoc(GUIDE_DOC);
  }

  /**
   * Show the doc at *path* (relative to ``docs/user/``) in the guide pane,
   * scrolled to *fragment* if given, else to the top. Fetches it on first use.
   */
  private showDoc(path: string, fragment = ''): void {
    const cached = this.rawDocs.get(path);
    if (cached !== undefined) {
      this.docPath.set(path);
      this.renderGuide(cached);
      this.scrollGuideTo(fragment);
      return;
    }
    this.http.get(GUIDE_ASSET_BASE + path, { responseType: 'text' }).subscribe({
      next: (md) => {
        this.rawDocs.set(path, md);
        this.docPath.set(path);
        this.guideError.set(null);
        this.renderGuide(md);
        this.scrollGuideTo(fragment);
      },
      error: (err) => {
        const what = path === GUIDE_DOC ? 'user guide' : path;
        this.guideError.set(`Failed to load ${what}: ${err?.message ?? err}`);
      },
    });
  }

  /** Return to the doc the pane showed before the last followed link. */
  back(): void {
    const stack = this.history();
    if (!stack.length) {
      return;
    }
    this.history.set(stack.slice(0, -1));
    this.guideError.set(null);
    this.showDoc(stack[stack.length - 1]);
  }

  /**
   * Once the new doc has rendered, bring *fragment*'s heading (or, with no
   * fragment, the top of the doc) into view in the guide pane.
   */
  private scrollGuideTo(fragment: string): void {
    afterNextRender(
      () => {
        const root = this.host.nativeElement;
        if (fragment) {
          const id = decodeURIComponent(fragment);
          const heading = Array.from(root.querySelectorAll('.guide-body h1, .guide-body h2, .guide-body h3, .guide-body h4, .guide-body h5, .guide-body h6')).find((h) => h.id === id);
          // `scrollIntoView` is absent under jsdom; the scroll is cosmetic.
          heading?.scrollIntoView?.({ block: 'start' });
          return;
        }
        const pane = root.querySelector('.guide') as HTMLElement | null;
        if (pane) {
          pane.scrollTop = 0;
        }
      },
      { injector: this.injector },
    );
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
   * Follow a link inside the guide pane without navigating the app.
   *
   * A link to another doc under ``docs/user/`` (``howto/find-and-fix.md``,
   * ``../USER_GUIDE.md#autopilot-the-guided-workflow``) opens that doc in this
   * pane, with Back to return. A bare `href="#..."` would push a fragment onto
   * the SPA's URL (and, in a modal, scroll a container the browser picks
   * rather than the guide pane), so intercept the click and scroll the
   * matching heading into view here. Links to anything else are left alone.
   */
  onGuideClick(event: MouseEvent): void {
    const anchor = (event.target as Element | null)?.closest?.('a');
    const href = anchor?.getAttribute('href') ?? '';
    const target = resolveDocPath(this.docPath(), href);
    if (target !== null && target.endsWith('.md')) {
      // A link to another user doc (a how-to page, or back to the guide):
      // open it in this pane, the way the same link reads on GitHub.
      event.preventDefault();
      const hash = href.indexOf('#');
      const fragment = hash >= 0 ? href.slice(hash + 1) : '';
      if (target === this.docPath()) {
        this.scrollGuideTo(fragment);
        return;
      }
      this.history.set([...this.history(), this.docPath()]);
      this.showDoc(target, fragment);
      return;
    }
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
   * - Resolve relative ``src`` paths against the current doc, then the
   *   docs' served directory.
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
        src = GUIDE_ASSET_BASE + (resolveDocPath(this.docPath(), src) ?? src);
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
