import { ChangeDetectorRef, Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

import {
  SHELL_ICON_SVGS,
  letterGlyphSvg,
  loadExtendedIconSvgs,
  peekExtendedIconSvgs,
} from './icon-svgs';

/**
 * Maps emoji strings (from backend or hardcoded) to an SVG icon type.
 * Falls back to a generic "file" icon for unrecognised values.
 */
const KNOWN_TYPES = new Set([
  'audio', 'image', 'file-text', 'video', 'document',
  'server', 'globe', 'email', 'satellite',
  'folder', 'folder-open', 'upload',
  'graduation', 'arrow-up', 'shuffle', 'elephant', 'cloud',
  'check', 'warning', 'x-circle', 'info', 'file', 'robot',
  'list', 'grid', 'cursor-click', 'cursor-hover',
  'palette', 'sort-descending', 'steering-wheel', 'cloud-upload',
  'thumbs-up', 'thumbs-down', 'factory',
  'house', 'lightning', 'flask', 'cubes', 'database',
  'checkbox-checked', 'search', 'trophy',
]);

function emojiToType(icon: string): string {
  if (!icon) return '';
  // Pass through known SVG type names directly
  if (KNOWN_TYPES.has(icon)) return icon;
  // Normalise: strip variation selectors (U+FE0E, U+FE0F)
  const norm = icon.replace(/[︎️]/g, '');
  const map: Record<string, string> = {
    // Infrastructure
    '🖥': 'server',      // 🖥
    '🌐': 'globe',       // 🌐
    '📧': 'email',       // 📧
    '📡': 'satellite',   // 📡
    // Files & folders
    '📂': 'folder-open', // 📂
    '📁': 'folder',      // 📁
    '📤': 'upload',      // 📤
    // Actions & misc
    '🎓': 'graduation',  // 🎓
    '🏭': 'factory',     // 🏭
    '🧪': 'factory',     // 🧪 — alias for synthetic-data factory
    '🗄': 'database',     // 🗄
    '⬆': 'arrow-up',          // ⬆
    '🔀': 'shuffle',     // 🔀
    '🐘': 'elephant',    // 🐘
    '☁': 'cloud',             // ☁
    '✅': 'check',             // ✅
    // Dialog types
    '⚠': 'warning',            // ⚠
    '❌': 'x-circle',           // ❌
    'ℹ': 'info',               // ℹ
  };
  if (map[norm]) return map[norm];
  // Single capital letter → pass through as letter icon
  if (/^[A-Z]$/.test(norm)) return norm;
  return 'file';
}

/**
 * Cache of sanitised SVG markup keyed by icon name (or
 * ``__letter:<X>`` for letter glyphs).  Shared across every
 * `IconComponent` instance — sanitisation per icon happens once per
 * process and the result is trusted SVG that Angular can blit via
 * ``[innerHTML]``.
 */
const sanitizedCache = new Map<string, SafeHtml>();

@Component({
  selector: 'vt-icon',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (svgHtml) {
      <span class="vt-icon__svg" [style.width.px]="size" [style.height.px]="size" [innerHTML]="svgHtml"></span>
    }
  `,
  styles: [`
    :host {
      display: inline-flex;
      align-items: center;
      vertical-align: middle;
    }
    .vt-icon__svg {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      line-height: 0;
    }
    .vt-icon__svg ::ng-deep svg {
      width: 100%;
      height: 100%;
      display: block;
      vertical-align: middle;
    }
  `],
})
export class IconComponent implements OnChanges {
  @Input() icon = '';
  @Input() size = 16;
  /** Directly set the icon type, bypassing emoji mapping. */
  @Input() type = '';

  protected svgHtml: SafeHtml | null = null;

  constructor(private sanitizer: DomSanitizer, private cdr: ChangeDetectorRef) {}

  ngOnChanges(): void {
    this.renderIcon();
  }

  get iconType(): string {
    if (this.type) return this.type;
    return emojiToType(this.icon);
  }

  /** Non-empty if iconType is a single capital letter (A–Z). */
  get letterChar(): string {
    const t = this.iconType;
    return /^[A-Z]$/.test(t) ? t : '';
  }

  private renderIcon(): void {
    const letter = this.letterChar;
    if (letter) {
      this.svgHtml = this.cached(`__letter:${letter}`, () => letterGlyphSvg(letter));
      return;
    }
    const t = this.iconType;
    if (!t) {
      this.svgHtml = null;
      return;
    }
    if (SHELL_ICON_SVGS[t]) {
      this.svgHtml = this.cached(t, () => SHELL_ICON_SVGS[t]);
      return;
    }
    const peeked = peekExtendedIconSvgs();
    if (peeked) {
      this.svgHtml = this.cached(t, () => peeked[t] ?? SHELL_ICON_SVGS['file']);
      return;
    }
    // Extended set hasn't been loaded yet — defer the render until the
    // dynamic import resolves.  Until then we render nothing rather
    // than flashing the fallback (avoids a visible glyph swap).
    this.svgHtml = null;
    loadExtendedIconSvgs().then((map) => {
      this.svgHtml = this.cached(t, () => map[t] ?? SHELL_ICON_SVGS['file']);
      this.cdr.markForCheck();
    });
  }

  private cached(key: string, build: () => string): SafeHtml {
    const hit = sanitizedCache.get(key);
    if (hit !== undefined) return hit;
    const safe = this.sanitizer.bypassSecurityTrustHtml(build());
    sanitizedCache.set(key, safe);
    return safe;
  }
}
