/**
 * Shell icon SVGs — kept in the initial bundle.
 *
 * Only the icons that components in the eager app shell (today: the
 * `DialogHostComponent`, which renders ``info`` / ``warning`` /
 * ``x-circle`` / ``check`` for confirm/prompt dialogs) plus the
 * ``file`` fallback live here. Every other icon is lazy-loaded by
 * `IconComponent` from `icon-svgs-extended.ts` on first request.
 *
 * SVGs are stored as plain strings without ``width`` / ``height``
 * attributes — `IconComponent` sets the size on its wrapper span so
 * the SVG (which has a 24x24 ``viewBox``) scales via CSS.
 */
export const SHELL_ICON_SVGS: Record<string, string> = {
  check:
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
  warning:
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  'x-circle':
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
  info:
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
  file:
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>',
};

let extendedCache: Record<string, string> | null = null;
let extendedPromise: Promise<Record<string, string>> | null = null;

/**
 * Lazy-load the extended icon table.  Returns a cached map on
 * subsequent calls so the dynamic import fires at most once per
 * process.
 */
export function loadExtendedIconSvgs(): Promise<Record<string, string>> {
  if (extendedCache) return Promise.resolve(extendedCache);
  if (!extendedPromise) {
    extendedPromise = import('./icon-svgs-extended').then((m) => {
      extendedCache = m.EXTENDED_ICON_SVGS;
      return extendedCache;
    });
  }
  return extendedPromise;
}

/** Returns the extended map only if it has already been loaded. */
export function peekExtendedIconSvgs(): Record<string, string> | null {
  return extendedCache;
}

/**
 * Build the SVG for a single capital letter glyph (A–Z), used by
 * `IconComponent` when the icon resolves to a single uppercase letter.
 */
export function letterGlyphSvg(letter: string): string {
  return (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<rect x="2" y="2" width="20" height="20" rx="5" ry="5"/>' +
    '<text x="12" y="17" text-anchor="middle" font-size="14" font-weight="600" ' +
    'font-family="system-ui, sans-serif" fill="currentColor" stroke="none">' +
    letter +
    '</text></svg>'
  );
}
