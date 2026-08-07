/**
 * Guards for the `open_url` an exporter can hand back for the browser to open.
 *
 * The server already runs every such URL through `validate_browser_url`, so
 * this is the second half of a belt-and-braces pair: the check that matters is
 * the scheme, and it costs one regex to make the frontend independently unable
 * to navigate to a `javascript:` or `data:` URL — a class of bug that only ever
 * shows up once someone's plugin is already in the wild.
 *
 * Lives here rather than on one component because two surfaces act on the key:
 * the Export modal (a labelset export the user ran) and the Auto-Detect Results
 * modal (an Auto-Find auto-export that ran for them).
 */

/** Narrow an exporter-supplied `open_url` to something safe for `window.open`, or `null`. */
export function safeExternalUrl(url: unknown): string | null {
  if (typeof url !== 'string') return null;
  const trimmed = url.trim();
  return /^https?:\/\/\S+$/i.test(trimmed) ? trimmed : null;
}

/**
 * Open *url* in a new tab, reporting whether the tab actually opened.
 *
 * `noopener` severs the new page's `window.opener` handle so a third-party site
 * can't navigate the VTSearch tab out from under the user (reverse tabnabbing).
 * A `null` return means the popup blocker ate it, which is the normal outcome
 * when the call happens after an async response rather than inside a click
 * handler.
 */
export function openExternalUrl(url: string): boolean {
  return window.open(url, '_blank', 'noopener') !== null;
}
