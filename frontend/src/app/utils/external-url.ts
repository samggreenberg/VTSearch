/**
 * Guards for the `open_url` an exporter can hand back for the browser to open.
 *
 * The server already runs every such URL through `validate_browser_url`, so
 * `safeExternalUrl` is the second half of a belt-and-braces pair: the check that
 * matters is the scheme, and it costs one regex to make the frontend
 * independently unable to navigate to a `javascript:` or `data:` URL — a class
 * of bug that only ever shows up once someone's plugin is already in the wild.
 *
 * Lives here rather than on one component because two surfaces act on the key:
 * the Export modal (a labelset export the user ran) and the Auto-Detect Results
 * modal (an Auto-Find auto-export that ran for them).
 *
 * ## Why the opener is severed by hand, not by `noopener`
 *
 * The obvious spelling — `window.open(url, '_blank', 'noopener')` — is
 * unusable here, because the HTML standard's window open steps end with "if
 * noopener is true [...] then return null" regardless of whether the tab
 * opened. That makes the return value carry no information: the caller cannot
 * tell a popup the blocker ate from one sitting in front of the user, so it can
 * neither report what happened nor decide whether to offer a fallback (issue
 * #2898). Dropping `noopener` and assigning `opener = null` on the returned
 * handle buys back that signal at no real cost: the assignment runs
 * synchronously in the same task as the `open()` call, before the new document
 * can run any script of its own, and it permanently disowns the opener, so
 * reverse tabnabbing (`window.opener.location = evil`) stays impossible.
 */

/** Narrow an exporter-supplied `open_url` to something safe for `window.open`, or `null`. */
export function safeExternalUrl(url: unknown): string | null {
  if (typeof url !== 'string') return null;
  const trimmed = url.trim();
  return /^https?:\/\/\S+$/i.test(trimmed) ? trimmed : null;
}

/** Disown *win* so the page it ends up showing has no handle on this one. */
function severOpener(win: Window): void {
  try {
    win.opener = null;
  } catch {
    // `opener` is cross-origin-settable per spec, but a browser that refuses
    // is not a reason to abandon a tab the user asked for.
  }
}

/**
 * A tab opened up front, waiting for a URL that an in-flight request will
 * carry. See {@link openBlankTab} for why it exists.
 */
export interface PendingTab {
  /** Point the tab at *url*. False if it is gone (the user closed it). */
  navigate(url: string): boolean;
  /** Close the tab — the request came back with no URL to show it. */
  close(): void;
}

/**
 * Open an empty tab *now*, to be navigated when a pending request resolves.
 *
 * A popup is allowed while the click that triggered it still counts as user
 * activation; a `window.open()` deferred to an HTTP response callback is
 * precisely the shape popup blockers exist to stop, and gets swallowed with no
 * error (issue #2898). So a flow that knows a URL is coming — an exporter
 * declaring `opens_url` — claims the tab inside the click handler and points it
 * at the URL once the response lands.
 *
 * Returns `null` when even the gesture-time open was refused, which is a real
 * answer rather than the ambiguity `noopener` leaves behind: the caller can
 * fall back to an explicit "Open" button.
 */
export function openBlankTab(): PendingTab | null {
  const win = window.open('', '_blank');
  if (!win) return null;
  severOpener(win);
  return {
    navigate(url: string): boolean {
      try {
        if (win.closed) return false;
        win.location.href = url;
        return true;
      } catch {
        return false;
      }
    },
    close(): void {
      try {
        if (!win.closed) win.close();
      } catch {
        // Already gone; nothing to clean up.
      }
    },
  };
}

/**
 * Open *url* in a new tab, reporting whether the tab actually opened.
 *
 * Use this from a real click handler (a toast's "Open" button, a link-shaped
 * control). From an async callback, prefer {@link openBlankTab} at gesture time
 * — a `false` here usually means the popup blocker ate it.
 */
export function openExternalUrl(url: string): boolean {
  const win = window.open(url, '_blank');
  if (!win) return false;
  severOpener(win);
  return true;
}
