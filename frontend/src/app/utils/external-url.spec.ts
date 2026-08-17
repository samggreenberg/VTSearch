import { afterEach, describe, expect, it, vi } from 'vitest';

import { openBlankTab, openExternalUrl, safeExternalUrl } from './external-url';

describe('safeExternalUrl', () => {
  it.each(['https://example.com/r?ids=a', 'http://localhost:9000/viewer', '  https://x.test  '])(
    'passes %s through, trimmed',
    (url) => {
      expect(safeExternalUrl(url)).toBe(url.trim());
    },
  );

  it.each([
    'javascript:alert(1)',
    'data:text/html,x',
    'file:///etc/passwd',
    '/relative',
    'https://',
    '',
    null,
    undefined,
    42,
  ])('rejects %s', (url) => {
    expect(safeExternalUrl(url)).toBeNull();
  });
});

/** A stand-in for the `Window` handle `window.open` hands back. */
function fakeWindow() {
  return { closed: false, opener: {}, location: { href: '' }, close: vi.fn() };
}

function stubWindowOpen(handle: unknown) {
  return vi.spyOn(window, 'open').mockReturnValue(handle as unknown as Window);
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('openExternalUrl', () => {
  // `noopener` in the features string would make `window.open` return null even
  // when the tab opened (HTML standard, window open steps), leaving the caller
  // unable to tell success from a blocked popup — issue #2898.
  it('opens the URL and severs the opener without asking for noopener', () => {
    const win = fakeWindow();
    const openSpy = stubWindowOpen(win);
    expect(openExternalUrl('https://example.com/r')).toBe(true);
    expect(openSpy).toHaveBeenCalledWith('https://example.com/r', '_blank');
    expect(win.opener).toBeNull();
  });

  it('reports a blocked popup', () => {
    stubWindowOpen(null);
    expect(openExternalUrl('https://example.com/r')).toBe(false);
  });
});

describe('openBlankTab', () => {
  it('claims an empty tab up front and navigates it later', () => {
    const win = fakeWindow();
    const openSpy = stubWindowOpen(win);

    const tab = openBlankTab();
    expect(openSpy).toHaveBeenCalledWith('', '_blank');
    expect(win.opener).toBeNull();
    expect(win.location.href).toBe('');

    expect(tab!.navigate('https://example.com/r')).toBe(true);
    expect(win.location.href).toBe('https://example.com/r');
  });

  it('returns null when even the gesture-time open is refused', () => {
    stubWindowOpen(null);
    expect(openBlankTab()).toBeNull();
  });

  it('refuses to navigate a tab the user already closed', () => {
    const win = fakeWindow();
    stubWindowOpen(win);
    const tab = openBlankTab();
    win.closed = true;
    expect(tab!.navigate('https://example.com/r')).toBe(false);
    expect(win.location.href).toBe('');
  });

  it('closes the tab when there is nothing to show in it', () => {
    const win = fakeWindow();
    stubWindowOpen(win);
    openBlankTab()!.close();
    expect(win.close).toHaveBeenCalled();
  });

  it('leaves an already-closed tab alone', () => {
    const win = fakeWindow();
    stubWindowOpen(win);
    const tab = openBlankTab();
    win.closed = true;
    tab!.close();
    expect(win.close).not.toHaveBeenCalled();
  });
});
