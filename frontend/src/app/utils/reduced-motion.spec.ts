import {
  ANIMATIONS_OFF_CLASS,
  browserPrefersReducedMotion,
  onBrowserReducedMotionChange,
  prefersReducedMotion,
} from './reduced-motion';

interface StubMediaQueryList {
  matches: boolean;
  media: string;
  onchange: ((e: MediaQueryListEvent) => void) | null;
  addEventListener: (type: string, listener: (e: MediaQueryListEvent) => void) => void;
  removeEventListener: (type: string, listener: (e: MediaQueryListEvent) => void) => void;
  dispatchEvent: (event: Event) => boolean;
  _listeners: Array<(e: MediaQueryListEvent) => void>;
  _fireChange: (matches: boolean) => void;
}

function makeStubMedia(initialMatches: boolean): StubMediaQueryList {
  return {
    matches: initialMatches,
    media: '(prefers-reduced-motion: reduce)',
    onchange: null,
    _listeners: [],
    addEventListener(_type, l) {
      this._listeners.push(l);
    },
    removeEventListener(_type, l) {
      this._listeners = this._listeners.filter((x) => x !== l);
    },
    dispatchEvent: () => false,
    _fireChange(matches: boolean) {
      this.matches = matches;
      const ev = { matches, media: this.media } as MediaQueryListEvent;
      this._listeners.forEach((l) => l(ev));
    },
  };
}

describe('reduced-motion utils', () => {
  let originalMatchMedia: typeof window.matchMedia;
  let stub: StubMediaQueryList;

  function installStub(initialMatches: boolean): StubMediaQueryList {
    stub = makeStubMedia(initialMatches);
    (window as unknown as { matchMedia: (q: string) => MediaQueryList }).matchMedia = () =>
      stub as unknown as MediaQueryList;
    return stub;
  }

  beforeEach(() => {
    originalMatchMedia = window.matchMedia;
    document.documentElement.classList.remove(ANIMATIONS_OFF_CLASS);
  });

  afterEach(() => {
    (window as unknown as { matchMedia: typeof window.matchMedia }).matchMedia = originalMatchMedia;
    document.documentElement.classList.remove(ANIMATIONS_OFF_CLASS);
  });

  describe('browserPrefersReducedMotion', () => {
    it('returns true when the OS/browser query matches', () => {
      installStub(true);
      expect(browserPrefersReducedMotion()).toBe(true);
    });

    it('returns false when the OS/browser query does not match', () => {
      installStub(false);
      expect(browserPrefersReducedMotion()).toBe(false);
    });

    it('ignores the app-level animations-off class (reports OS state only)', () => {
      installStub(false);
      document.documentElement.classList.add(ANIMATIONS_OFF_CLASS);
      expect(browserPrefersReducedMotion()).toBe(false);
    });

    it('falls back to false when matchMedia is unavailable', () => {
      (window as unknown as { matchMedia: unknown }).matchMedia = undefined;
      expect(browserPrefersReducedMotion()).toBe(false);
    });
  });

  describe('onBrowserReducedMotionChange', () => {
    it('invokes the callback when the OS preference flips and stops after teardown', () => {
      const s = installStub(false);
      const seen: boolean[] = [];
      const teardown = onBrowserReducedMotionChange((v) => seen.push(v));

      s._fireChange(true);
      s._fireChange(false);
      expect(seen).toEqual([true, false]);

      teardown();
      s._fireChange(true);
      expect(seen).toEqual([true, false]);
    });

    it('returns a no-op teardown when matchMedia is unavailable', () => {
      (window as unknown as { matchMedia: unknown }).matchMedia = undefined;
      const teardown = onBrowserReducedMotionChange(() => {});
      expect(() => teardown()).not.toThrow();
    });
  });

  describe('prefersReducedMotion', () => {
    it('is true when the animations-off class is present, regardless of OS query', () => {
      installStub(false);
      document.documentElement.classList.add(ANIMATIONS_OFF_CLASS);
      expect(prefersReducedMotion()).toBe(true);
    });

    it('is true when the OS query matches even without the class', () => {
      installStub(true);
      expect(prefersReducedMotion()).toBe(true);
    });

    it('is false when neither the class nor the OS query is set', () => {
      installStub(false);
      expect(prefersReducedMotion()).toBe(false);
    });
  });
});
