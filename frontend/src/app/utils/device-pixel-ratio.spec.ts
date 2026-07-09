import { onDevicePixelRatioChange } from './device-pixel-ratio';

interface StubMediaQueryList {
  matches: boolean;
  media: string;
  onchange: ((e: MediaQueryListEvent) => void) | null;
  addEventListener: (type: string, listener: (e: MediaQueryListEvent) => void) => void;
  removeEventListener: (type: string, listener: (e: MediaQueryListEvent) => void) => void;
  dispatchEvent: (event: Event) => boolean;
  _listeners: Array<(e: MediaQueryListEvent) => void>;
  _fire: () => void;
}

function makeStubMedia(media: string): StubMediaQueryList {
  return {
    matches: true,
    media,
    onchange: null,
    _listeners: [],
    addEventListener(_type, l) {
      this._listeners.push(l);
    },
    removeEventListener(_type, l) {
      this._listeners = this._listeners.filter((x) => x !== l);
    },
    dispatchEvent: () => false,
    _fire() {
      const ev = { matches: this.matches, media: this.media } as MediaQueryListEvent;
      // Copy so a re-subscribe inside the handler doesn't mutate the list we're iterating.
      [...this._listeners].forEach((l) => l(ev));
    },
  };
}

describe('onDevicePixelRatioChange', () => {
  let originalMatchMedia: typeof window.matchMedia;
  let originalDpr: number;
  let stubs: StubMediaQueryList[];
  let queries: string[];

  function setDpr(value: number): void {
    Object.defineProperty(window, 'devicePixelRatio', {
      configurable: true,
      value,
    });
  }

  beforeEach(() => {
    originalMatchMedia = window.matchMedia;
    originalDpr = window.devicePixelRatio;
    stubs = [];
    queries = [];
    setDpr(1);
    (window as unknown as { matchMedia: (q: string) => MediaQueryList }).matchMedia = (q: string) => {
      queries.push(q);
      const stub = makeStubMedia(q);
      stubs.push(stub);
      return stub as unknown as MediaQueryList;
    };
  });

  afterEach(() => {
    (window as unknown as { matchMedia: typeof window.matchMedia }).matchMedia = originalMatchMedia;
    Object.defineProperty(window, 'devicePixelRatio', {
      configurable: true,
      value: originalDpr,
    });
  });

  it('pins the media query to the current devicePixelRatio', () => {
    setDpr(2);
    onDevicePixelRatioChange(() => {});
    expect(queries).toEqual(['(resolution: 2dppx)']);
  });

  it('invokes the callback when the density changes', () => {
    let count = 0;
    onDevicePixelRatioChange(() => count++);
    stubs[0]._fire();
    expect(count).toBe(1);
  });

  it('re-subscribes at the new dpr so successive changes keep firing', () => {
    let count = 0;
    onDevicePixelRatioChange(() => count++);

    // First density change: 1 -> 2. The handler re-pins to the new dpr.
    setDpr(2);
    stubs[0]._fire();
    // Second change: 2 -> 3, delivered on the freshly-subscribed query.
    setDpr(3);
    stubs[stubs.length - 1]._fire();

    expect(count).toBe(2);
    expect(queries).toEqual(['(resolution: 1dppx)', '(resolution: 2dppx)', '(resolution: 3dppx)']);
  });

  it('stops firing after teardown', () => {
    let count = 0;
    const teardown = onDevicePixelRatioChange(() => count++);
    teardown();
    stubs[0]._fire();
    expect(count).toBe(0);
    expect(stubs[0]._listeners).toEqual([]);
  });

  it('returns a no-op teardown when matchMedia is unavailable', () => {
    (window as unknown as { matchMedia: unknown }).matchMedia = undefined;
    const teardown = onDevicePixelRatioChange(() => {});
    expect(() => teardown()).not.toThrow();
  });
});
