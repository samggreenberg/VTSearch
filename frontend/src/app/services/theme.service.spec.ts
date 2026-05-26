import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ThemeService } from './theme.service';

interface StubMediaQueryList {
  matches: boolean;
  media: string;
  onchange: ((e: MediaQueryListEvent) => void) | null;
  addListener: (listener: (e: MediaQueryListEvent) => void) => void;
  removeListener: (listener: (e: MediaQueryListEvent) => void) => void;
  addEventListener: (type: string, listener: (e: MediaQueryListEvent) => void) => void;
  removeEventListener: (type: string, listener: (e: MediaQueryListEvent) => void) => void;
  dispatchEvent: (event: Event) => boolean;
  _listeners: Array<(e: MediaQueryListEvent) => void>;
  _fireChange: (matches: boolean) => void;
}

function makeStubMedia(initialMatches: boolean): StubMediaQueryList {
  const stub: StubMediaQueryList = {
    matches: initialMatches,
    media: '',
    onchange: null,
    _listeners: [],
    addListener(l) {
      this._listeners.push(l);
    },
    removeListener(l) {
      this._listeners = this._listeners.filter((x) => x !== l);
    },
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
  return stub;
}

describe('ThemeService', () => {
  let service: ThemeService;
  let httpMock: HttpTestingController;
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
    // Default: OS prefers dark.
    installStub(false);
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ThemeService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
    document.documentElement.removeAttribute('data-theme');
    (window as unknown as { matchMedia: typeof window.matchMedia }).matchMedia = originalMatchMedia;
  });

  it('should be created with system default', () => {
    expect(service).toBeTruthy();
    expect(service.currentTheme).toBe('system');
  });

  it('setTheme should update data-theme attribute', () => {
    service.setTheme('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(service.currentTheme).toBe('light');
    httpMock.expectOne('/api/settings').flush({});
  });

  it('setTheme should persist via SettingsApi', () => {
    service.setTheme('highviz');
    const req = httpMock.expectOne('/api/settings');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ theme: 'highviz' });
    req.flush({});
  });

  it('loadFromSettings should apply backend theme', () => {
    service.loadFromSettings();
    httpMock.expectOne('/api/settings').flush({ volume: 1.0, theme: 'light' });
    expect(service.currentTheme).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('theme$ should emit on change', () => {
    const emitted: string[] = [];
    service.theme$.subscribe((t) => emitted.push(t));
    service.setTheme('highviz');
    httpMock.expectOne('/api/settings').flush({});
    expect(emitted).toEqual(['system', 'highviz']);
  });

  it('system theme resolves to dark when OS prefers dark', () => {
    installStub(false);
    service.setTheme('system');
    expect(service.currentTheme).toBe('system');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    httpMock.expectOne('/api/settings').flush({});
  });

  it('system theme resolves to light when OS prefers light', () => {
    installStub(true);
    service.setTheme('system');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    httpMock.expectOne('/api/settings').flush({});
  });

  it('loadFromSettings defaults to system when backend has no theme', () => {
    installStub(true);
    service.loadFromSettings();
    httpMock.expectOne('/api/settings').flush({ volume: 1.0 });
    expect(service.currentTheme).toBe('system');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('system theme tracks live OS preference changes', () => {
    const media = installStub(false);
    service.setTheme('system');
    httpMock.expectOne('/api/settings').flush({});
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    media._fireChange(true);
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    media._fireChange(false);
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('non-system theme does not track OS changes', () => {
    const media = installStub(false);
    service.setTheme('dark');
    httpMock.expectOne('/api/settings').flush({});
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    media._fireChange(true);
    // Still dark - user picked dark explicitly.
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('switching from system to explicit theme detaches the OS listener', () => {
    const media = installStub(false);
    service.setTheme('system');
    httpMock.expectOne('/api/settings').flush({});
    expect(media._listeners.length).toBe(1);
    service.setTheme('light');
    httpMock.expectOne('/api/settings').flush({});
    expect(media._listeners.length).toBe(0);
  });

  it('detectOsTheme falls back to dark when matchMedia is unavailable', () => {
    (window as unknown as { matchMedia: unknown }).matchMedia = undefined;
    expect(service.detectOsTheme()).toBe('dark');
  });
});
