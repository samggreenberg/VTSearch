import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ThemeService } from './theme.service';

describe('ThemeService', () => {
  let service: ThemeService;
  let httpMock: HttpTestingController;
  let originalMatchMedia: typeof window.matchMedia;

  function stubMatchMedia(prefersLight: boolean): void {
    (window as unknown as { matchMedia: (q: string) => MediaQueryList }).matchMedia = (
      query: string,
    ) => {
      const matches = query.includes('light') ? prefersLight : !prefersLight;
      return {
        matches,
        media: query,
        onchange: null,
        addListener: () => undefined,
        removeListener: () => undefined,
        addEventListener: () => undefined,
        removeEventListener: () => undefined,
        dispatchEvent: () => false,
      } as MediaQueryList;
    };
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ThemeService);
    httpMock = TestBed.inject(HttpTestingController);
    originalMatchMedia = window.matchMedia;
  });

  afterEach(() => {
    httpMock.verify();
    document.documentElement.removeAttribute('data-theme');
    (window as unknown as { matchMedia: typeof window.matchMedia }).matchMedia = originalMatchMedia;
  });

  it('should be created with dark default', () => {
    expect(service).toBeTruthy();
    expect(service.currentTheme).toBe('dark');
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
    const req = httpMock.expectOne('/api/settings');
    req.flush({ volume: 1.0, theme: 'light', autorun_processors: [] });
    expect(service.currentTheme).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('theme$ should emit on change', () => {
    const emitted: string[] = [];
    service.theme$.subscribe(t => emitted.push(t));
    service.setTheme('highviz');
    httpMock.expectOne('/api/settings').flush({});
    expect(emitted).toEqual(['dark', 'highviz']);
  });

  it('loadFromSettings detects light OS preference and persists it when theme is null', () => {
    stubMatchMedia(true);
    service.loadFromSettings();
    httpMock.expectOne('/api/settings').flush({ volume: 1.0, theme: null });
    const persist = httpMock.expectOne('/api/settings');
    expect(persist.request.method).toBe('PUT');
    expect(persist.request.body).toEqual({ theme: 'light' });
    persist.flush({});
    expect(service.currentTheme).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('loadFromSettings detects dark OS preference and persists it when theme is missing', () => {
    stubMatchMedia(false);
    service.loadFromSettings();
    httpMock.expectOne('/api/settings').flush({ volume: 1.0 });
    const persist = httpMock.expectOne('/api/settings');
    expect(persist.request.body).toEqual({ theme: 'dark' });
    persist.flush({});
    expect(service.currentTheme).toBe('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('detectOsTheme falls back to dark when matchMedia is unavailable', () => {
    (window as unknown as { matchMedia: unknown }).matchMedia = undefined;
    expect(service.detectOsTheme()).toBe('dark');
  });
});
