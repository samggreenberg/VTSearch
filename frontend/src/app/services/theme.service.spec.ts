import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ThemeService } from './theme.service';

describe('ThemeService', () => {
  let service: ThemeService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ThemeService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
    document.documentElement.removeAttribute('data-theme');
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
});
