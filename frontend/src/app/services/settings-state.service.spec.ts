import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { SettingsStateService } from './settings-state.service';

describe('SettingsStateService', () => {
  let service: SettingsStateService;
  let httpMock: HttpTestingController;

  const mockSettings = {
    volume: 0.8,
    theme: 'dark' as const,
    show_animations: true,
    view_mode_left: { audio: 'grid' as const, image: 'grid' as const },
    view_mode_right: { audio: 'list' as const, image: 'list' as const },
    inclusion: 0.5,
  };

  // `load()` drives an `rxResource`, whose loader runs in an effect rather than
  // synchronously. `TestBed.tick()` flushes pending effects so the resource
  // issues its request / propagates its value; we tick after `load()` to make
  // the HTTP request observable, and again after `flush()` to settle the value.
  function load() {
    service.load();
    TestBed.tick();
    httpMock.expectOne('/api/settings').flush(mockSettings);
    TestBed.tick();
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(SettingsStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should start with null settings and not fetch until load()', () => {
    TestBed.tick();
    expect(service.settings).toBeNull();
    httpMock.expectNone('/api/settings');
  });

  it('load should fetch and store settings', () => {
    load();
    expect(service.settings).toEqual(mockSettings);
  });

  it('update should PUT and update cached settings', () => {
    load();

    const updated = { ...mockSettings, volume: 0.5 };
    service.update({ volume: 0.5 }).subscribe();
    const req = httpMock.expectOne('/api/settings');
    expect(req.request.method).toBe('PUT');
    req.flush(updated);
    TestBed.tick();
    expect(service.settings?.volume).toBe(0.5);
  });

  it('clear should reset settings to null', () => {
    load();

    service.clear();
    TestBed.tick();
    expect(service.settings).toBeNull();
  });

  it('settings$ should emit on load', () => new Promise<void>((done) => {
    const emissions: (typeof mockSettings | null)[] = [];
    service.settings$.subscribe((s) => emissions.push(s));

    load();

    setTimeout(() => {
      expect(emissions.length).toBeGreaterThanOrEqual(2);
      expect(emissions[emissions.length - 1]).toEqual(mockSettings);
      done();
    });
  }));
});
