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

  // Drain the asynchrony rxResource introduces: its loader is promise-based, so
  // the stream value commits on a microtask; we then flush effects so the
  // computed settings signal and the settings$ bridge update.
  async function settle() {
    await new Promise<void>((resolve) => setTimeout(resolve));
    TestBed.tick();
  }

  // `load()` drives the `rxResource`, whose loader runs in an effect rather than
  // synchronously. `TestBed.tick()` runs the loader effect so the request is
  // issued; `settle()` then lets the flushed response propagate to the value.
  async function load() {
    service.load();
    TestBed.tick();
    httpMock.expectOne('/api/settings').flush(mockSettings);
    await settle();
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

  it('load should fetch and store settings', async () => {
    await load();
    expect(service.settings).toEqual(mockSettings);
  });

  it('update should PUT and update cached settings', async () => {
    await load();

    const updated = { ...mockSettings, volume: 0.5 };
    service.update({ volume: 0.5 }).subscribe();
    const req = httpMock.expectOne('/api/settings');
    expect(req.request.method).toBe('PUT');
    req.flush(updated);
    TestBed.tick();
    expect(service.settings?.volume).toBe(0.5);
  });

  it('clear should reset settings to null', async () => {
    await load();

    service.clear();
    TestBed.tick();
    expect(service.settings).toBeNull();
  });

  it('settings$ should replay the loaded settings to subscribers', async () => {
    const emissions: unknown[] = [];
    const sub = service.settings$.subscribe((s) => emissions.push(s));

    await load();

    sub.unsubscribe();
    expect(emissions.length).toBeGreaterThanOrEqual(1);
    expect(emissions[emissions.length - 1]).toEqual(mockSettings);
  });
});
