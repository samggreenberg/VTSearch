import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { SettingsStateService } from './settings-state.service';
import { settleResource } from '../testing/settle-resource';

describe('SettingsStateService', () => {
  let service: SettingsStateService;
  let httpMock: HttpTestingController;

  const mockSettings = {
    volume: 0.8,
    theme: 'dark' as const,
    show_animations: 'os' as const,
    inclusion: 0.5,
  };

  // `load()` drives the `rxResource`, whose loader runs in an effect rather than
  // synchronously. `TestBed.tick()` runs the loader effect so the request is
  // issued; `settleResource()` then lets the flushed response propagate to the
  // value.
  async function load() {
    service.load();
    TestBed.tick();
    httpMock.expectOne('/api/settings').flush(mockSettings);
    await settleResource();
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
    expect(service.settingsSignal()).toBeNull();
    httpMock.expectNone('/api/settings');
  });

  it('load should fetch and store settings', async () => {
    await load();
    expect(service.settingsSignal()).toEqual(mockSettings);
  });

  it('update should PUT and update cached settings', async () => {
    await load();

    const updated = { ...mockSettings, volume: 0.5 };
    service.update({ volume: 0.5 }).subscribe();
    const req = httpMock.expectOne('/api/settings');
    expect(req.request.method).toBe('PUT');
    req.flush(updated);
    TestBed.tick();
    expect(service.settingsSignal()?.volume).toBe(0.5);
  });

  it('clear should reset settings to null', async () => {
    await load();

    service.clear();
    TestBed.tick();
    expect(service.settingsSignal()).toBeNull();
  });

  it('settingsSignal should expose the loaded settings', async () => {
    await load();
    expect(service.settingsSignal()).toEqual(mockSettings);
  });

  describe('Show Animations -> <html> class mirroring', () => {
    afterEach(() => {
      document.documentElement.classList.remove('animations-off', 'animations-on');
    });

    async function loadWith(mode: 'show' | 'hide' | 'os') {
      service.load();
      TestBed.tick();
      httpMock.expectOne('/api/settings').flush({ ...mockSettings, show_animations: mode });
      await settleResource();
      TestBed.tick();
    }

    it('"hide" adds animations-off and not animations-on', async () => {
      await loadWith('hide');
      const cl = document.documentElement.classList;
      expect(cl.contains('animations-off')).toBe(true);
      expect(cl.contains('animations-on')).toBe(false);
    });

    it('"show" adds animations-on and not animations-off', async () => {
      await loadWith('show');
      const cl = document.documentElement.classList;
      expect(cl.contains('animations-on')).toBe(true);
      expect(cl.contains('animations-off')).toBe(false);
    });

    it('"os" leaves both classes off so the platform preference governs', async () => {
      await loadWith('os');
      const cl = document.documentElement.classList;
      expect(cl.contains('animations-off')).toBe(false);
      expect(cl.contains('animations-on')).toBe(false);
    });
  });
});
