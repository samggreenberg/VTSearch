import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { SettingsStateService } from './settings-state.service';
import { settleResource } from '../testing/settle-resource';
import { provideHttpTesting } from '../testing/test-providers';

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
      providers: [...provideHttpTesting()],
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

  /**
   * `perMediaType` is the shared replacement for the read/coerce/merge-write
   * dance that used to be hand-rolled at every per-media-type settings
   * consumer. Two properties are load-bearing and pinned here: the value is a
   * real `computed` (so a template binding on it repaints under zoneless), and
   * the setter MERGES — dropping a sibling media type's entry would silently
   * destroy a user preference on every media-type switch.
   */
  describe('perMediaType', () => {
    const withDicts = {
      ...mockSettings,
      grid_icon_size_right: { audio: 'L', image: 'S' },
      browse_mouse_zooms_per_level: { audio: 9, image: 2 },
    };

    async function loadDicts() {
      service.load();
      TestBed.tick();
      httpMock.expectOne('/api/settings').flush(withDicts);
      await settleResource();
    }

    it('resolves the active media type\'s entry, and re-resolves on a switch', async () => {
      const mediaType = signal('audio');
      const pref = service.perMediaType<string>('grid_icon_size_right', mediaType, {
        fallback: 'M',
      });
      await loadDicts();

      expect(pref.value()).toBe('L');
      // The whole point of keying on a signal: no mirror to re-hydrate.
      mediaType.set('image');
      expect(pref.value()).toBe('S');
    });

    it('falls back for an unset media type, and for an empty one', async () => {
      const mediaType = signal('video');
      const pref = service.perMediaType<string>('grid_icon_size_right', mediaType, {
        fallback: 'M',
      });
      await loadDicts();

      expect(pref.value()).toBe('M');
      mediaType.set('');
      expect(pref.value()).toBe('M');
    });

    it('falls back before settings have loaded', () => {
      const pref = service.perMediaType<string>('grid_icon_size_right', signal('audio'), {
        fallback: 'M',
      });
      expect(pref.value()).toBe('M');
      expect(pref.dict()).toEqual({});
    });

    it('uses coerce to reject an out-of-range stored value', async () => {
      // A hand-edited settings file (or an older server) can hold anything;
      // `coerce` is where the clamp/enum check that used to sit inline lives.
      const pref = service.perMediaType<number>(
        'browse_mouse_zooms_per_level',
        signal('audio'),
        {
          fallback: 2,
          coerce: (raw) => (typeof raw === 'number' && raw >= 1 && raw <= 3 ? raw : undefined),
        },
      );
      await loadDicts();
      // Stored value is 9 — out of range, so the fallback wins.
      expect(pref.value()).toBe(2);
    });

    it('set() merges, preserving every other media type\'s entry', async () => {
      const pref = service.perMediaType<string>('grid_icon_size_right', signal('audio'), {
        fallback: 'M',
      });
      await loadDicts();

      pref.set('XL')?.subscribe();
      const req = httpMock.expectOne('/api/settings');
      expect(req.request.method).toBe('PUT');
      // `image: 'S'` MUST survive. If this regressed, switching media type
      // would silently reset the sibling type's thumbnail size.
      expect(req.request.body).toEqual({
        grid_icon_size_right: { audio: 'XL', image: 'S' },
      });
      req.flush({ ...withDicts, grid_icon_size_right: { audio: 'XL', image: 'S' } });
      TestBed.tick();
      expect(pref.value()).toBe('XL');
    });

    it('set() is a no-op with no active media type (nothing to key the write on)', async () => {
      const pref = service.perMediaType<string>('grid_icon_size_right', signal(''), {
        fallback: 'M',
      });
      await loadDicts();

      expect(pref.set('XL')).toBeNull();
      httpMock.expectNone('/api/settings');
    });

    it('does not keep a stale mirror when the key goes absent server-side', async () => {
      const mediaType = signal('audio');
      const pref = service.perMediaType<string>('grid_icon_size_right', mediaType, {
        fallback: 'M',
      });
      await loadDicts();
      expect(pref.value()).toBe('L');

      // Every old consumer hydrated its shadow dict behind an
      // `if (dict && typeof dict === 'object')` guard, so a key that vanished
      // (a settings reset) left the last-seen copy in place forever. Reading
      // through a computed cannot do that.
      service.update({ volume: 0.4 }).subscribe();
      httpMock.expectOne('/api/settings').flush({ ...mockSettings, volume: 0.4 });
      TestBed.tick();
      expect(pref.dict()).toEqual({});
      expect(pref.value()).toBe('M');
    });

    it('ignores a non-dict value stored under the key', async () => {
      const pref = service.perMediaType<string>('grid_icon_size_right', signal('audio'), {
        fallback: 'M',
      });
      service.load();
      TestBed.tick();
      httpMock
        .expectOne('/api/settings')
        .flush({ ...mockSettings, grid_icon_size_right: 'L' as unknown });
      await settleResource();

      expect(pref.dict()).toEqual({});
      expect(pref.value()).toBe('M');
    });
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
