import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { SettingsStateService } from './settings-state.service';

describe('SettingsStateService', () => {
  let service: SettingsStateService;
  let httpMock: HttpTestingController;

  const mockSettings = {
    volume: 0.8,
    theme: 'dark',
    swipe_animation: true,
    view_mode_left: { audio: 'grid', image: 'grid' },
    view_mode_right: { audio: 'list', image: 'list' },
    inclusion: 0.5,
  };

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

  it('should start with null settings', () => {
    expect(service.settings).toBeNull();
  });

  it('load should fetch and store settings', () => {
    service.load();
    const req = httpMock.expectOne('/api/settings');
    req.flush(mockSettings);
    expect(service.settings).toEqual(mockSettings);
  });

  it('update should PUT and update cached settings', () => {
    service.load();
    httpMock.expectOne('/api/settings').flush(mockSettings);

    const updated = { ...mockSettings, volume: 0.5 };
    service.update({ volume: 0.5 }).subscribe();
    const req = httpMock.expectOne('/api/settings');
    expect(req.request.method).toBe('PUT');
    req.flush(updated);
    expect(service.settings?.volume).toBe(0.5);
  });

  it('clear should reset settings to null', () => {
    service.load();
    httpMock.expectOne('/api/settings').flush(mockSettings);

    service.clear();
    expect(service.settings).toBeNull();
  });

  it('settings$ should emit on load', (done) => {
    const emissions: any[] = [];
    service.settings$.subscribe((s) => emissions.push(s));

    service.load();
    httpMock.expectOne('/api/settings').flush(mockSettings);

    setTimeout(() => {
      expect(emissions.length).toBeGreaterThanOrEqual(2);
      expect(emissions[emissions.length - 1]).toEqual(mockSettings);
      done();
    });
  });
});
