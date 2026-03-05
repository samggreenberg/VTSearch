import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { SettingsApiService } from './settings-api.service';

describe('SettingsApiService', () => {
  let service: SettingsApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(SettingsApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('getSettings should GET', () => {
    service.getSettings().subscribe(data => expect(data.volume).toBe(0.8));
    const req = httpMock.expectOne('/api/settings');
    expect(req.request.method).toBe('GET');
    req.flush({ volume: 0.8, autorun_processors: [] });
  });

  it('updateSettings should PUT', () => {
    service.updateSettings({ volume: 0.5 }).subscribe();
    const req = httpMock.expectOne('/api/settings');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ volume: 0.5 });
    req.flush({ volume: 0.5 });
  });

  it('getDefaults should GET', () => {
    service.getDefaults().subscribe(data => expect(data.volume).toBeDefined());
    const req = httpMock.expectOne('/api/settings/defaults');
    expect(req.request.method).toBe('GET');
    req.flush({ volume: 1.0 });
  });

  it('getAutorunProcessors should GET', () => {
    service.getAutorunProcessors().subscribe(data => expect(data.autorun_processors).toBeDefined());
    const req = httpMock.expectOne('/api/settings/autorun-processors');
    expect(req.request.method).toBe('GET');
    req.flush({ autorun_processors: [] });
  });

  it('deleteAutorunProcessor should DELETE', () => {
    service.deleteAutorunProcessor('proc1').subscribe();
    const req = httpMock.expectOne('/api/settings/autorun-processors/proc1');
    expect(req.request.method).toBe('DELETE');
    req.flush({});
  });
});
