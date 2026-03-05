import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { TrainableModelsApiService } from './trainable-models-api.service';

describe('TrainableModelsApiService', () => {
  let service: TrainableModelsApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(TrainableModelsApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('list should GET', () => {
    service.list().subscribe(data => expect(data.models).toBeDefined());
    const req = httpMock.expectOne('/api/trainable-models');
    expect(req.request.method).toBe('GET');
    req.flush({ models: [] });
  });

  it('create should POST', () => {
    service.create({ name: 'm1' }).subscribe();
    const req = httpMock.expectOne('/api/trainable-models');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('get should GET by name', () => {
    service.get('m1').subscribe();
    const req = httpMock.expectOne('/api/trainable-models/m1');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('delete should DELETE by name', () => {
    service.delete('m1').subscribe();
    const req = httpMock.expectOne('/api/trainable-models/m1');
    expect(req.request.method).toBe('DELETE');
    req.flush({});
  });

  it('rename should PUT', () => {
    service.rename('m1', 'm2').subscribe();
    const req = httpMock.expectOne('/api/trainable-models/m1/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'm2' });
    req.flush({});
  });

  it('getRegistry should GET', () => {
    service.getRegistry().subscribe(data => expect(data.models).toBeDefined());
    const req = httpMock.expectOne('/api/models/registry');
    expect(req.request.method).toBe('GET');
    req.flush({ models: [] });
  });
});
