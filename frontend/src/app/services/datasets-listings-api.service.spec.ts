import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DatasetsListingsApiService } from './datasets-listings-api.service';

describe('DatasetsListingsApiService', () => {
  let service: DatasetsListingsApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DatasetsListingsApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('getMediaTypes should GET', () => {
    service.getMediaTypes().subscribe((data) => expect(data.media_types.length).toBeGreaterThan(0));
    const req = httpMock.expectOne('/api/media-types');
    expect(req.request.method).toBe('GET');
    req.flush({ media_types: [{ type_id: 'audio', name: 'Audio' }] });
  });
});
