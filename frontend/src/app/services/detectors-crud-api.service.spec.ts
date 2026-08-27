import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DetectorsCrudApiService } from './detectors-crud-api.service';
import { ActiveContextService } from './active-context.service';

describe('DetectorsCrudApiService', () => {
  let service: DetectorsCrudApiService;
  let httpMock: HttpTestingController;

  // Stub ActiveContextService so the URL builders below are deterministic and
  // don't pull the real context graph into the test.
  const activeContextStub = {
    mediaUrl: (path: string) => `${path}?dataset_id=ds1`,
  };

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: ActiveContextService, useValue: activeContextStub },
      ],
    });
    service = TestBed.inject(DetectorsCrudApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('list should GET /api/detectors', () => {
    service.list().subscribe((r) => expect(r.detectors).toBeDefined());
    const req = httpMock.expectOne('/api/detectors');
    expect(req.request.method).toBe('GET');
    req.flush({ detectors: [] });
  });

  it('create should POST /api/detectors with the request body', () => {
    service.create({ name: 'd1', media_type: 'image' }).subscribe();
    const req = httpMock.expectOne('/api/detectors');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ name: 'd1', media_type: 'image' });
    req.flush({});
  });

  it('get should GET /api/detectors/{name}', () => {
    service.get('my det').subscribe();
    const req = httpMock.expectOne('/api/detectors/my%20det');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('delete should DELETE /api/detectors/{name}', () => {
    service.delete('d1').subscribe();
    const req = httpMock.expectOne('/api/detectors/d1');
    expect(req.request.method).toBe('DELETE');
    req.flush({});
  });

  it('rename should PUT /api/detectors/{name}/rename with new_name', () => {
    service.rename('old', 'new').subscribe();
    const req = httpMock.expectOne('/api/detectors/old/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ new_name: 'new' });
    req.flush({});
  });

  it('setExamples should PUT /api/detectors/{name}/examples with examples', () => {
    const examples = [{ path: 'a.png' }] as never;
    service.setExamples('d1', examples).subscribe();
    const req = httpMock.expectOne('/api/detectors/d1/examples');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ examples });
    req.flush({});
  });

  it('saveLabels should POST /api/detectors/{name}/labels', () => {
    service.saveLabels('d1').subscribe();
    const req = httpMock.expectOne('/api/detectors/d1/labels');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('getLabelsDetail should GET /api/detectors/{name}/labels-detail', () => {
    service.getLabelsDetail('d1').subscribe();
    const req = httpMock.expectOne('/api/detectors/d1/labels-detail');
    expect(req.request.method).toBe('GET');
    req.flush({});
  });

  it('voteLabelElement should POST the vote target', () => {
    service.voteLabelElement('d1', 'el-7', 'good').subscribe();
    const req = httpMock.expectOne('/api/detectors/d1/labels/el-7/vote');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ target: 'good' });
    req.flush({});
  });

  it('combine should POST /api/detectors/combine with names + policy', () => {
    service.combine(['a', 'b'], 'merged', 'rename').subscribe();
    const req = httpMock.expectOne('/api/detectors/combine');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ names: ['a', 'b'], new_name: 'merged', conflict_policy: 'rename' });
    req.flush({});
  });

  it('combine defaults the conflict policy to drop', () => {
    service.combine(['a', 'b'], 'merged').subscribe();
    const req = httpMock.expectOne('/api/detectors/combine');
    expect(req.request.body).toEqual({ names: ['a', 'b'], new_name: 'merged', conflict_policy: 'drop' });
    req.flush({});
  });

  it('labelPreviewUrl routes through ActiveContextService.mediaUrl with encoded segments', () => {
    const url = service.labelPreviewUrl('my det', 'el/7');
    expect(url).toBe('/api/detectors/my%20det/labels/el%2F7/preview?dataset_id=ds1');
  });

  it('labelThumbnailUrl routes through ActiveContextService.mediaUrl with encoded segments', () => {
    const url = service.labelThumbnailUrl('d1', 'el-7');
    expect(url).toBe('/api/detectors/d1/labels/el-7/thumbnail?dataset_id=ds1');
  });
});
