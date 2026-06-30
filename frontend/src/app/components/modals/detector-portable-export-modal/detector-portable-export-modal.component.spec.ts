import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { DetectorPortableExportModalComponent } from './detector-portable-export-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';

describe('DetectorPortableExportModalComponent', () => {
  let component: DetectorPortableExportModalComponent;
  let fixture: ComponentFixture<DetectorPortableExportModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DetectorPortableExportModalComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DetectorPortableExportModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    fixture.componentRef.setInput('detectorId', 'det-123');
    fixture.componentRef.setInput('detectorName', 'My Detector');
    // jsdom doesn't implement the object-URL APIs the download path uses.
    URL.createObjectURL = vi.fn(() => 'blob:mock');
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  it('posts to the portable-bundle endpoint as a blob and marks done on success', () => {
    component.download();
    const req = httpMock.expectOne('/api/detectors/det-123/portable-bundle');
    expect(req.request.method).toBe('POST');
    expect(req.request.responseType).toBe('blob');
    req.flush(new Blob(['zip-bytes'], { type: 'application/zip' }));

    expect(component.done()).toBe(true);
    expect(component.error()).toBe('');
    expect(component.downloading()).toBe(false);
    expect(URL.createObjectURL).toHaveBeenCalled();
  });

  it('surfaces the server error message from the blob body', async () => {
    component.download();
    const req = httpMock.expectOne('/api/detectors/det-123/portable-bundle');
    req.flush(new Blob([JSON.stringify({ message: 'No medias loaded' })], { type: 'application/json' }), {
      status: 400,
      statusText: 'Bad Request',
    });

    // readError reads the Blob asynchronously; let the microtask settle.
    await new Promise((resolve) => setTimeout(resolve));
    expect(component.error()).toBe('No medias loaded');
    expect(component.done()).toBe(false);
  });

  it('emits closed on cancel', () => {
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
