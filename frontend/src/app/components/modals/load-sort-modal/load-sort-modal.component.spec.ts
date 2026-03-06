import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LoadSortModalComponent } from './load-sort-modal.component';

describe('LoadSortModalComponent', () => {
  let component: LoadSortModalComponent;
  let fixture: ComponentFixture<LoadSortModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LoadSortModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(LoadSortModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushInit(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/detector/server-files').flush({
      files: [{ name: 'det1', path: '/data/det1.json', size_bytes: 100 }, { name: 'det2', path: '/data/det2.json', size_bytes: 200 }],
    });
    httpMock.expectOne('/api/server-media-files').flush({
      files: [{ name: 'example', filename: 'example.wav', path: '/data/example.wav', size_bytes: 1000 }, { name: 'sample', filename: 'sample.wav', path: '/data/sample.wav', size_bytes: 2000 }],
    });
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should load server files on init', () => {
    flushInit();
    expect(component.serverDetectors.length).toBe(2);
    expect(component.serverDetectors[0].name).toBe('det1');
    expect(component.serverMediaFiles.length).toBe(2);
    expect(component.serverMediaFiles[0].filename).toBe('example.wav');
    expect(component.loading).toBeFalse();
  });

  it('should load server detector and emit', () => {
    flushInit();
    spyOn(component.detectorLoaded, 'emit');
    spyOn(component.closed, 'emit');

    component.loadServerDetector('det1.json');
    httpMock.expectOne('/api/detector/server-files/det1.json').flush({ weights: [1, 2, 3] });

    expect(component.detectorLoaded.emit).toHaveBeenCalledWith({ weights: [1, 2, 3] });
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should handle detector load error', () => {
    flushInit();
    component.loadServerDetector('bad.json');
    httpMock
      .expectOne('/api/detector/server-files/bad.json')
      .flush({}, { status: 404, statusText: 'Not Found' });
    expect(component.error).toBe('Failed to load detector');
  });

  it('should load server media and emit', () => {
    flushInit();
    spyOn(component.exampleSortStarted, 'emit');
    spyOn(component.closed, 'emit');

    component.loadServerMedia('example.wav');
    httpMock.expectOne('/api/example-sort-server').flush({ results: [], threshold: 0.5 });

    expect(component.exampleSortStarted.emit).toHaveBeenCalled();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should render file lists', () => {
    flushInit();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const items = el.querySelectorAll('.file-item');
    expect(items.length).toBe(4); // 2 detectors + 2 media files
  });

  it('should emit closed on close', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
