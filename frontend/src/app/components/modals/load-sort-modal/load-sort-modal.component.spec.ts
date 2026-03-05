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
    httpMock.expectOne('/api/detector/server-files').flush({ files: ['det1.json', 'det2.json'] });
    httpMock.expectOne('/api/server-media-files').flush({ files: ['example.wav', 'sample.wav'] });
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should load server files on init', () => {
    flushInit();
    expect(component.serverDetectors).toEqual(['det1.json', 'det2.json']);
    expect(component.serverMediaFiles).toEqual(['example.wav', 'sample.wav']);
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
