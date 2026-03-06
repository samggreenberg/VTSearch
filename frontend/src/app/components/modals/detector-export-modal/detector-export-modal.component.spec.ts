import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { DetectorExportModalComponent } from './detector-export-modal.component';

describe('DetectorExportModalComponent', () => {
  let component: DetectorExportModalComponent;
  let fixture: ComponentFixture<DetectorExportModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DetectorExportModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DetectorExportModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    component.detectorName = 'test-detector';
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushInit(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/exporters').flush([{ name: 'gui', label: 'Browser' }]);
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should set title from detector name', () => {
    flushInit();
    expect(component.title).toBe('Export "test-detector"');
  });

  it('should set generic title when no name', () => {
    component.detectorName = '';
    flushInit();
    expect(component.title).toBe('Export Detector');
  });

  it('should export to server', () => {
    flushInit();
    spyOn(component.exported, 'emit');
    component.exportServer();
    httpMock.expectOne('/api/autorun-detectors/test-detector/export-server').flush({});
    expect(component.exported.emit).toHaveBeenCalled();
  });

  it('should handle server export error', () => {
    flushInit();
    component.exportServer();
    httpMock
      .expectOne('/api/autorun-detectors/test-detector/export-server')
      .flush({}, { status: 500, statusText: 'Error' });
    expect(component.error).toBe('Failed to save to server');
  });

  it('should emit closed on close', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
