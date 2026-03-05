import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { AutoDetectResultsModalComponent } from './autodetect-results-modal.component';

describe('AutoDetectResultsModalComponent', () => {
  let component: AutoDetectResultsModalComponent;
  let fixture: ComponentFixture<AutoDetectResultsModalComponent>;
  let httpMock: HttpTestingController;

  const mockData = {
    media_type: 'audio',
    detectors_run: '2',
    results: {
      detector1: {
        hits: [
          { md5: 'abc123', filename: 'song.wav', origin_name: 'Song 1' },
          { md5: 'def456', filename: 'track.wav', origin_name: 'Track 2' },
        ],
        negative_hits: [
          { md5: 'ghi789', filename: 'noise.wav', origin_name: 'Noise 1' },
        ],
      },
    },
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutoDetectResultsModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(AutoDetectResultsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    component.data = mockData as any;
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushInit(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/exporters').flush([
      { name: 'json', label: 'JSON', fields: [] },
      { name: 'csv', label: 'CSV', fields: [{ key: 'path', field_type: 'text', label: 'Path' }] },
    ]);
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should count good and bad hits', () => {
    flushInit();
    expect(component.goodCount).toBe(2);
    expect(component.badCount).toBe(1);
  });

  it('should display good hits by default', () => {
    flushInit();
    component.exportSides = 'good';
    expect(component.displayHits.length).toBe(2);
  });

  it('should display bad hits when selected', () => {
    flushInit();
    component.exportSides = 'bad';
    expect(component.displayHits.length).toBe(1);
  });

  it('should display all hits when both selected', () => {
    flushInit();
    component.exportSides = 'both';
    expect(component.displayHits.length).toBe(3);
  });

  it('should format origin with params', () => {
    flushInit();
    const hit = {
      md5: 'x',
      origin: { importer: 'folder', params: { path: '/data' } },
    };
    expect(component.formatOrigin(hit)).toBe('folder(/data)');
  });

  it('should format origin without params', () => {
    flushInit();
    expect(component.formatOrigin({ md5: 'x', origin: { importer: 'folder' } })).toBe('folder');
    expect(component.formatOrigin({ md5: 'x' })).toBe('');
  });

  it('should load exporters on init', () => {
    flushInit();
    expect(component.exporters.length).toBe(2);
    expect(component.selectedExporter).toBe('json');
  });

  it('should update exporter fields when exporter changes', () => {
    flushInit();
    component.selectedExporter = 'csv';
    component.onExporterChange();
    expect(component.exporterFields.length).toBe(1);
  });

  it('should emit closed on close', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should render results table', () => {
    flushInit();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const rows = el.querySelectorAll('.results-table tbody tr');
    expect(rows.length).toBe(2); // good hits by default
  });
});
