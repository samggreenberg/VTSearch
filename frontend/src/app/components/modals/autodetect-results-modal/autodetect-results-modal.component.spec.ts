import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { AutoDetectResultsModalComponent } from './autodetect-results-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

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
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(AutoDetectResultsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    fixture.componentRef.setInput('data', mockData as any);
  });

  afterEach(() => {
    httpMock.verify();
  });

  // Settle runs ngOnInit (issues GET /api/exporters), flush the response, settle
  // again so the exporter signals repaint. No manual detectChanges.
  async function flushInit(): Promise<void> {
    await settleZoneless(fixture);
    httpMock.expectOne('/api/exporters').flush([
      { name: 'json', label: 'JSON', fields: [] },
      { name: 'csv', label: 'CSV', fields: [{ key: 'path', field_type: 'text', label: 'Path' }] },
    ]);
    await settleZoneless(fixture);
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
  });

  it('should count good and bad hits', async () => {
    await flushInit();
    expect(component.goodCount).toBe(2);
    expect(component.badCount).toBe(1);
  });

  it('should display good hits by default', async () => {
    await flushInit();
    component.exportSides = 'good';
    expect(component.displayHits.length).toBe(2);
  });

  it('should display bad hits when selected', async () => {
    await flushInit();
    component.exportSides = 'bad';
    expect(component.displayHits.length).toBe(1);
  });

  it('should display all hits when both selected', async () => {
    await flushInit();
    component.exportSides = 'both';
    expect(component.displayHits.length).toBe(3);
  });

  it('should format origin with params', async () => {
    await flushInit();
    const hit = {
      md5: 'x',
      origin: { importer: 'folder', params: { path: '/data' } },
    };
    expect(component.formatOrigin(hit)).toBe('folder(/data)');
  });

  it('should format origin without params', async () => {
    await flushInit();
    expect(component.formatOrigin({ md5: 'x', origin: { importer: 'folder' } })).toBe('folder');
    expect(component.formatOrigin({ md5: 'x' })).toBe('');
  });

  // Zoneless canary: exporters/selectedExporter are written from the
  // getExporters() subscribe (not a CD trigger) — as signals they repaint the
  // dropdown.
  it('should load exporters on init and render the dropdown options', async () => {
    await flushInit();
    expect(component.exporters().length).toBe(2);
    expect(component.selectedExporter()).toBe('json');
    const options = fixture.nativeElement.querySelectorAll('.export-row select option');
    expect(options.length).toBe(2);
    expect(options[0].getAttribute('value')).toBe('json');
  });

  it('should update exporter fields when exporter changes', async () => {
    await flushInit();
    component.selectedExporter.set('csv');
    component.onExporterChange();
    expect(component.exporterFields().length).toBe(1);
  });

  it('should emit closed on close', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should render results table', async () => {
    await flushInit();
    const el = fixture.nativeElement as HTMLElement;
    const rows = el.querySelectorAll('.results-table tbody tr');
    expect(rows.length).toBe(2); // good hits by default
  });
});
