import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ExportModalComponent } from './export-modal.component';
import { settleResource } from '../../../testing/settle-resource';

describe('ExportModalComponent', () => {
  let component: ExportModalComponent;
  let fixture: ComponentFixture<ExportModalComponent>;
  let httpMock: HttpTestingController;

  const mockExporters = [
    { name: 'server_json_file', display_name: 'Server JSON', fields: [] },
    { name: 'hidden', display_name: 'Hidden', hidden_from_picker: true, fields: [] },
  ];
  const mockLabels = {
    labels: [
      { md5: 'a', label: 'good', filename: 'a.wav' },
      { md5: 'b', label: 'bad', filename: 'b.wav' },
      { md5: 'c', label: 'good', filename: 'c.wav', is_correction: true },
    ],
    available_columns: ['label', 'md5', 'filename', 'category', 'extra'],
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ExportModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ExportModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  // The three init reads (dataset status, exporter list, labels) ride
  // `rxResource`, whose loaders run in a root effect rather than during
  // `detectChanges()`; tick to issue the GETs (the labels read also waits for
  // `ngOnInit` to set the input-derived filter), then settle before asserting.
  async function flushInit(): Promise<void> {
    fixture.detectChanges();
    TestBed.tick();
    // The eager status/exporter GETs fire on the tick; the labels GET is
    // released by `ngOnInit`'s signal flip a microtask later, so settle first
    // to let all three become pending before flushing.
    await settleResource();
    httpMock.expectOne('/api/dataset/status').flush({ display_name: 'My Dataset' });
    httpMock.expectOne('/api/exporters').flush(mockExporters);
    httpMock.expectOne((r) => r.url === '/api/labels/export').flush(mockLabels);
    await settleResource();
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
  });

  it('loads the exporter list, filtering hidden entries', async () => {
    await flushInit();
    expect(component.exporters().length).toBe(1);
    expect(component.exporters()[0].name).toBe('server_json_file');
  });

  it('builds columns from available_columns once labels resolve', async () => {
    await flushInit();
    expect(component.labelsLoaded()).toBe(true);
    const keys = component.columns.map((c) => c.key);
    expect(keys).toContain('extra'); // discovered metadata column
    expect(keys).not.toContain('origin'); // always-export keys stay out of the checkboxes
  });

  it('slices the fetched labels by the active category', async () => {
    await flushInit();
    component.labelFilter = 'good';
    expect(component.filteredLabels.length).toBe(2);
    component.labelFilter = 'bad';
    expect(component.filteredLabels.length).toBe(1);
    component.labelFilter = 'corrections';
    expect(component.filteredLabels.length).toBe(1);
  });

  it('reports correction availability', async () => {
    await flushInit();
    expect(component.hasCorrections).toBe(true);
  });

  it('emits exported after a successful export run', async () => {
    await flushInit();
    vi.spyOn(component.exported, 'emit');
    // A fieldless exporter exports immediately.
    component.startExporter(mockExporters[0] as never);
    httpMock.expectOne('/api/exporters/export').flush({ success: true });
    expect(component.exported.emit).toHaveBeenCalled();
  });

  it('emits closed on close', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
