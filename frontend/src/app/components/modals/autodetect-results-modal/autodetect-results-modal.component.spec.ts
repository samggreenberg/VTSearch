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
      { name: 'json', label: 'JSON', fields: [], supported_payloads: ['find_results'] },
      {
        name: 'csv',
        label: 'CSV',
        fields: [{ key: 'path', field_type: 'text', label: 'Path' }],
        supported_payloads: ['find_results', 'labelset'],
      },
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

  // An Auto-Find auto-export can format the run into a third-party site's URL
  // rather than delivering it anywhere (#2898). It's offered as a click, not
  // opened on arrival: these results land from an async response, where an
  // unprompted window.open() is what popup blockers exist to stop.
  describe('auto-export open_url', () => {
    function withAutoExport(auto_export: Record<string, unknown>): void {
      fixture.componentRef.setInput('data', { ...mockData, auto_export } as any);
    }

    it('offers an Open button for an openable URL', async () => {
      withAutoExport({ exporter: 'open_url', success: true, open_url: 'https://example.com/r?ids=a' });
      await flushInit();
      const btn = fixture.nativeElement.querySelector('.auto-export-open') as HTMLButtonElement;
      expect(btn).toBeTruthy();
      expect(btn.getAttribute('title')).toBe('https://example.com/r?ids=a');
    });

    it('opens the URL in a new tab when clicked, never handing over the opener', async () => {
      withAutoExport({ exporter: 'open_url', success: true, open_url: 'https://example.com/r' });
      await flushInit();
      // `noopener` in the features string would make `window.open` return null
      // even on success, so the opener is severed on the handle instead (#2898).
      const win = { closed: false, opener: {}, location: { href: '' }, close: vi.fn() };
      const open = vi.spyOn(window, 'open').mockReturnValue(win as unknown as Window);
      (fixture.nativeElement.querySelector('.auto-export-open') as HTMLButtonElement).click();
      expect(open).toHaveBeenCalledWith('https://example.com/r', '_blank');
      expect(win.opener).toBeNull();
      open.mockRestore();
    });

    it('does not open anything on arrival', async () => {
      const open = vi.spyOn(window, 'open').mockReturnValue(null);
      withAutoExport({ exporter: 'open_url', success: true, open_url: 'https://example.com/r' });
      await flushInit();
      expect(open).not.toHaveBeenCalled();
      open.mockRestore();
    });

    it('ignores a URL the browser must not navigate to', async () => {
      withAutoExport({ exporter: 'evil', success: true, open_url: 'javascript:alert(1)' });
      await flushInit();
      expect(component.autoExportUrl()).toBeNull();
      expect(fixture.nativeElement.querySelector('.auto-export-open')).toBeNull();
    });

    it('offers nothing for a failed export or a plain delivery', async () => {
      withAutoExport({ exporter: 'open_url', success: false, open_url: 'https://example.com/r' });
      await flushInit();
      expect(component.autoExportUrl()).toBeNull();

      withAutoExport({ exporter: 'server_json_file', success: true, message: 'Wrote it.' });
      await settleZoneless(fixture);
      expect(component.autoExportUrl()).toBeNull();
      expect(fixture.nativeElement.querySelector('.auto-export-open')).toBeNull();
    });
  });
});
