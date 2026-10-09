import { signal, WritableSignal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';

import { HttpTestingController } from '@angular/common/http/testing';
import { Router } from '@angular/router';
import { AutoDetectResultsModalComponent } from './autodetect-results-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { BrowseSubsetPrepService } from '../../../services/browse-subset-prep.service';
import { ContextSwitchService } from '../../../services/context-switch.service';
import { ToastService } from '../../../services/toast.service';

describe('AutoDetectResultsModalComponent', () => {
  let component: AutoDetectResultsModalComponent;
  let fixture: ComponentFixture<AutoDetectResultsModalComponent>;
  let httpMock: HttpTestingController;
  let toast: { success: ReturnType<typeof vi.fn> };

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
      providers: [
        ...provideZoneless(),
        ...provideHttpTesting(),
        { provide: ToastService, useFactory: () => (toast = { success: vi.fn() }) },
      ],
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

  it("names the detectors that ran as the Goods' centroid, with what they owe (#4643)", async () => {
    const quota = (tier: string, goods: number, bads: number) => ({
      tier,
      n_good: 3 - goods,
      n_bad: 4 - bads,
      goods_owed: goods,
      bads_owed: bads,
      good_quota: 3,
      bad_quota: 4,
    });
    fixture.componentRef.setInput('data', {
      ...mockData,
      results: {
        under: { detector_name: 'Under', hits: [], label_quota: quota('centroid', 1, 3) },
        trained: { detector_name: 'Trained', hits: [], label_quota: quota('trained', 0, 0) },
      },
    } as any);
    await flushInit();
    expect(component.centroidDetectors).toEqual(['Under (1 more Good and 3 more Bads)']);
    expect(fixture.nativeElement.querySelector('.centroid-note')?.textContent).toContain("Goods' centroid");
  });

  it('says nothing about the centroid when every detector trained', async () => {
    await flushInit();
    expect(component.centroidDetectors).toEqual([]);
    expect(fixture.nativeElement.querySelector('.centroid-note')).toBeNull();
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

  // An AutoFind auto-export can format the run into a third-party site's URL
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

  // The dialog used to render an exporter picker with no button to run it.
  describe('Export', () => {
    it('renders an Export button for the chosen exporter', async () => {
      await flushInit();
      const btn = Array.from(fixture.nativeElement.querySelectorAll('.export-actions button')).find(
        (b) => (b as HTMLElement).textContent?.trim() === 'Export',
      ) as HTMLButtonElement | undefined;
      expect(btn).toBeTruthy();
      expect(btn!.disabled).toBe(false);
    });

    it('exports exactly the listed rows: the chosen side moves into hits', async () => {
      await flushInit();
      component.exportSides = 'bad';
      const bad = component.exportPayload().results['detector1'];
      expect(bad.hits.map((h) => h.md5)).toEqual(['ghi789']);
      expect(bad.negative_hits).toEqual([]);
      expect(bad.total_hits).toBe(1);

      component.exportSides = 'both';
      const both = component.exportPayload().results['detector1'];
      expect(both.hits.map((h) => h.label)).toEqual(['good', 'good', 'bad']);
    });

    it('sends the run to the chosen exporter as find_results and confirms', async () => {
      await flushInit();
      component.selectedExporter.set('csv');
      component.onExporterChange();
      component.exportFieldValues['path'] = '/tmp/out.csv';
      component.exportResults();

      const req = httpMock.expectOne('/api/exporters/export');
      expect(req.request.body.exporter_name).toBe('csv');
      expect(req.request.body.payload_kind).toBe('find_results');
      expect(req.request.body.field_values).toEqual({ path: '/tmp/out.csv' });
      expect(req.request.body.results.results.detector1.hits.length).toBe(2);
      expect(component.exporting()).toBe(true);
      req.flush({ success: true, message: 'Saved.' });

      expect(component.exporting()).toBe(false);
      expect(toast.success).toHaveBeenCalledWith(expect.objectContaining({ message: 'Exported 2 results to csv' }));
    });
  });
});

/**
 * The dialog's Browse button (#4615): the listed rows' media, once each, are
 * mapped on their own once the run's dataset is active, and Browse's Back
 * returns to these results.
 */
describe('AutoDetectResultsModalComponent Browse', () => {
  let component: AutoDetectResultsModalComponent;
  let fixture: ComponentFixture<AutoDetectResultsModalComponent>;
  let switched: Subject<void>;
  let contextSwitch: { applyActivePair: ReturnType<typeof vi.fn> };
  let browsePrep: { preparing: WritableSignal<boolean>; start: ReturnType<typeof vi.fn>; cancel: ReturnType<typeof vi.fn> };

  const run = {
    run_id: '_autofind_7',
    dataset_id: 'ds1',
    dataset_name: 'Birds',
    media_type: 'audio',
    detectors_run: 2,
    results: {
      owl: {
        hits: [
          { id: 1, md5: 'a' },
          { id: 2, md5: 'b' },
        ],
        negative_hits: [{ id: 3, md5: 'c' }],
      },
      // A second detector that also called item 2 Good: one item on the map.
      wren: { hits: [{ id: 2, md5: 'b' }], negative_hits: [] },
    },
  };

  beforeEach(async () => {
    switched = new Subject<void>();
    contextSwitch = { applyActivePair: vi.fn(() => switched) };
    browsePrep = {
      preparing: signal(false),
      start: vi.fn(),
      cancel: vi.fn(),
    };
    await TestBed.configureTestingModule({
      imports: [AutoDetectResultsModalComponent],
      providers: [
        ...provideZoneless(),
        ...provideHttpTesting(),
        { provide: ToastService, useValue: { success: vi.fn() } },
        { provide: ContextSwitchService, useValue: contextSwitch },
        { provide: BrowseSubsetPrepService, useValue: browsePrep },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AutoDetectResultsModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('data', run as any);
  });

  it('browses the listed rows, once each, following the Good / Bad / Both filter', () => {
    expect(component.browseIds).toEqual([1, 2]);
    component.exportSides = 'bad';
    expect(component.browseIds).toEqual([3]);
    component.exportSides = 'both';
    expect(component.browseIds).toEqual([1, 2, 3]);
  });

  it('makes the run dataset active, then builds the map with a Back to these results', () => {
    const closed = vi.fn();
    component.closed.subscribe(closed);

    component.browse();
    expect(contextSwitch.applyActivePair).toHaveBeenCalledWith('ds1', '');
    expect(component.browseStarting()).toBe(true);
    expect(browsePrep.start).not.toHaveBeenCalled();

    switched.next();
    expect(component.browseStarting()).toBe(false);
    expect(browsePrep.start).toHaveBeenCalledWith('ds1', [1, 2], { kind: 'results', runId: '_autofind_7' }, expect.any(Function));

    // The dialog closes only once the map is ready and Browse opens.
    expect(closed).not.toHaveBeenCalled();
    browsePrep.start.mock.calls[0][3]();
    expect(closed).toHaveBeenCalledTimes(1);
  });

  it('starts nothing when the dataset fails to load', () => {
    component.browse();
    switched.complete();
    expect(browsePrep.start).not.toHaveBeenCalled();
    expect(component.browseStarting()).toBe(false);
  });

  it('is unavailable for results that name no run or dataset', () => {
    fixture.componentRef.setInput('data', { ...run, run_id: undefined } as any);
    expect(component.browseBlocker).not.toBe('');
    component.browse();
    expect(contextSwitch.applyActivePair).not.toHaveBeenCalled();
  });

  it('cancels a map still building when the dialog is closed', () => {
    browsePrep.preparing.set(true);
    component.close();
    expect(browsePrep.cancel).toHaveBeenCalled();
  });
});

/**
 * While the dialog is open, the server is asked to lay out the run's Good
 * results for Browse whenever it has nothing else to do (#4683): asked again,
 * easing off, while it answers `busy`, and never again once it has started.
 */
describe('AutoDetectResultsModalComponent Browse warm-up', () => {
  let component: AutoDetectResultsModalComponent;
  let fixture: ComponentFixture<AutoDetectResultsModalComponent>;
  let httpMock: HttpTestingController;
  let router: Router;
  let toast: { success: ReturnType<typeof vi.fn>; error: ReturnType<typeof vi.fn>; warning: ReturnType<typeof vi.fn> };

  const PREP_URL = '/api/autofind/runs/_autofind_7/browse-prep';
  const run = {
    run_id: '_autofind_7',
    dataset_id: 'ds1',
    dataset_name: 'Birds',
    media_type: 'audio',
    detectors_run: 1,
    results: { owl: { hits: [{ id: 1, md5: 'a' }], negative_hits: [{ id: 3, md5: 'c' }] } },
  };

  beforeEach(async () => {
    vi.useFakeTimers();
    toast = { success: vi.fn(), error: vi.fn(), warning: vi.fn() };
    await TestBed.configureTestingModule({
      imports: [AutoDetectResultsModalComponent],
      providers: [
        ...provideZoneless(),
        ...provideHttpTesting(),
        { provide: ToastService, useValue: toast },
        { provide: ContextSwitchService, useValue: { applyActivePair: vi.fn(() => new Subject<void>()) } },
        { provide: BrowseSubsetPrepService, useValue: { preparing: signal(false), start: vi.fn(), cancel: vi.fn() } },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AutoDetectResultsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    router = TestBed.inject(Router);
  });

  afterEach(() => {
    fixture.destroy();
    httpMock.verify();
    vi.useRealTimers();
  });

  /** Open the dialog on *data*, and answer its exporter list. */
  function open(data: unknown): void {
    fixture.componentRef.setInput('data', data as any);
    fixture.detectChanges();
    httpMock.match('/api/exporters').forEach((req) => req.flush([]));
  }

  function answer(status: string): void {
    const req = httpMock.expectOne(PREP_URL);
    expect(req.request.method).toBe('POST');
    req.flush({ status });
  }

  it('asks once the dialog opens, and not again once the server has started', async () => {
    open(run);
    answer('building');
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_SLOW_MS * 2);
    httpMock.expectNone(PREP_URL);
  });

  it('asks again while the server is busy, until it is free', async () => {
    open(run);
    answer('busy');
    httpMock.expectNone(PREP_URL);
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_FAST_MS);
    answer('busy');
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_FAST_MS);
    answer('ready');
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_SLOW_MS * 2);
    httpMock.expectNone(PREP_URL);
  });

  it('stops asking when Browse is pressed: its own build takes over', async () => {
    open(run);
    answer('busy');
    component.browse();
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_SLOW_MS * 2);
    httpMock.expectNone(PREP_URL);
  });

  it('stops asking when the dialog closes', async () => {
    open(run);
    answer('busy');
    fixture.destroy();
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_SLOW_MS * 2);
    httpMock.expectNone(PREP_URL);
  });

  it('asks nothing for results that name no run', async () => {
    open({ ...run, run_id: undefined });
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_SLOW_MS);
    httpMock.expectNone((req) => req.url.endsWith('/browse-prep'));
  });

  it('gives up quietly when the run is gone', async () => {
    open(run);
    httpMock.expectOne(PREP_URL).flush({ message: 'gone' }, { status: 404, statusText: 'Not Found' });
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_SLOW_MS * 2);
    httpMock.expectNone(PREP_URL);
    expect(toast.error).not.toHaveBeenCalled();
  });

  it("waits while this tab shows a subset map of the run's dataset", async () => {
    const url = vi.spyOn(router, 'url', 'get').mockReturnValue('/browse/ds1?subset=1&from=test');
    open(run);
    httpMock.expectNone(PREP_URL);
    url.mockReturnValue('/dashboard');
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_SLOW_MS);
    answer('building');
  });

  it('starts over for another run swapped into the open dialog', async () => {
    open(run);
    answer('busy');
    fixture.componentRef.setInput('data', { ...run, run_id: '_autofind_8' } as any);
    fixture.detectChanges();
    httpMock.expectOne('/api/autofind/runs/_autofind_8/browse-prep').flush({ status: 'building' });
    await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.WARM_SLOW_MS * 2);
    httpMock.expectNone(PREP_URL);
  });
});
