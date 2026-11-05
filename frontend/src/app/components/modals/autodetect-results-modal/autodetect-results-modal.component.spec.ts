import { signal, WritableSignal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';

import { HttpTestingController } from '@angular/common/http/testing';
import { Router } from '@angular/router';
import { AUTOFIND_TAB, AutoDetectResultsModalComponent } from './autodetect-results-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { BrowseSubsetPrepService } from '../../../services/browse-subset-prep.service';
import { ContextSwitchService } from '../../../services/context-switch.service';
import { MediaTypeCapabilityService } from '../../../services/media-type-capability.service';
import { DATASET_OVERRIDE } from '../../../interceptors/active-context.interceptor';
import { SettingsStateService } from '../../../services/settings-state.service';
import { ToastService } from '../../../services/toast.service';
import { makeSettingsStateStub } from '../../../testing/mocks';
import type { AppSettings } from '../../../generated/api-client/models/app-settings';

/** The media-type registry, answered without a request: audio and image have
 *  thumbnails, and each type's icon is its own name. */
function mediaTypeCapsStub(): Partial<MediaTypeCapabilityService> {
  return {
    ensureLoaded: vi.fn(),
    usesThumbnails: (type: string) => type === 'audio' || type === 'image',
    iconFor: (type: string) => type,
  };
}

describe('AutoDetectResultsModalComponent', () => {
  let component: AutoDetectResultsModalComponent;
  let fixture: ComponentFixture<AutoDetectResultsModalComponent>;
  let httpMock: HttpTestingController;
  let toast: { success: ReturnType<typeof vi.fn>; error: ReturnType<typeof vi.fn> };
  let settings: WritableSignal<AppSettings | null>;

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
    // The column order is kept across sessions; a drag in one test must not
    // reorder the next.
    localStorage.clear();
    const settingsState = makeSettingsStateStub();
    settings = settingsState.settings;
    await TestBed.configureTestingModule({
      imports: [AutoDetectResultsModalComponent],
      providers: [
        ...provideZoneless(),
        ...provideHttpTesting(),
        { provide: MediaTypeCapabilityService, useFactory: mediaTypeCapsStub },
        { provide: ToastService, useFactory: () => (toast = { success: vi.fn(), error: vi.fn() }) },
        { provide: SettingsStateService, useValue: settingsState.stub },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AutoDetectResultsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    fixture.componentRef.setInput('data', mockData as any);
  });

  afterEach(() => {
    httpMock.verify();
    localStorage.clear();
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

  describe('the summary\'s Media Type (#5106)', () => {
    const summary = () => (fixture.nativeElement.querySelector('.results-summary') as HTMLElement).textContent ?? '';

    it('names the run\'s media type on a server open to every type', async () => {
      await flushInit();
      expect(summary().replace(/\s+/g, ' ')).toContain('Media type: Audio');
    });

    it('shows the type with its icon, as the Dashboard does (#5464)', async () => {
      await flushInit();
      const icon = fixture.nativeElement.querySelector('.results-summary .type-cell vt-icon');
      expect(icon).toBeTruthy();
    });

    it('leaves it out on a server locked to that type, where it could be nothing else', async () => {
      settings.set({ solo_media_type: 'audio' } as AppSettings);
      await flushInit();
      expect(summary()).not.toContain('Media type');
      expect(summary().replace(/\s+/g, ' ')).toContain('Detector: detector1');
    });

    it('still names a type the lock does not account for', async () => {
      settings.set({ solo_media_type: 'image' } as AppSettings);
      await flushInit();
      expect(summary().replace(/\s+/g, ' ')).toContain('Media type: Audio');
    });
  });

  // The detectors that ran, by name, and the one label style for the
  // summary, the Threshold and Columns (#5584).
  describe('the summary names the detectors that ran', () => {
    const line = () => fixture.nativeElement.querySelector('.results-detectors') as HTMLElement;
    const text = () => line().textContent?.replace(/\s+/g, ' ').trim() ?? '';
    const withDetectors = (...names: string[]) =>
      fixture.componentRef.setInput('data', {
        ...mockData,
        detectors_run: names.length,
        results: Object.fromEntries(names.map((name, i) => [`d${i}`, { detector_name: name, hits: [] }])),
      } as any);

    it('lists a few by name, singular for one, with every name on hover', async () => {
      withDetectors('Cats');
      await flushInit();
      expect(text()).toBe('Detector: Cats');
      withDetectors('Cats', 'Dogs', 'Owls');
      await settleZoneless(fixture);
      expect(text()).toBe('Detectors: Cats, Dogs, Owls');
      expect(line().title).toBe('Cats\nDogs\nOwls');
    });

    it('gives a count past three, the names on hover', async () => {
      withDetectors('A', 'B', 'C', 'D');
      await flushInit();
      expect(text()).toBe('Detectors: 4');
      expect(line().title).toBe('A\nB\nC\nD');
    });

    it('styles every label as a form label, in sentence case with a colon, bold nowhere', async () => {
      fixture.componentRef.setInput('data', { ...mockData, dataset_name: 'Birds' } as any);
      await flushInit();
      const labels = Array.from(
        fixture.nativeElement.querySelectorAll('.results-summary .results-control-label'),
      ) as HTMLElement[];
      expect(labels.map((l) => l.textContent?.trim())).toEqual(['Dataset:', 'Media type:', 'Detector:']);
      expect(fixture.nativeElement.querySelector('.results-summary strong')).toBeNull();
      const columns = fixture.nativeElement.querySelector('vt-column-chooser .results-control-label') as HTMLElement;
      expect(columns.textContent?.trim()).toBe('Columns:');
    });
  });

  it('should display good hits by default', async () => {
    await flushInit();
    component.sides.set('good');
    expect(component.displayHits().length).toBe(2);
  });

  it('should display bad hits when selected', async () => {
    await flushInit();
    component.sides.set('bad');
    expect(component.displayHits().length).toBe(1);
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
      dry_bad_quota: 16,
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

  it("warns, in Tune's words, for each detector this dataset looks unlike the training dataset of (#5501)", async () => {
    const check = (frac: number, shifted: boolean) => ({
      reference_dataset_id: 'ref',
      reference_dataset_name: 'Caltech-101 (S)',
      labels_total: 39,
      labels_in_reference: 39,
      labels_in_active: 0,
      domain_shift: { frac_atypical: frac, shifted },
      reason: '',
    });
    fixture.componentRef.setInput('data', {
      ...mockData,
      dataset_name: 'Caltech-256 (S)',
      results: {
        mammal: { detector_name: 'Mammal', hits: [], training_domain: check(0.68, true) },
        bird: { detector_name: 'Bird', hits: [], training_domain: check(0.04, false) },
        typed: { detector_name: 'Typed', hits: [], training_domain: null },
      },
    } as any);
    await flushInit();
    const notes = fixture.nativeElement.querySelectorAll('.domain-note');
    expect(notes.length).toBe(1);
    expect(notes[0].textContent).toContain('68% of this dataset looks atypical vs Caltech-101 (S): likely domain shift');
    // Every example is there, so it says no count; its next step is the Browse button (#5545).
    expect(notes[0].textContent).toContain(
      '"Mammal" learned from "Caltech-101 (S)". Check its matches with Browse, or test it in Tune.',
    );
  });

  it('names the detectors that ran as their typed query apart from the centroid ones (#4758)', async () => {
    const quota = (tier: string, nGood: number, nBad: number) => ({
      tier,
      n_good: nGood,
      n_bad: nBad,
      goods_owed: Math.max(0, 3 - nGood),
      bads_owed: Math.max(0, 4 - nBad),
      good_quota: 3,
      bad_quota: 4,
      dry_bad_quota: 16,
    });
    fixture.componentRef.setInput('data', {
      ...mockData,
      results: {
        typed: { detector_name: 'Typed', hits: [], label_quota: quota('query', 0, 0) },
        under: { detector_name: 'Under', hits: [], label_quota: quota('centroid', 2, 1) },
        trained: { detector_name: 'Trained', hits: [], label_quota: quota('trained', 3, 4) },
      },
    } as any);
    await flushInit();
    expect(component.queryDetectors).toEqual(['Typed (3 more Goods and 4 more Bads)']);
    expect(component.centroidDetectors).toEqual(['Under (1 more Good and 3 more Bads)']);
    const queryNote = fixture.nativeElement.querySelector('.query-note')?.textContent ?? '';
    expect(queryNote).toContain('ran as their typed query');
    expect(queryNote).toContain('Typed');
    expect(queryNote).not.toContain('Under');
    expect(fixture.nativeElement.querySelector('.centroid-note')?.textContent).not.toContain('Typed');
  });

  // #5532: a note's detectors are a bulleted list, one per line, not a row
  // joined with semicolons.
  it('lists the detectors under each note one per item', async () => {
    const quota = { tier: 'query', n_good: 0, n_bad: 0, goods_owed: 3, bads_owed: 4, good_quota: 3, bad_quota: 4, dry_bad_quota: 16 };
    fixture.componentRef.setInput('data', {
      ...mockData,
      results: {
        a: { detector_name: 'Alpha', hits: [], label_quota: quota },
        b: { detector_name: 'Beta', hits: [], label_quota: quota },
      },
      missing_detectors: ['Gone, the first', 'Gone too'],
    } as any);
    await flushInit();
    const items = (selector: string) =>
      Array.from(fixture.nativeElement.querySelectorAll(`${selector} ul.name-list li`)).map((li) =>
        (li as HTMLElement).textContent?.trim(),
      );
    expect(items('.query-note')).toEqual(['Alpha (3 more Goods and 4 more Bads)', 'Beta (3 more Goods and 4 more Bads)']);
    expect(items('.missing-note')).toEqual(['Gone, the first', 'Gone too']);
    expect(fixture.nativeElement.querySelector('.query-note')?.textContent).not.toContain(';');
  });

  it('names the detectors held on their typed query past the quota apart from those short of it (#4889)', async () => {
    const quota = (nGood: number, nBad: number, holdOwed: number) => ({
      tier: 'query',
      n_good: nGood,
      n_bad: nBad,
      goods_owed: Math.max(0, 3 - nGood),
      bads_owed: Math.max(0, 4 - nBad),
      good_quota: 3,
      bad_quota: 4,
      dry_bad_quota: 16,
      hold_labels: 30,
      hold_owed: holdOwed,
    });
    fixture.componentRef.setInput('data', {
      ...mockData,
      results: {
        held: { detector_name: 'Held', hits: [], label_quota: quota(12, 15, 3) },
        short: { detector_name: 'Short', hits: [], label_quota: quota(1, 0, 0) },
      },
    } as any);
    await flushInit();
    expect(component.heldDetectors).toEqual(['Held (3 more labels)']);
    expect(component.queryDetectors).toEqual(['Short (2 more Goods and 4 more Bads)']);
    const heldNote = fixture.nativeElement.querySelector('.held-note')?.textContent ?? '';
    expect(heldNote).toContain('until it has 30 labels');
    expect(heldNote).toContain('Held');
    expect(heldNote).not.toContain('Too few labels');
    expect(fixture.nativeElement.querySelector('.query-note')?.textContent).not.toContain('Held');
  });

  it('says nothing about the centroid when every detector trained', async () => {
    await flushInit();
    expect(component.centroidDetectors).toEqual([]);
    expect(component.queryDetectors).toEqual([]);
    expect(component.heldDetectors).toEqual([]);
    expect(fixture.nativeElement.querySelector('.centroid-note')).toBeNull();
    expect(fixture.nativeElement.querySelector('.query-note')).toBeNull();
    expect(fixture.nativeElement.querySelector('.held-note')).toBeNull();
  });

  it('should display all hits when both selected', async () => {
    await flushInit();
    component.sides.set('both');
    expect(component.displayHits().length).toBe(3);
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

  // Zoneless canary: exporters are written from the getExporters() subscribe
  // (not a CD trigger) — as a signal they repaint the tabs.
  it('lists Clipboard and then each exporter as tabs, opening on Clipboard (#5464)', async () => {
    await flushInit();
    expect(component.exporters().length).toBe(2);
    expect(component.activeTab()).toBe('clipboard');
    const tabs = Array.from(fixture.nativeElement.querySelectorAll('.export-tabs .tab')) as HTMLElement[];
    expect(tabs.map((t) => t.textContent?.trim())).toEqual(['Clipboard', 'json', 'csv', 'AutoFind']);
    expect(tabs[0].classList).toContain('active');
    expect(fixture.nativeElement.querySelector('.tab-content vt-clipboard-copy')).toBeTruthy();
  });

  it("opens an exporter's tab with its fields", async () => {
    await flushInit();
    const csvTab = Array.from(fixture.nativeElement.querySelectorAll('.export-tabs .tab')).find(
      (t) => (t as HTMLElement).textContent?.trim() === 'csv',
    ) as HTMLButtonElement;
    csvTab.click();
    await settleZoneless(fixture);
    expect(component.activeTab()).toBe('csv');
    expect(component.exporterFields().length).toBe(1);
    expect(fixture.nativeElement.querySelector('#adr-path')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.tab-content vt-clipboard-copy')).toBeNull();
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
    const rows = el.querySelectorAll('.results-table .results-row');
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

  // The exporter by the name the user picked it under, not its registry name,
  // and on a failure where to fix it (#5463).
  describe('auto-export status', () => {
    function statusText(): string {
      return (fixture.nativeElement.querySelector('.auto-export-status') as HTMLElement).textContent ?? '';
    }

    it('names a failed exporter by its display name and points at Settings › AutoFind', async () => {
      fixture.componentRef.setInput('data', {
        ...mockData,
        auto_export: { exporter: 'webhook', exporter_display_name: 'Webhook (HTTP POST)', success: false, error: 'Connection refused' },
      });
      await flushInit();
      const text = statusText();
      expect(text).toContain('via Webhook (HTTP POST) failed: Connection refused.');
      expect(text).not.toContain('via webhook');
      expect(text).toContain('Settings › AutoFind');
    });

    it('names a successful exporter by its display name', async () => {
      fixture.componentRef.setInput('data', {
        ...mockData,
        auto_export: { exporter: 'server_json_file', exporter_display_name: 'Server JSON File', success: true },
      });
      await flushInit();
      expect(statusText()).toContain('Auto-exported via Server JSON File');
    });
  });

  // The dialog used to render an exporter picker with no button to run it;
  // now the footer carries Close / Export / Export and Close (#5464).
  describe('Export', () => {
    const footer = () =>
      Array.from(fixture.nativeElement.querySelectorAll('.form-actions button')) as HTMLButtonElement[];

    it('ends with Close, Export and Export and Close', async () => {
      await flushInit();
      expect(footer().map((b) => b.textContent?.trim())).toEqual(['Close', 'Export', 'Export and Close']);
      expect(footer()[1].disabled).toBe(false);
      expect(footer()[2].disabled).toBe(false);
    });

    it('exports exactly the listed rows: the chosen side moves into hits', async () => {
      await flushInit();
      component.sides.set('bad');
      const bad = component.exportPayload().results['detector1'];
      expect(bad.hits.map((h) => h.md5)).toEqual(['ghi789']);
      expect(bad.negative_hits).toEqual([]);
      expect(bad.total_hits).toBe(1);

      component.sides.set('both');
      const both = component.exportPayload().results['detector1'];
      expect(both.hits.map((h) => h.label)).toEqual(['good', 'good', 'bad']);
    });

    it('sends the ticked columns, in order, and never the thumbnail', async () => {
      fixture.componentRef.setInput('data', { ...mockData, dataset_id: 'ds1' } as any);
      await flushInit();
      expect(component.columns()[0].key).toBe('thumbnail');
      expect(component.exportPayload().selected_columns).toEqual(['origin_name', 'score', 'filename', 'origin', 'md5']);
      component.setColumnEnabled('md5', false);
      expect(component.exportPayload().selected_columns).toEqual(['origin_name', 'score', 'filename', 'origin']);
    });

    it('sends the run to the chosen exporter as find_results and confirms', async () => {
      await flushInit();
      component.selectTab('csv');
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

    it('closes after the export with Export and Close, and not without', async () => {
      await flushInit();
      const closed = vi.fn();
      component.closed.subscribe(closed);
      component.selectTab('json');

      component.exportResults();
      httpMock.expectOne('/api/exporters/export').flush({ success: true });
      expect(closed).not.toHaveBeenCalled();

      component.exportResults(true);
      httpMock.expectOne('/api/exporters/export').flush({ success: true });
      expect(closed).toHaveBeenCalledTimes(1);
    });

    it('copies the listed rows from the Clipboard tab, in the ticked columns', async () => {
      await flushInit();
      const writeText = vi.fn().mockResolvedValue(undefined);
      Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
      component.setColumnEnabled('origin', false);
      component.setColumnEnabled('md5', false);
      await settleZoneless(fixture);

      await component.exportResults();
      expect(writeText).toHaveBeenCalledWith('Name,Score,Filename\nSong 1,,song.wav\nTrack 2,,track.wav');
      expect(toast.success).toHaveBeenCalledWith(expect.objectContaining({ message: 'Copied 2 results to the clipboard' }));
      httpMock.expectNone('/api/exporters/export');
    });

    it('cannot export with every column cleared', async () => {
      await flushInit();
      component.toggleAllColumns();
      await settleZoneless(fixture);
      expect(component.exportBlocker).toBe('Tick a column to export');
      expect(footer()[1].disabled).toBe(true);
    });
  });

  // The controls that change the list sit above it (#5464).
  // What the other notches of the Threshold would have returned (#5540): the
  // run's scores split at each preset's cut, with no retrain.
  describe('the Threshold', () => {
    const lines = (four: number, one: number, quarter: number) => [
      { beta: 4, threshold: four },
      { beta: 1, threshold: one },
      { beta: 0.25, threshold: quarter },
    ];
    const cats = {
      detector_name: 'Cats',
      threshold: 0.5,
      beta: 1,
      lines: lines(0.3, 0.5, 0.9),
      balance: { status: 'unchecked', beta: 1, count: 2 },
      hits: [
        { id: 1, md5: 'a', score: 0.95 },
        { id: 2, md5: 'b', score: 0.6 },
      ],
      negative_hits: [
        { id: 3, md5: 'c', score: 0.4 },
        { id: 4, md5: 'd', score: 0.1 },
      ],
    };
    const runOf = (results: Record<string, unknown>) =>
      ({ media_type: 'image', dataset_id: 'ds1', detectors_run: Object.keys(results).length, results }) as any;
    const radios = () =>
      Array.from(fixture.nativeElement.querySelectorAll('.results-threshold input[type="radio"]')) as HTMLInputElement[];
    const note = () => fixture.nativeElement.querySelector('.notch-note') as HTMLElement | null;

    beforeEach(() => fixture.componentRef.setInput('data', runOf({ cats })));

    it("leads the row, before Good / Bad / Both; the sort is the headings' (#5584)", async () => {
      await flushInit();
      const row = fixture.nativeElement.querySelector('.results-controls-row') as HTMLElement;
      const order = Array.from(row.children).map((el) =>
        el.matches('vt-balance') ? 'threshold' : el.matches('.results-sides') ? 'sides' : 'other',
      );
      expect(order).toEqual(['threshold', 'sides']);
      expect(fixture.nativeElement.querySelector('.results-sort')).toBeNull();
    });

    it('shows the notch the run was cut at, and lists it as the run came', async () => {
      await flushInit();
      expect(radios().map((r) => r.checked)).toEqual([false, true, false]);
      expect([component.goodCount, component.badCount]).toEqual([2, 2]);
      expect(note()).toBeNull();
    });

    it('splits the run at another notch, Good and Bad alike, and says so', async () => {
      await flushInit();
      radios()[0].checked = true;
      radios()[0].dispatchEvent(new Event('change'));
      await settleZoneless(fixture);

      expect(component.notch()).toBe(4);
      expect(radios().map((r) => r.checked)).toEqual([true, false, false]);
      expect([component.goodCount, component.badCount]).toEqual([3, 1]);
      expect(component.displayHits().map((h) => h.id)).toEqual([1, 2, 3]);
      expect(note()?.textContent).toContain("The detectors' own Thresholds are unchanged");

      component.setNotch(0.25);
      component.sides.set('bad');
      expect(component.displayHits().map((h) => h.id)).toEqual([2, 3, 4]);
    });

    it("goes back to the run as it came at the run's own notch", async () => {
      await flushInit();
      component.setNotch(4);
      component.setNotch(1);
      await settleZoneless(fixture);
      expect(component.notch()).toBeNull();
      expect(note()).toBeNull();
    });

    it("exports the listed notch: its cut as threshold, its balance as beta, and not the run's balance", async () => {
      await flushInit();
      component.setNotch(4);
      const sent = component.exportPayload().results['cats'];
      expect(sent.hits.map((h) => h.id)).toEqual([1, 2, 3]);
      expect(sent.total_hits).toBe(3);
      expect(sent.threshold).toBe(0.3);
      expect(sent.beta).toBe(4);
      expect(sent.balance).toBeUndefined();

      component.setNotch(1);
      const own = component.exportPayload().results['cats'];
      expect([own.threshold, own.beta, own.balance?.count]).toEqual([0.5, 1, 2]);
    });

    it('is frozen, and says why, when no notch would list anything else', async () => {
      fixture.componentRef.setInput('data', runOf({ cats: { ...cats, lines: lines(0.5, 0.5, 0.5) } }));
      await flushInit();
      expect(radios().every((r) => r.disabled)).toBe(true);
      expect((radios()[0].parentElement as HTMLElement).title).toBe(component.notchFrozenHint);
    });

    it('is frozen for results that carry no cut at the other notches', async () => {
      const { lines: _none, ...bare } = cats;
      fixture.componentRef.setInput('data', runOf({ cats: bare }));
      await flushInit();
      expect(radios().every((r) => r.disabled)).toBe(true);
      expect([component.goodCount, component.badCount]).toEqual([2, 2]);
    });

    it('checks no notch for detectors cut at different balances, until one is picked', async () => {
      const dogs = {
        detector_name: 'Dogs',
        threshold: 0.2,
        beta: 4,
        lines: lines(0.2, 0.7, 0.9),
        hits: [{ id: 5, md5: 'e', score: 0.5 }],
        negative_hits: [],
      };
      fixture.componentRef.setInput('data', runOf({ cats, dogs }));
      await flushInit();
      expect(radios().map((r) => r.checked)).toEqual([false, false, false]);

      // Cats keeps its own cut at its own notch; Dogs is split at that notch's.
      component.setNotch(1);
      await settleZoneless(fixture);
      expect(radios().map((r) => r.checked)).toEqual([false, true, false]);
      expect([component.goodCount, component.badCount]).toEqual([2, 3]);
    });

    it('lists another run at its own notch', async () => {
      await flushInit();
      component.setNotch(4);
      fixture.componentRef.setInput('data', runOf({ cats: { ...cats, md5: 'other' } }));
      await settleZoneless(fixture);
      expect(component.notch()).toBeNull();
    });
  });

  describe('what the list shows', () => {
    const scored = {
      media_type: 'image',
      dataset_id: 'ds1',
      detectors_run: 2,
      results: {
        cats: {
          detector_name: 'Cats',
          hits: [
            { id: 1, md5: 'a', filename: 'a.jpg', score: 0.7, custom_metadata: { width: 64 } },
            { id: 2, md5: 'b', filename: 'b.jpg', score: 0.95 },
          ],
          negative_hits: [{ id: 3, md5: 'c', filename: 'c.jpg', score: 0.2 }],
        },
        dogs: { detector_name: 'Dogs', hits: [{ id: 4, md5: 'd', filename: 'd.jpg', score: 0.8 }], negative_hits: [] },
      },
    };
    const keys = () => component.columns().map((c) => c.key);

    beforeEach(() => fixture.componentRef.setInput('data', scored as any));

    it('orders the rows by score across every detector', async () => {
      await flushInit();
      expect(component.displayHits().map((h) => h.id)).toEqual([2, 4, 1]);
    });

    it('offers Label only for both sides, Detector only for several detectors, metadata unticked', async () => {
      await flushInit();
      expect(keys()).toEqual(['thumbnail', 'origin_name', 'score', 'similarity', 'detector', 'filename', 'origin', 'md5', 'width']);
      expect(component.columns().find((c) => c.key === 'width')!.enabled).toBe(false);
      expect(component.columns().find((c) => c.key === 'similarity')!.enabled).toBe(false);
      component.sides.set('both');
      expect(keys()).toContain('label');
      fixture.componentRef.setInput('data', { ...scored, results: { cats: scored.results.cats } } as any);
      expect(keys()).not.toContain('detector');
    });

    it('lists only the ticked columns, header and rows alike', async () => {
      await flushInit();
      component.setColumnEnabled('thumbnail', false);
      component.setColumnEnabled('origin', false);
      await settleZoneless(fixture);
      const head = Array.from(fixture.nativeElement.querySelectorAll('.results-head .results-col-label')) as HTMLElement[];
      expect(head.map((c) => c.textContent?.trim())).toEqual(['Name', 'Score', 'Detector', 'Filename', 'MD5']);
      const firstRow = fixture.nativeElement.querySelector('.results-row') as HTMLElement;
      expect(firstRow.querySelectorAll('.results-cell').length).toBe(5);
    });

    it('keeps the thumbnail size buttons on the columns row, and only with thumbnails (#5584)', async () => {
      await flushInit();
      expect(fixture.nativeElement.querySelector('vt-column-chooser .column-chooser-end vt-view-controls')).toBeTruthy();
      expect(fixture.nativeElement.querySelector('.results-controls-row vt-view-controls')).toBeNull();
      component.setColumnEnabled('thumbnail', false);
      await settleZoneless(fixture);
      expect(fixture.nativeElement.querySelector('vt-view-controls')).toBeNull();
    });

    it('offers no thumbnail, and no Similarity, for results that name no dataset; no thumbnail for a type without them', async () => {
      fixture.componentRef.setInput('data', { ...scored, dataset_id: undefined } as any);
      await flushInit();
      expect(keys()).not.toContain('thumbnail');
      expect(keys()).not.toContain('similarity');
      fixture.componentRef.setInput('data', { ...scored, media_type: 'text' } as any);
      expect(keys()).not.toContain('thumbnail');
    });

    it('sizes a row to its thumbnail, and to a line of text without one', async () => {
      await flushInit();
      expect(component.rowHeight()).toBe(component.thumbPx() + 4);
      component.setColumnEnabled('thumbnail', false);
      expect(component.rowHeight()).toBe(30);
    });

    // The headings, as the Dashboard's grids have them (#5584).
    describe('the headings', () => {
      const heads = () => Array.from(fixture.nativeElement.querySelectorAll('.results-head .results-col-head')) as HTMLElement[];
      const headOf = (key: string) => heads().find((h) => h.dataset['col'] === key)!;
      const drag = (from: string, to: string) => {
        component.cols.onColDragStart({ dataTransfer: null } as unknown as DragEvent, from);
        component.dropColumn({ preventDefault() {} } as unknown as DragEvent, to);
      };

      it('sort the list: Score highest first to start, a click sorts by the column, another turns it round', async () => {
        await flushInit();
        expect(component.sort()).toEqual({ column: 'score', asc: false });
        expect(headOf('score').getAttribute('aria-sort')).toBe('descending');
        headOf('filename').click();
        await settleZoneless(fixture);
        expect(component.displayHits().map((h) => h.filename)).toEqual(['a.jpg', 'b.jpg', 'd.jpg']);
        expect(headOf('filename').getAttribute('aria-sort')).toBe('ascending');
        expect(headOf('score').getAttribute('aria-sort')).toBe('none');
        headOf('filename').click();
        expect(component.displayHits().map((h) => h.filename)).toEqual(['d.jpg', 'b.jpg', 'a.jpg']);
        headOf('score').click();
        expect(component.displayHits().map((h) => h.id)).toEqual([1, 4, 2]);
        expect(headOf('thumbnail').getAttribute('aria-sort')).toBeNull();
      });

      it('sort by a metadata column as numbers, rows with none last either way', async () => {
        await flushInit();
        component.setColumnEnabled('width', true);
        await settleZoneless(fixture);
        headOf('width').click();
        expect(component.displayHits().map((h) => h.id)).toEqual([1, 2, 4]);
        headOf('width').click();
        expect(component.displayHits().map((h) => h.id)).toEqual([1, 2, 4]);
      });

      it('sort Label with Good first, and hand the sort back to Score once the column is gone', async () => {
        await flushInit();
        component.sides.set('both');
        await settleZoneless(fixture);
        headOf('label').click();
        expect(component.displayHits().map((h) => h.id)).toEqual([2, 4, 1, 3]);
        headOf('label').click();
        expect(component.displayHits().map((h) => h.id)).toEqual([3, 2, 4, 1]);
        component.sides.set('good');
        await settleZoneless(fixture);
        expect(component.sort()).toEqual({ column: 'score', asc: false });
      });

      it('reorder by a drag, and the clipboard and export follow', async () => {
        await flushInit();
        drag('md5', 'origin_name');
        await settleZoneless(fixture);
        expect(heads().map((h) => h.dataset['col'])).toEqual([
          'thumbnail', 'md5', 'origin_name', 'score', 'detector', 'filename', 'origin',
        ]);
        expect(component.exportPayload().selected_columns).toEqual(['md5', 'origin_name', 'score', 'detector', 'filename', 'origin']);
        expect(component.clipboardColumns().map((c) => c.label)).toEqual(['MD5', 'Name', 'Score', 'Detector', 'Filename', 'Origin']);
        // A metadata column the controller only learned of with the run can be dragged too.
        component.setColumnEnabled('width', true);
        await settleZoneless(fixture);
        drag('width', 'score');
        await settleZoneless(fixture);
        expect(heads().map((h) => h.dataset['col'])).toEqual([
          'thumbnail', 'md5', 'origin_name', 'width', 'score', 'detector', 'filename', 'origin',
        ]);
      });

      it('carry a resize handle between two columns, but not on the first nor beside the thumbnail', async () => {
        await flushInit();
        const handles = () => heads().map((h) => !!h.querySelector('.col-resize-handle'));
        // thumbnail, name, score, detector, filename, origin, md5
        expect(handles()).toEqual([false, false, true, true, true, true, true]);
        component.setColumnEnabled('thumbnail', false);
        await settleZoneless(fixture);
        expect(handles()).toEqual([false, true, true, true, true, true]);
      });

      it('resize: the grid follows the drag, the thumbnail keeping its tile', async () => {
        await flushInit();
        const thumb = `${component.thumbPx() + 12}px`;
        expect(component.gridTemplate().split(' ')[0]).toBe(thumb);
        expect(component.gridTemplate()).toContain('minmax(6rem, 2fr) 5rem');
        // jsdom lays nothing out: give each heading a width of its own.
        for (const head of heads()) head.getBoundingClientRect = () => ({ width: 100 }) as DOMRect;
        const handle = headOf('score').querySelector('.col-resize-handle') as HTMLElement;
        handle.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: 300 }));
        document.dispatchEvent(new MouseEvent('mousemove', { clientX: 370 }));
        await settleZoneless(fixture);
        // Seven headings of 100px: a seventh each; Name grew by 70px (ten
        // points) and Score shrank by as much. The thumbnail is still its tile.
        const share = (track: string) => parseFloat(track.match(/^minmax\(0, (.*)fr\)$/)![1]);
        const tracks = component.gridTemplate().split(' ');
        expect(tracks[0]).toBe(thumb);
        expect(share(tracks[1])).toBeCloseTo(100 / 7 + 10, 6);
        expect(share(tracks[2])).toBeCloseTo(100 / 7 - 10, 6);
        document.dispatchEvent(new MouseEvent('mouseup'));
        expect(document.body.style.cursor).toBe('');
        const row = fixture.nativeElement.querySelector('.results-row') as HTMLElement;
        expect(row.style.gridTemplateColumns).toBe(component.gridTemplate());
      });
    });

    // The Similarity column (#5584): each item's place in the dataset's 1-D
    // order, asked for when the column is ticked; sort by it for look-alikes.
    describe('the Similarity column', () => {
      const URL = '/api/medias/similarity-order';
      const tick = async () => {
        component.setColumnEnabled('similarity', true);
        await settleZoneless(fixture);
      };
      const cells = () => component.sortedRows().map((row) => component.cellValue(row, 'similarity'));

      it("asks the run dataset for its order once ticked, shows each item's place, and sorts by it", async () => {
        await flushInit();
        httpMock.expectNone(URL);
        await tick();
        const req = httpMock.expectOne(URL);
        // Asked of the run's dataset by id, whichever pair is active.
        expect(req.request.context.get(DATASET_OVERRIDE)).toBe('ds1');
        expect(req.request.body.ids).toEqual([1, 2, 3, 4]);
        req.flush({ status: 'ready', ids: [4, 1, 3, 2], method: 'umap' });
        // Still the score order, with each item's place shown and copied.
        expect(component.displayHits().map((h) => h.id)).toEqual([2, 4, 1]);
        expect(cells()).toEqual(['4', '1', '2']);
        expect(component.clipboardRows()[0]['similarity']).toBe('4');
        component.cols.sortBy('similarity');
        expect(component.displayHits().map((h) => h.id)).toEqual([4, 1, 2]);
        component.cols.sortBy('similarity');
        expect(component.displayHits().map((h) => h.id)).toEqual([2, 1, 4]);
        // Unticked and ticked again: the order is in hand.
        component.setColumnEnabled('similarity', false);
        await settleZoneless(fixture);
        await tick();
        httpMock.expectNone(URL);
      });

      it('waits on the fit, asking again, and keeps the score order meanwhile', async () => {
        await flushInit();
        vi.useFakeTimers();
        try {
          component.setColumnEnabled('similarity', true);
          TestBed.tick();
          httpMock.expectOne(URL).flush({ status: 'building', job_id: 'j', overall: 0.4 });
          expect(component.similarity().status).toBe('building');
          expect(component.similarityFraction(component.similarity())).toBe(0.4);
          expect(cells()).toEqual(['', '', '']);
          component.cols.sortBy('similarity');
          expect(component.displayHits().map((h) => h.id)).toEqual([2, 4, 1]);
          await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.SIMILARITY_POLL_MS);
          httpMock.expectOne(URL).flush({ status: 'ready', ids: [1, 2, 3, 4], method: 'umap' });
          expect(component.displayHits().map((h) => h.id)).toEqual([1, 2, 4]);
        } finally {
          vi.useRealTimers();
        }
      });

      it('Cancel drops the column, hands the sort back to Score, and stops asking', async () => {
        await flushInit();
        vi.useFakeTimers();
        try {
          component.setColumnEnabled('similarity', true);
          TestBed.tick();
          httpMock.expectOne(URL).flush({ status: 'building', job_id: 'j' });
          component.cols.sortBy('similarity');
          component.cancelSimilarity();
          TestBed.tick();
          expect(component.showsSimilarity()).toBe(false);
          expect(component.sort()).toEqual({ column: 'score', asc: false });
          await vi.advanceTimersByTimeAsync(AutoDetectResultsModalComponent.SIMILARITY_POLL_MS * 2);
          httpMock.expectNone(URL);
        } finally {
          vi.useRealTimers();
        }
      });

      it('shows why there is no order, and asks again on Try again', async () => {
        await flushInit();
        await tick();
        httpMock.expectOne(URL).flush({ message: 'Dataset ds1 is not loaded.' }, { status: 409, statusText: 'Conflict' });
        await settleZoneless(fixture);
        expect(component.similarityError()).toBe('Dataset ds1 is not loaded.');
        const error = fixture.nativeElement.querySelector('.similarity-error') as HTMLElement;
        expect(error.textContent).toContain('not loaded');
        (error.querySelector('button') as HTMLButtonElement).click();
        httpMock.expectOne(URL).flush({ status: 'ready', ids: [], method: 'umap' });
      });
    });
  });

  // The AutoFind tab (#5584): Export sends the run through the exporter set
  // up for AutoFind, as a finished run does, with read-only lines on what
  // that is.
  describe('the AutoFind tab', () => {
    const run = { ...mockData, run_id: '_autofind_9', dataset_id: 'ds1' };
    const EXPORT_URL = '/api/autofind/runs/_autofind_9/export';
    const tabs = () => Array.from(fixture.nativeElement.querySelectorAll('.export-tabs .tab')) as HTMLElement[];
    const autofindTab = () => tabs().find((t) => t.textContent?.trim() === 'AutoFind') as HTMLButtonElement;
    const content = () =>
      (fixture.nativeElement.querySelector('.tab-content') as HTMLElement).textContent?.replace(/\s+/g, ' ') ?? '';
    const exportButton = () => Array.from(fixture.nativeElement.querySelectorAll('.form-actions button'))[1] as HTMLButtonElement;
    const configured = (paused = false) =>
      settings.set({
        autofind_exporter: 'csv',
        autofind_exporter_field_values: { csv: { path: '/out/run.csv' } },
        autofind_export_enabled: !paused,
      } as AppSettings);

    beforeEach(() => fixture.componentRef.setInput('data', run as any));

    it('is the last tab, offered once there are exporters', async () => {
      await flushInit();
      expect(tabs().map((t) => t.textContent?.trim())).toEqual(['Clipboard', 'json', 'csv', 'AutoFind']);
    });

    it('says to set an exporter up when none is, and holds Export', async () => {
      settings.set({ autofind_exporter: '' } as AppSettings);
      await flushInit();
      autofindTab().click();
      await settleZoneless(fixture);
      expect(content()).toContain('No results exporter is set up for AutoFind');
      expect(exportButton().disabled).toBe(true);
      expect(exportButton().title).toBe('Set up a results exporter on the Settings AutoFind tab');
    });

    it('shows, read-only, what Export will send and with what settings', async () => {
      configured(true);
      await flushInit();
      autofindTab().click();
      await settleZoneless(fixture);
      const text = content();
      expect(text).toContain("Export sends this run's 2 results via csv, as a finished run does");
      expect(text).toContain('Export results when AutoFind finishes');
      const readout = Array.from(fixture.nativeElement.querySelectorAll('.autofind-readout dt, .autofind-readout dd')).map(
        (el) => (el as HTMLElement).textContent?.trim(),
      );
      expect(readout).toEqual(['Path:', '/out/run.csv']);
      expect(fixture.nativeElement.querySelector('.tab-content input')).toBeNull();
      expect(exportButton().disabled).toBe(false);
      expect(exportButton().title).toBe("Send this run's results via csv, as a finished run does");
    });

    it('hides a password, shows a checkbox as Yes or No, and a blank as its default', async () => {
      configured();
      await flushInit();
      const field = (key: string, field_type: string, dflt?: string) => ({ key, field_type, default: dflt }) as any;
      expect(component.autofindFieldText(field('p', 'password'))).toBe('—');
      settings.set({
        autofind_exporter: 'csv',
        autofind_exporter_field_values: { csv: { p: 'hunter2', c: 'true', d: '' } },
      } as AppSettings);
      expect(component.autofindFieldText(field('p', 'password'))).toBe('••••••••');
      expect(component.autofindFieldText(field('c', 'checkbox'))).toBe('Yes');
      expect(component.autofindFieldText(field('x', 'checkbox'))).toBe('No');
      expect(component.autofindFieldText(field('d', 'text', 'out.csv'))).toBe('out.csv');
    });

    it('sends the run through the AutoFind exporter, and confirms', async () => {
      configured();
      await flushInit();
      component.selectTab(AUTOFIND_TAB);
      component.exportResults();
      const req = httpMock.expectOne(EXPORT_URL);
      expect(req.request.method).toBe('POST');
      expect(component.exporting()).toBe(true);
      req.flush({ exporter: 'csv', exporter_display_name: 'CSV', success: true, message: 'Saved.' });
      expect(component.exporting()).toBe(false);
      expect(toast.success).toHaveBeenCalledWith(
        expect.objectContaining({ message: 'Exported 2 results to csv', detail: 'Saved.' }),
      );
      httpMock.expectNone('/api/exporters/export');
    });

    it("reports the exporter's failure in the run's own words, and stays open", async () => {
      configured();
      await flushInit();
      const closed = vi.fn();
      component.closed.subscribe(closed);
      component.selectTab(AUTOFIND_TAB);
      component.exportResults(true);
      httpMock
        .expectOne(EXPORT_URL)
        .flush({ exporter: 'csv', exporter_display_name: 'CSV', success: false, error: 'Disk full' });
      expect(toast.error).toHaveBeenCalledWith(
        expect.objectContaining({ detail: expect.stringContaining('via CSV failed: Disk full') }),
      );
      expect(toast.success).not.toHaveBeenCalled();
      expect(closed).not.toHaveBeenCalled();
    });

    it('is held for results that name no run', async () => {
      configured();
      fixture.componentRef.setInput('data', { ...run, run_id: undefined } as any);
      await flushInit();
      component.selectTab(AUTOFIND_TAB);
      expect(component.exportBlocker).toBe('These results name no run to send');
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
        { provide: MediaTypeCapabilityService, useFactory: mediaTypeCapsStub },
        { provide: ToastService, useValue: { success: vi.fn() } },
        { provide: ContextSwitchService, useValue: contextSwitch },
        { provide: BrowseSubsetPrepService, useValue: browsePrep },
        { provide: SettingsStateService, useValue: makeSettingsStateStub().stub },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(AutoDetectResultsModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('data', run as any);
  });

  it('browses the listed rows, once each, following the Good / Bad / Both filter', () => {
    expect(component.browseIds).toEqual([1, 2]);
    component.sides.set('bad');
    expect(component.browseIds).toEqual([3]);
    component.sides.set('both');
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
    expect(browsePrep.start).toHaveBeenCalledWith(
      'ds1',
      [1, 2],
      { kind: 'results', runId: '_autofind_7', sides: 'good' },
      expect.any(Function),
    );

    // The dialog closes only once the map is ready and Browse opens.
    expect(closed).not.toHaveBeenCalled();
    browsePrep.start.mock.calls[0][3]();
    expect(closed).toHaveBeenCalledTimes(1);
  });

  // The browse URL names the picked rows, so a link opened without the
  // in-memory handoff rebuilds the same map from the run (#5129).
  it('names the picked rows in the browse origin', () => {
    component.sides.set('bad');
    component.browse();
    switched.next();
    expect(browsePrep.start).toHaveBeenCalledWith(
      'ds1',
      [3],
      { kind: 'results', runId: '_autofind_7', sides: 'bad' },
      expect.any(Function),
    );
  });

  // A list split at another notch of the Threshold (#5540) browses that
  // split, and the URL names the notch so a reload lays out the same map.
  it('browses the notch the list is split at, and names it in the browse origin', () => {
    const lines = [
      { beta: 4, threshold: 0.1 },
      { beta: 1, threshold: 0.5 },
      { beta: 0.25, threshold: 0.9 },
    ];
    fixture.componentRef.setInput('data', {
      ...run,
      results: {
        owl: {
          beta: 1,
          threshold: 0.5,
          lines,
          hits: [
            { id: 1, md5: 'a', score: 0.95 },
            { id: 2, md5: 'b', score: 0.6 },
          ],
          negative_hits: [{ id: 3, md5: 'c', score: 0.2 }],
        },
      },
    } as any);
    component.setNotch(0.25);
    expect(component.browseIds).toEqual([1]);
    component.setNotch(4);
    expect(component.browseIds).toEqual([1, 2, 3]);

    component.browse();
    switched.next();
    expect(browsePrep.start).toHaveBeenCalledWith(
      'ds1',
      [1, 2, 3],
      { kind: 'results', runId: '_autofind_7', sides: 'good', notch: 4 },
      expect.any(Function),
    );
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

  // #5102: the way in to the results, not one small button among the footer's;
  // #5464: under the list, since it acts on the rows rather than changing them.
  it('sits large under the list, not in the footer', async () => {
    fixture.detectChanges();
    await fixture.whenStable();
    const el: HTMLElement = fixture.nativeElement;
    const lead = el.querySelector('.browse-lead button') as HTMLButtonElement;
    expect(lead.textContent?.trim()).toBe('Browse');
    expect(lead.classList).toContain('btn--lg');
    expect(lead.disabled).toBe(false);
    const list = el.querySelector('.results-table') as HTMLElement;
    expect(list.compareDocumentPosition(lead) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const footer = Array.from(el.querySelectorAll('.form-actions button')).map((b) => b.textContent?.trim());
    expect(footer).not.toContain('Browse');

    lead.click();
    expect(contextSwitch.applyActivePair).toHaveBeenCalledWith('ds1', '');
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
        { provide: MediaTypeCapabilityService, useFactory: mediaTypeCapsStub },
        { provide: ToastService, useValue: toast },
        { provide: ContextSwitchService, useValue: { applyActivePair: vi.fn(() => new Subject<void>()) } },
        { provide: BrowseSubsetPrepService, useValue: { preparing: signal(false), start: vi.fn(), cancel: vi.fn() } },
        { provide: SettingsStateService, useValue: makeSettingsStateStub().stub },
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
