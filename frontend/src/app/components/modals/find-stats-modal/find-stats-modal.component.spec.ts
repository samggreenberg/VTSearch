import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { FindStatsModalComponent } from './find-stats-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { ActiveContextService } from '../../../services/active-context.service';
import type { DatasetRegistryEntry } from '../../../models/api.models';
import { SAMPLE_RANGE, wireBalance } from '../../../testing/line-balance';

describe('FindStatsModalComponent', () => {
  let component: FindStatsModalComponent;
  let fixture: ComponentFixture<FindStatsModalComponent>;
  let httpMock: HttpTestingController;

  const mockStats = {
    stale: false,
    total_good: 40,
    total_bad: 60,
    verified_count: 12,
    confirmed_good: 30,
    rescued_false_neg: 4,
    culled_false_pos: 5,
    confirmed_bad: 50,
    agreements: 90,
    corrections: 10,
    agreement_rate: 0.9,
    verified_precision: 0.7,
    verified_called_good: 10,
    verified_kept_good: 7,
    threshold: 0.5,
    balance: wireBalance('checked'),
    n_scored: 1000,
    n_returned: 40,
    precision_curve: [
      { n_returned: 1, threshold: 0.99, checked: 0, checked_good: 0, verified_precision: null },
      { n_returned: 10, threshold: 0.9, checked: 2, checked_good: 2, verified_precision: 1 },
      { n_returned: 40, threshold: 0.5, checked: 10, checked_good: 7, verified_precision: 0.7 },
      { n_returned: 1000, threshold: 0.01, checked: 12, checked_good: 7, verified_precision: 0.5833 },
    ],
  };

  // Evidence-coverage is fetched on init too; the "nothing to measure" reply
  // keeps the section hidden. Flushed by every test so `httpMock.verify()` sees
  // no dangling request.
  const mockEvidenceUnavailable = {
    available: false,
    n_items: 0,
    n_pos_labels: 0,
    n_neg_labels: 0,
    k: 1,
    alpha: 0.05,
    frac_unsupported: 0,
    expected_unsupported: 0.05,
    z_score: 0,
    median_support: 1,
    frac_low_trust: 0,
    median_trust: 1,
    unsupported: false,
  };

  beforeEach(async () => {
    await configureZoneless({
      imports: [FindStatsModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(FindStatsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/find/stats').flush(mockStats);
    httpMock.expectOne('/api/find/evidence-coverage').flush(mockEvidenceUnavailable);
    await settleZoneless(fixture);
    expect(component).toBeTruthy();
  });

  // Zoneless staleness canary: the stats land in an HTTP subscribe (an unpatched
  // callback). The table repaints only because `loading`/`stats` are signals read
  // in the template. Flush the GET and assert the loaded DOM renders with no
  // manual `detectChanges`.
  it('repaints from loading to the loaded table (zoneless canary)', async () => {
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('.loading-text')).toBeTruthy();

    httpMock.expectOne('/api/find/stats').flush(mockStats);
    httpMock.expectOne('/api/find/evidence-coverage').flush(mockEvidenceUnavailable);
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.loading-text')).toBeFalsy();
    expect(fixture.nativeElement.querySelector('.precision-chart')).toBeTruthy();
    expect(fixture.nativeElement.textContent).toContain('90%'); // agreement rate
  });

  it('repaints the error text on a failed load (zoneless canary)', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/find/stats').flush(
      { message: 'no find run' },
      { status: 404, statusText: 'Not Found' },
    );
    httpMock.expectOne('/api/find/evidence-coverage').flush(mockEvidenceUnavailable);
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('no find run');
  });

  it('emits closed on close', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/find/stats').flush(mockStats);
    httpMock.expectOne('/api/find/evidence-coverage').flush(mockEvidenceUnavailable);
    await settleZoneless(fixture);

    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('renders no domain-overlap section without a reference candidate', async () => {
    // The real (empty) DatasetStateService yields no candidates, so the
    // section is absent and no domain-shift request is made.
    await fixture.whenStable();
    httpMock.expectOne('/api/find/stats').flush(mockStats);
    httpMock.expectOne('/api/find/evidence-coverage').flush(mockEvidenceUnavailable);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.domain-overlap')).toBeFalsy();
  });

  it('renders the evidence-coverage chip when the report is available', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/find/stats').flush(mockStats);
    httpMock.expectOne('/api/find/evidence-coverage').flush({
      available: true,
      n_items: 80,
      n_pos_labels: 20,
      n_neg_labels: 30,
      k: 1,
      alpha: 0.05,
      frac_unsupported: 0.62,
      expected_unsupported: 0.05,
      z_score: 9.1,
      median_support: 0.03,
      frac_low_trust: 0.25,
      median_trust: 0.8,
      unsupported: true,
    });
    await settleZoneless(fixture);

    // The evidence section reuses the .domain-chip shell; the second chip is it.
    const chips = fixture.nativeElement.querySelectorAll('.domain-chip');
    const evidenceChip = chips[chips.length - 1] as HTMLElement;
    expect(evidenceChip).toBeTruthy();
    expect(evidenceChip.classList.contains('shifted')).toBe(true);
    expect(evidenceChip.textContent).toContain('62%');
    expect(fixture.nativeElement.textContent).toContain('evidence vacuum');
  });

  describe('precision-vs-returned chart (#4242)', () => {
    async function load(overrides: Record<string, unknown> = {}) {
      await fixture.whenStable();
      httpMock.expectOne('/api/find/stats').flush({ ...mockStats, ...overrides });
      httpMock.expectOne('/api/find/evidence-coverage').flush(mockEvidenceUnavailable);
      await settleZoneless(fixture);
      return fixture.nativeElement as HTMLElement;
    }

    it('draws the checked curve, skipping the counts with nothing checked', async () => {
      const el = await load();
      const ver = el.querySelector('.line-verified')!.getAttribute('points')!.trim().split(' ');
      expect(ver.length).toBe(3); // the top 1 has nothing checked
    });

    it('makes no bound claim: no estimate curve, and no "at least" anywhere (#4360)', async () => {
      const el = await load();
      expect(el.querySelectorAll('.precision-chart polyline').length).toBe(1);
      const chart = el.querySelector('.chart-wrap')!.textContent!.replace(/\s+/g, ' ');
      expect(chart).not.toMatch(/at least|estimat/i);
    });

    it('puts counts on a log scale from 1 to the corpus size', async () => {
      await load();
      // 320 wide until the ResizeObserver measures it (jsdom has none).
      expect(component.xFor(1)).toBeCloseTo(40);
      expect(component.xFor(1000)).toBeCloseTo(304);
      // 10 and 100 split the three decades evenly.
      expect(component.xFor(100) - component.xFor(10)).toBeCloseTo(component.xFor(10) - component.xFor(1));
      expect(component.xTicks.map((t) => t.label)).toEqual(['1', '10', '100', '1k']);
    });

    it('marks the line and reads the checked precision off it', async () => {
      const el = await load();
      expect(el.querySelector('.precision-chart .current')).toBeTruthy();
      const readout = el.querySelector('.chart-readout')!.textContent!.replace(/\s+/g, ' ');
      expect(readout).toContain('At the line (40 returned): checked 70%');
      expect(readout).toContain('(7 of 10 Good)');
    });

    it('shows the Kept rate as verified precision with its count', async () => {
      const el = await load();
      const text = el.textContent!.replace(/\s+/g, ' ');
      expect(text).toContain('70% (7 of 10 checked)');
    });

    it('says so when no match has been checked', async () => {
      const el = await load({ verified_precision: null, verified_called_good: 0, verified_kept_good: 0 });
      expect(el.textContent).toContain('(no matches checked yet)');
    });

    it('follows the pointer to the nearest sampled count', async () => {
      const el = await load();
      const svg = el.querySelector('.precision-chart') as SVGSVGElement;
      vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({ left: 0, width: 320 } as DOMRect);
      svg.dispatchEvent(new MouseEvent('mousemove', { clientX: component.xFor(10) + 1 }));
      await settleZoneless(fixture);
      expect(component.hoverIndex()).toBe(1);
      expect(el.querySelector('.crosshair')).toBeTruthy();
      expect(el.querySelector('.chart-readout')!.textContent).toContain('Top 10');
      // The top 1 has nothing checked in it.
      svg.dispatchEvent(new MouseEvent('mousemove', { clientX: component.xFor(1) }));
      await settleZoneless(fixture);
      expect(el.querySelector('.chart-readout')!.textContent!.replace(/\s+/g, ' ')).toContain(
        'Top 1: nothing checked',
      );
      svg.dispatchEvent(new MouseEvent('mouseleave'));
      await settleZoneless(fixture);
      expect(el.querySelector('.crosshair')).toBeFalsy();
    });

    it('draws in its measured width rather than stretching a fixed one', async () => {
      let notify: ResizeObserverCallback = () => {};
      class FakeResizeObserver {
        constructor(cb: ResizeObserverCallback) {
          notify = cb;
        }
        observe(): void {}
        disconnect(): void {}
      }
      vi.stubGlobal('ResizeObserver', FakeResizeObserver);
      try {
        const el = await load();
        notify([{ contentRect: { width: 900 } } as ResizeObserverEntry], {} as ResizeObserver);
        await settleZoneless(fixture);
        expect(el.querySelector('.precision-chart')!.getAttribute('viewBox')).toBe('0 0 900 170');
        expect(component.xFor(1000)).toBeCloseTo(884);
      } finally {
        vi.unstubAllGlobals();
      }
    });
  });

  describe('the balance on the chart (#4413)', () => {
    async function load(overrides: Record<string, unknown> = {}) {
      await fixture.whenStable();
      httpMock.expectOne('/api/find/stats').flush({ ...mockStats, ...overrides });
      httpMock.expectOne('/api/find/evidence-coverage').flush(mockEvidenceUnavailable);
      await settleZoneless(fixture);
      return fixture.nativeElement as HTMLElement;
    }

    const legend = (el: HTMLElement) => el.querySelector('.chart-legend')!.textContent!.replace(/\s+/g, ' ');

    it('draws no line across the chart: a balance is a preference, not a precision to keep', async () => {
      const el = await load({ balance: wireBalance('checked', { beta: 0.5 }) });
      expect(el.querySelector('.precision-chart .floor')).toBeNull();
      expect(el.querySelector('.swatch-floor')).toBeNull();
      expect(el.querySelector('.precision-chart .current')!.getAttribute('class')).toBe('current');
      // The balance by neither name nor number (#4298, #4317); the check's range and the axis stay numbers.
      expect(legend(el)).not.toContain('Threshold');
      expect(legend(el)).not.toMatch(/Centered|Complete|Correct|beta|F1/);
      expect(legend(el)).toContain('Line: checked (32 kept)');
      expect(legend(el)).toContain('Likely 55–100% right (checked 5)');
      expect(legend(el)).not.toContain('0.5');
      // The Inclusion stepper's legend is gone.
      expect(legend(el)).not.toContain('incl');
    });

    it('stands the range at the line, from its low end to its high end (#4273)', async () => {
      const el = await load({ balance: wireBalance('checked', { precision: SAMPLE_RANGE }) });
      const bar = el.querySelector('.precision-chart .likely-range .range-bar')!;
      expect(Number(bar.getAttribute('y'))).toBeCloseTo(component.yFor(0.73));
      expect(Number(bar.getAttribute('height'))).toBeCloseTo(component.yFor(0.11) - component.yFor(0.73));
      const cut = el.querySelector('.precision-chart .current')!;
      expect(Number(bar.getAttribute('x')) + 4).toBeCloseTo(Number(cut.getAttribute('x1')));
      expect(el.querySelector('.precision-chart .likely-range title')!.textContent).toBe(
        'Likely 11–73% right, from 5 random picks (2 right).',
      );
    });

    it('draws a stale range exactly as a current one; only its tooltip differs', async () => {
      const markup = (el: HTMLElement) =>
        el.querySelector('.likely-range')!.outerHTML.replace(/<title[^>]*>[^<]*<\/title>/, '').replace(/aria-label="[^"]*"/, '');
      const fresh = await load({ balance: wireBalance('checked', { precision: SAMPLE_RANGE }) });
      const freshMarkup = markup(fresh);
      const freshLegend = legend(fresh);
      component.stats.set({ ...component.stats()!, balance: wireBalance('checked', { precision: { ...SAMPLE_RANGE, stale: true } }) } as never);
      await settleZoneless(fixture);
      const el = fixture.nativeElement as HTMLElement;
      expect(markup(el)).toBe(freshMarkup);
      expect(legend(el)).toBe(freshLegend);
      expect(el.querySelector('.likely-range title')!.textContent).toContain('Measured before your later votes');
    });

    it('says what a check found on the set the line keeps, the share right as a number and the found in words, with a plain line', async () => {
      const el = await load({ balance: wireBalance('checked', { beta: 0.5 }) });
      expect(el.querySelector('.precision-chart .current')!.getAttribute('class')).toBe('current');
      expect(legend(el)).toContain('Line: checked (32 kept)');
      const text = el.textContent!.replace(/\s+/g, ' ');
      expect(text).toContain('Checked: 5 random picks found the 32 the line keeps likely 55–100% right, with about half of them found.');
      expect(el.querySelector('.chart-note[title]')!.getAttribute('title')).toContain('Likely 55–100% right, from 5 random picks (5 right).');
      // Nothing is met or fallen short of, and the copy names no cause.
      expect(text).not.toMatch(/\bshort\b|\bfell\b|confirmed|enough|sparse|weak model|unpromised/i);
    });

    it('notes a checked line with no ranges by its legend alone', async () => {
      const el = await load({ balance: wireBalance('checked', { precision: null, recall: null, fbeta: null }) });
      expect(legend(el)).toContain('Line: checked (32 kept)');
      expect(el.querySelector('.likely-range')).toBeNull();
      expect(el.textContent).not.toContain('Checked:');
    });

    it('says an unchecked line keeps its starting candidate, with no range', async () => {
      const el = await load({
        balance: wireBalance('unchecked', { beta: 2, count: 128, schedule: { candidate: 128, rounds: 3, picks: 5 } }),
      });
      expect(el.querySelector('.precision-chart .current')!.getAttribute('class')).toBe('current');
      expect(el.querySelector('.likely-range')).toBeNull();
      expect(legend(el)).toContain('Line: the top 128, unchecked');
      expect(legend(el)).not.toContain('Likely');
      expect(legend(el)).not.toContain('Threshold');
      const notes = Array.from(el.querySelectorAll('.chart-note')).map((n) => n.textContent!.replace(/\s+/g, ' '));
      expect(notes.some((n) => n.includes('The line keeps the top 128, unchecked'))).toBe(true);
      // Find tests the balance it was given: nothing here points at a check (#4317).
      expect(el.textContent).not.toMatch(/Check \d+ picks/);
      expect(el.textContent).not.toContain('default cut');
      expect(el.textContent).not.toContain('unpromised');
    });
  });
});

describe('FindStatsModalComponent — training-domain overlap', () => {
  let fixture: ComponentFixture<FindStatsModalComponent>;
  let httpMock: HttpTestingController;

  const mockStats = {
    stale: false,
    total_good: 40,
    total_bad: 60,
    verified_count: 12,
    confirmed_good: 30,
    rescued_false_neg: 4,
    culled_false_pos: 5,
    confirmed_bad: 50,
    agreements: 90,
    corrections: 10,
    agreement_rate: 0.9,
    verified_precision: 0.7,
    verified_called_good: 10,
    verified_kept_good: 7,
    threshold: 0.5,
    balance: wireBalance('unchecked'),
    n_scored: 100,
    n_returned: 10,
    precision_curve: [],
  };

  // Active dataset 'ds-b' (siglip); 'ds-a' is a loaded siglip reference,
  // 'ds-c' is filtered out (different embedder), 'ds-d' is filtered out
  // (not loaded).
  const datasets: DatasetRegistryEntry[] = [
    { id: 'ds-b', name: 'Haystack B', media_type: 'audio', loaded: true, embedder: 'siglip' },
    { id: 'ds-a', name: 'Haystack A', media_type: 'audio', loaded: true, embedder: 'siglip' },
    { id: 'ds-c', name: 'Other embedder', media_type: 'audio', loaded: true, embedder: 'clap' },
    { id: 'ds-d', name: 'Unloaded', media_type: 'audio', loaded: false, embedder: 'siglip' },
  ];

  async function setup(): Promise<void> {
    await configureZoneless({
      imports: [FindStatsModalComponent],
      providers: [
        ...provideHttpTesting(),
        {
          provide: DatasetStateService,
          useValue: {
            datasets,
            datasetById: () => new Map(datasets.map((d) => [d.id, d])),
          },
        },
      ],
    }).compileComponents();

    TestBed.inject(ActiveContextService).setActivePair('ds-b', '');
    fixture = TestBed.createComponent(FindStatsModalComponent);
    httpMock = TestBed.inject(HttpTestingController);
  }

  afterEach(() => {
    httpMock.verify();
  });

  it('auto-selects the sole matching-embedder reference and shows the chip', async () => {
    await setup();
    await fixture.whenStable();
    httpMock.expectOne('/api/find/stats').flush(mockStats);
    // The one eligible reference is ds-a (loaded, same embedder, not active).
    httpMock.expectOne('/api/datasets/registry/ds-a/domain-shift').flush({
      reference_dataset_id: 'ds-a',
      n_items: 100,
      alpha: 0.05,
      frac_atypical: 0.31,
      expected_atypical: 0.05,
      z_score: 8.2,
      median_pvalue: 0.4,
      shifted: true,
    });
    httpMock.expectOne('/api/find/evidence-coverage').flush({
      available: false,
      n_items: 0,
      n_pos_labels: 0,
      n_neg_labels: 0,
      k: 1,
      alpha: 0.05,
      frac_unsupported: 0,
      expected_unsupported: 0.05,
      z_score: 0,
      median_support: 1,
      frac_low_trust: 0,
      median_trust: 1,
      unsupported: false,
    });
    await settleZoneless(fixture);

    const chip = fixture.nativeElement.querySelector('.domain-chip') as HTMLElement;
    expect(chip).toBeTruthy();
    expect(chip.classList.contains('shifted')).toBe(true);
    expect(chip.textContent).toContain('31%');
    expect(chip.textContent).toContain('Haystack A');
    expect(chip.textContent).toContain('likely domain shift');
    // Only the two eligible references appear in the picker (ds-a), plus the
    // "no reference" placeholder option — ds-b (active), ds-c (embedder), and
    // ds-d (unloaded) are excluded.
    const options = fixture.nativeElement.querySelectorAll('.domain-ref-select option');
    expect(options.length).toBe(2);
  });

  it('surfaces the backend message when the reference has no atlas', async () => {
    await setup();
    await fixture.whenStable();
    httpMock.expectOne('/api/find/stats').flush(mockStats);
    httpMock.expectOne('/api/datasets/registry/ds-a/domain-shift').flush(
      { message: 'Reference dataset has no coverage atlas; build it first' },
      { status: 400, statusText: 'Bad Request' },
    );
    httpMock.expectOne('/api/find/evidence-coverage').flush({
      available: false,
      n_items: 0,
      n_pos_labels: 0,
      n_neg_labels: 0,
      k: 1,
      alpha: 0.05,
      frac_unsupported: 0,
      expected_unsupported: 0.05,
      z_score: 0,
      median_support: 1,
      frac_low_trust: 0,
      median_trust: 1,
      unsupported: false,
    });
    await settleZoneless(fixture);

    const note = fixture.nativeElement.querySelector('.domain-note') as HTMLElement;
    expect(note).toBeTruthy();
    expect(note.textContent).toContain('coverage atlas');
    expect(fixture.nativeElement.querySelector('.domain-chip')).toBeFalsy();
  });
});
