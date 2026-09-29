import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { FindStatsModalComponent } from './find-stats-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { ActiveContextService } from '../../../services/active-context.service';
import type { DatasetRegistryEntry } from '../../../models/api.models';
import { wireFloor } from '../../../testing/line-floor';

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
    floor: wireFloor('promised', { calibrationPositives: 14 }),
    n_scored: 1000,
    n_returned: 40,
    precision_curve: [
      { n_returned: 1, threshold: 0.99, checked: 0, checked_good: 0, verified_precision: null, estimated_precision: 0.95 },
      { n_returned: 10, threshold: 0.9, checked: 2, checked_good: 2, verified_precision: 1, estimated_precision: 0.9 },
      { n_returned: 40, threshold: 0.5, checked: 10, checked_good: 7, verified_precision: 0.7, estimated_precision: 0.62 },
      { n_returned: 1000, threshold: 0.01, checked: 12, checked_good: 7, verified_precision: 0.5833, estimated_precision: 0.05 },
    ],
    estimate_status: 'estimated',
    calibration_positives: 14,
    min_calibration_positives: 10,
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

    it('draws both curves, skipping the points each lacks', async () => {
      const el = await load();
      const est = el.querySelector('.line-estimate')!.getAttribute('points')!.trim().split(' ');
      const ver = el.querySelector('.line-verified')!.getAttribute('points')!.trim().split(' ');
      expect(est.length).toBe(4);
      expect(ver.length).toBe(3); // the top 1 has nothing checked
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

    it('marks the line and reads both precisions off it', async () => {
      const el = await load();
      expect(el.querySelector('.precision-chart .current')).toBeTruthy();
      const readout = el.querySelector('.chart-readout')!.textContent!.replace(/\s+/g, ' ');
      expect(readout).toContain('At the line (40 returned)');
      expect(readout).toContain('estimated at least 62%');
      expect(readout).toContain('checked 70%');
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
        'Top 1: estimated at least 95% · nothing checked',
      );
      svg.dispatchEvent(new MouseEvent('mouseleave'));
      await settleZoneless(fixture);
      expect(el.querySelector('.crosshair')).toBeFalsy();
    });

    it('explains a withheld estimate below the calibration gate', async () => {
      const el = await load({
        estimate_status: 'insufficient_evidence',
        calibration_positives: 4,
        precision_curve: mockStats.precision_curve.map((p) => ({ ...p, estimated_precision: null })),
      });
      expect(el.querySelector('.line-estimate')!.getAttribute('points')).toBe('');
      expect(el.querySelector('.chart-note')!.textContent).toContain('needs 10 Good votes');
      expect(el.querySelector('.chart-note')!.textContent).toContain('has 4');
      // The readout drops the missing estimate rather than printing a dash for it.
      const readout = el.querySelector('.chart-readout')!.textContent!.replace(/\s+/g, ' ');
      expect(readout).toContain('(40 returned): checked 70%');
      expect(readout).not.toContain('estimated');
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

    it('explains a detector with no calibration folds', async () => {
      const el = await load({ estimate_status: 'unavailable', calibration_positives: 0 });
      expect(el.textContent).toContain('too few votes to hold any out');
    });
  });

  describe('the precision floor on the chart (#4246)', () => {
    async function load(overrides: Record<string, unknown> = {}) {
      await fixture.whenStable();
      httpMock.expectOne('/api/find/stats').flush({ ...mockStats, ...overrides });
      httpMock.expectOne('/api/find/evidence-coverage').flush(mockEvidenceUnavailable);
      await settleZoneless(fixture);
      return fixture.nativeElement as HTMLElement;
    }

    const legend = (el: HTMLElement) => el.querySelector('.chart-legend')!.textContent!.replace(/\s+/g, ' ');

    it('draws the floor across the chart at X, and a promised line keeps it', async () => {
      const el = await load({ floor: wireFloor('promised', { minPrecision: 0.75 }) });
      const floor = el.querySelector('.precision-chart .floor')!;
      expect(Number(floor.getAttribute('y1'))).toBeCloseTo(component.yFor(0.75));
      expect(floor.getAttribute('y2')).toBe(floor.getAttribute('y1'));
      expect(el.querySelector('.precision-chart .current')!.classList).not.toContain('unpromised');
      expect(legend(el)).toContain('Floor: at least 75%');
      expect(legend(el)).toContain('Line: keeps the 75% floor');
      // The Inclusion stepper's legend is gone.
      expect(legend(el)).not.toContain('incl');
    });

    it('labels the line as the unpromised default cut when the floor is unreachable', async () => {
      const el = await load({ floor: wireFloor('unreachable', { minPrecision: 0.9, calibrationPositives: 20 }) });
      expect(el.querySelector('.precision-chart .floor')).toBeTruthy();
      expect(el.querySelector('.precision-chart .current')!.classList).toContain('unpromised');
      expect(el.querySelector('.swatch-current')!.classList).toContain('unpromised');
      expect(legend(el)).toContain('Line: the default cut, unpromised');
      expect(el.textContent).toContain('No cut reaches the 90% floor on this dataset');
    });

    it('says the line waits on evidence alongside the withheld estimate', async () => {
      const el = await load({
        estimate_status: 'insufficient_evidence',
        calibration_positives: 3,
        floor: wireFloor('insufficient_evidence'),
      });
      expect(el.querySelector('.precision-chart .current')!.classList).toContain('unpromised');
      const notes = Array.from(el.querySelectorAll('.chart-note')).map((n) => n.textContent!.replace(/\s+/g, ' '));
      expect(notes.some((n) => n.includes('has 3') && n.includes('Until then the line is the default cut'))).toBe(
        true,
      );
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
    floor: wireFloor('insufficient_evidence', { calibrationPositives: 0 }),
    n_scored: 100,
    n_returned: 10,
    precision_curve: [],
    estimate_status: 'unavailable',
    calibration_positives: 0,
    min_calibration_positives: 10,
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
