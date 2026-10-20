import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { LineTestResultComponent } from './line-test-result.component';
import { PairScopeService } from '../../../services/pair-scope.service';
import { ActiveContextService } from '../../../services/active-context.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { wireDone, wireLineTest, wireTest } from '../../../testing/line-test';
import { wireBalance } from '../../../testing/line-balance';
import type { LineTestResponse } from '../../../generated/api-client/models/line-test-response';

const STATS = {
  stale: false,
  total_good: 64,
  total_bad: 136,
  verified_count: 45,
  confirmed_good: 30,
  rescued_false_neg: 4,
  culled_false_pos: 5,
  confirmed_bad: 6,
  agreements: 190,
  corrections: 10,
  agreement_rate: 0.95,
  verified_precision: 0.86,
  verified_called_good: 35,
  verified_kept_good: 30,
  threshold: 0.68,
  balance: wireBalance('unchecked', { count: 64 }),
  n_scored: 200,
  n_returned: 64,
  precision_curve: [],
};

const NO_EVIDENCE = {
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
  baseline: 'none',
  n_baseline: 0,
  baseline_dataset_id: '',
  baseline_dataset_name: '',
  frac_beyond: null,
  z_beyond: null,
  unsupported: false,
};

describe('LineTestResultComponent (#4524)', () => {
  let fixture: ComponentFixture<LineTestResultComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [LineTestResultComponent],
      providers: [...provideHttpTesting(), PairScopeService],
    }).compileComponents();
    TestBed.inject(ActiveContextService).setActivePair('ds1', 'det1');
    fixture = TestBed.createComponent(LineTestResultComponent);
    httpMock = TestBed.inject(HttpTestingController);
    await settleZoneless(fixture);
  });

  afterEach(() => {
    fixture.destroy();
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush([]);
    });
  });

  async function show(response: LineTestResponse | null, scoring = false): Promise<HTMLElement> {
    fixture.componentRef.setInput('response', response);
    fixture.componentRef.setInput('scoring', scoring);
    await settleZoneless(fixture);
    return fixture.nativeElement as HTMLElement;
  }

  /** The reads a verdict triggers: the checks' 2×2 and the trust chips. */
  function flushChecks(): void {
    httpMock.match('/api/find/stats').forEach((req) => req.flush(STATS));
    httpMock.match('/api/find/evidence-coverage').forEach((req) => req.flush(NO_EVIDENCE));
  }

  it('waits on the line while the pass runs', async () => {
    const el = await show(null, true);
    expect(el.querySelector('.result-note')!.textContent).toContain('once the line is drawn');
    httpMock.expectNone('/api/find/stats');
  });

  it('shows the ranges with their lights and the F-beta headline as a running test forms', async () => {
    const el = await show(wireLineTest(wireTest()));
    expect(el.querySelector('.headline-value')!.textContent!.trim()).toBe('0.60 (0.45–0.78)');
    const rows = Array.from(el.querySelectorAll('.range-row')) as HTMLElement[];
    expect(rows[0].textContent).toContain('Right');
    expect(rows[0].querySelector('dd')!.textContent!.trim()).toBe('55–95%');
    expect(rows[0].dataset['light']).toBe('red');
    expect(rows[1].querySelector('dd')!.textContent).toContain('about half of them found (30–70%)');
    expect(el.querySelector('.result-note')!.textContent).toContain('12 matches it likely holds are the detector\'s own estimate');
    // No verdict and no exits yet.
    expect(el.querySelector('.exits')).toBeNull();
    httpMock.expectNone('/api/find/stats');
  });

  it('draws the band-resolution curve: one point and one range bar per band edge, and the line', async () => {
    const el = await show(wireLineTest(wireTest()));
    expect(el.querySelectorAll('.precision-chart .dot-verified').length).toBe(6);
    expect(el.querySelectorAll('.precision-chart .range-bar').length).toBe(6);
    expect(el.querySelector('.precision-chart .current')).not.toBeNull();
    expect(el.querySelector('.chart-legend')!.textContent).toContain('Line: the top 64');
    expect(el.querySelector('.chart-readout')!.textContent).toContain('Top 64: likely 55–95% right');
    expect(el.textContent).not.toContain('Checked by you');
  });

  it('lists the picks by band once there are any', async () => {
    const t = wireTest();
    t.bands[3] = { ...t.bands[3], labelled: 5, right: 4, range: { lo: 0.36, hi: 0.98, labelled: 5, right: 4 } };
    const el = await show(wireLineTest(t));
    const rows = Array.from(el.querySelectorAll('.bands tbody tr')) as HTMLElement[];
    expect(rows.length).toBe(1);
    expect(rows[0].textContent).toContain('▲ 33–64');
    expect(rows[0].textContent).toContain('36–98%');
  });

  it('with no class model reads Found in words only and says it is unmeasured below the picks (#4542)', async () => {
    const t = wireDone({ class_model: false });
    t.bands[4] = { ...t.bands[4], labelled: 5, right: 0, range: { lo: 0, hi: 0.45, labelled: 5, right: 0 } };
    t.estimates = { ...t.estimates!, tail_positives: 0 };
    const el = await show(wireLineTest(t));
    flushChecks();
    await settleZoneless(fixture);

    const found = el.querySelectorAll('.range-row')[1] as HTMLElement;
    expect(found.querySelector('dd')!.textContent!.trim()).toBe('about half of them found');
    expect(found.querySelector('dt')!.getAttribute('title')).toContain('no count of its own');
    const notes = Array.from(el.querySelectorAll('.result-note')).map((n) => n.textContent!).join(' ');
    expect(notes).toContain('no class model, so nothing estimates the matches below the top 72');
    expect(notes).not.toContain("detector's own estimate");
  });

  it('at Done reads the checks, and leaves the verdict and its exits to the centre pane (#5100)', async () => {
    const el = await show(wireLineTest(wireDone(), { stale: true }));
    flushChecks();
    await settleZoneless(fixture);

    expect(el.querySelector('.headline-value')).not.toBeNull();
    expect(el.querySelector('.verdict')).toBeNull();
    expect(el.querySelector('.exit-btn')).toBeNull();
    expect(el.querySelector('.presets')).toBeNull();
    // The out-of-date note stands over the verdict instead.
    expect(el.querySelector('.stale-note')).toBeNull();

    // The Stats modal's 2×2, under the result.
    expect(el.querySelector('.confusion')).not.toBeNull();
    expect(el.textContent).toContain('45 checked by hand');
  });

  it('says a running test is stale after corrections', async () => {
    const el = await show(wireLineTest(wireTest(), { stale: true }));
    expect(el.querySelector('.stale-note')!.textContent).toContain('Out of date');
  });

  it('says when a test resumed from the picks the detector keeps (#4526)', async () => {
    let el = await show(wireLineTest(wireDone()));
    flushChecks();
    await settleZoneless(fixture);
    expect(el.querySelector('.kept-note')).toBeNull();

    const keptAt = new Date(2026, 9, 5, 12, 0).getTime() / 1000;
    el = await show(wireLineTest(wireDone({ kept_at: keptAt })));
    expect(el.querySelector('.kept-note')!.textContent).toContain('Kept from your test of 2026-10-05');
    expect(el.querySelector('.kept-note')!.textContent).toContain('its 65 picks still hold');
  });

  describe('Training-domain overlap (#4961)', () => {
    const SHIFT = {
      reference_dataset_id: 'photos',
      n_items: 200,
      alpha: 0.05,
      frac_atypical: 0.31,
      expected_atypical: 0.05,
      z_score: 9,
      median_pvalue: 0.2,
      shifted: true,
    };

    /** Loaded datasets on the active one's embedder: 'drawings' first, the training set 'photos' second. */
    function loadRegistry(): void {
      TestBed.inject(DatasetStateService).refresh();
      httpMock.expectOne('/api/datasets/registry').flush({
        datasets: [
          { id: 'ds1', name: 'Screenshots', media_type: 'image', embedder: 'clip', loaded: true },
          { id: 'drawings', name: 'Drawings', media_type: 'image', embedder: 'clip', loaded: true },
          { id: 'photos', name: 'Photos', media_type: 'image', embedder: 'clip', loaded: true },
        ],
      });
      httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
    }

    function compareAgainst(el: HTMLElement): string {
      return (el.querySelector('.domain-ref-select') as HTMLSelectElement).value;
    }

    it('compares against the dataset holding the detector\'s examples, and reuses its report', async () => {
      loadRegistry();
      const el = await show(wireLineTest(wireDone()));
      flushChecks();
      httpMock.expectOne('/api/find/training-domain').flush({
        reference_dataset_id: 'photos',
        reference_dataset_name: 'Photos',
        labels_total: 52,
        labels_in_reference: 40,
        labels_in_active: 0,
        reason: '',
        domain_shift: SHIFT,
      });
      await settleZoneless(fixture);

      expect(compareAgainst(el)).toBe('photos');
      expect(el.querySelector('.domain-chip')!.textContent).toContain('31%');
      expect(el.querySelector('.domain-chip')!.textContent).toContain('vs Photos');
      expect(el.querySelector('.domain-chip')!.textContent).toContain('likely domain shift');
      // The report came with the reference: the picker does not ask for it again.
      httpMock.expectNone('/api/datasets/registry/photos/domain-shift');
    });

    it('falls back to the first candidate when no loaded dataset holds the examples', async () => {
      loadRegistry();
      const el = await show(wireLineTest(wireDone()));
      flushChecks();
      httpMock.expectOne('/api/find/training-domain').flush({
        reference_dataset_id: '',
        reference_dataset_name: '',
        labels_total: 52,
        labels_in_reference: 0,
        labels_in_active: 0,
        reason: "None of the loaded datasets holds the detector's labelled examples",
        domain_shift: null,
      });
      httpMock.expectOne('/api/datasets/registry/drawings/domain-shift').flush({ ...SHIFT, reference_dataset_id: 'drawings', shifted: false, frac_atypical: 0.03 });
      await settleZoneless(fixture);

      expect(compareAgainst(el)).toBe('drawings');
      expect(el.querySelector('.domain-chip')!.textContent).toContain('largely in-domain');
    });
  });

  describe('the Evidence coverage chip (#5089)', () => {
    const EVIDENCE = {
      ...NO_EVIDENCE,
      available: true,
      n_items: 4000,
      n_pos_labels: 30,
      n_neg_labels: 120,
      frac_unsupported: 0.18,
      z_score: 37,
      median_support: 0.4,
    };

    async function chip(report: object): Promise<HTMLElement> {
      const el = await show(wireLineTest(wireDone()));
      httpMock.match('/api/find/stats').forEach((req) => req.flush(STATS));
      httpMock.expectOne('/api/find/evidence-coverage').flush(report);
      await settleZoneless(fixture);
      return el.querySelector('.evidence-chip') as HTMLElement;
    }

    it('reads the share beyond the training baseline, with its verdict', async () => {
      const el = await chip({
        ...EVIDENCE,
        baseline: 'training',
        n_baseline: 11000,
        baseline_dataset_id: 'photos',
        baseline_dataset_name: 'Photos',
        frac_beyond: 0.62,
        z_beyond: 120,
        unsupported: true,
      });
      expect(el.textContent).toContain('62%');
      expect(el.textContent).toContain('than 95% of Photos');
      expect(el.textContent).toContain('often calling without support');
      expect(el.classList).toContain('shifted');
    });

    it('in-domain, the share sits at alpha and the chip says so', async () => {
      const el = await chip({
        ...EVIDENCE,
        baseline: 'training',
        n_baseline: 11000,
        baseline_dataset_name: 'Photos',
        frac_beyond: 0.05,
        z_beyond: 0.1,
      });
      expect(el.textContent).toContain('5%');
      expect(el.textContent).toContain('as well covered as its training data');
      expect(el.classList).not.toContain('shifted');
    });

    it('without a baseline, shows the share and draws no verdict', async () => {
      const el = await chip(EVIDENCE);
      expect(el.textContent).toContain('18%');
      expect(el.textContent).toContain('no verdict until you run Tune on the dataset the detector was trained on');
      expect(el.classList).not.toContain('shifted');
    });
  });

  it('says there is nothing to test', async () => {
    const el = await show(wireLineTest(wireTest({ phase: 'nothing', picks: [] })));
    expect(el.querySelector('.section-title')!.textContent).toContain('Nothing to test');
    expect(el.querySelector('.headline-value')).toBeNull();
  });
});
