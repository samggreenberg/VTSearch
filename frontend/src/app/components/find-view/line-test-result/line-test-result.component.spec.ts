import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { LineTestResultComponent } from './line-test-result.component';
import { PairScopeService } from '../../../services/pair-scope.service';
import { ActiveContextService } from '../../../services/active-context.service';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { estimate, wireDone, wireLineTest, wireTest } from '../../../testing/line-test';
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

  it('at Done shows the verdict with its exits, reads the checks, and lets the Threshold lean on the presets', async () => {
    const presets = [
      { beta: 0.25, count: 31, precision: estimate(0.9, 0.75, 1), recall: estimate(0.3, 0.2, 0.4), fbeta: estimate(0.5, 0.4, 0.6), found: 'about a quarter of them found' },
      { beta: 1, count: 64, precision: estimate(0.75, 0.55, 0.95), recall: estimate(0.5, 0.3, 0.7), fbeta: estimate(0.6, 0.45, 0.78), found: 'about half of them found' },
      { beta: 4, count: 128, precision: estimate(0.5, 0.35, 0.65), recall: estimate(0.8, 0.6, 0.95), fbeta: estimate(0.7, 0.55, 0.85), found: 'about three quarters of them found' },
    ];
    const el = await show(wireLineTest(wireDone(), { presets }));
    flushChecks();
    await settleZoneless(fixture);

    expect(el.querySelector('.verdict')!.textContent).toContain('likely 55–95% of them are right, with about half of them found');
    const exits = Array.from(el.querySelectorAll('.exit-btn')).map((b) => b.textContent!.trim());
    expect(exits).toEqual(['Move to AutoRun', 'Add Corrections and retrain']);

    const rows = Array.from(el.querySelectorAll('.presets tbody tr')) as HTMLElement[];
    expect(rows.length).toBe(3);
    expect(rows.map((r) => r.querySelector('.preset-btn')!.textContent!.trim())).toEqual(['False Positives', 'Between', 'False Negatives']);
    expect(rows[1].classList.contains('current')).toBe(true);
    expect((rows[1].querySelector('.preset-btn') as HTMLButtonElement).disabled).toBe(true);
    expect(rows[0].textContent).toContain('128');
    expect(rows[0].textContent).toContain('35–65%');

    const leaned: number[] = [];
    fixture.componentInstance.lean.subscribe((b) => leaned.push(b));
    (rows[0].querySelector('.preset-btn') as HTMLButtonElement).click();
    expect(leaned).toEqual([4]);

    const moved: number[] = [];
    fixture.componentInstance.moveToAutoRun.subscribe(() => moved.push(1));
    (el.querySelector('.exit-btn.btn--primary') as HTMLButtonElement).click();
    expect(moved.length).toBe(1);

    // The Stats modal's 2×2, under the result.
    expect(el.querySelector('.confusion')).not.toBeNull();
    expect(el.textContent).toContain('45 checked by hand');
  });

  it('says the result is stale after corrections, and that the line moved, offering a new test', async () => {
    let el = await show(wireLineTest(wireDone(), { stale: true }));
    flushChecks();
    await settleZoneless(fixture);
    expect(el.querySelector('.stale-note')!.textContent).toContain('Out of date');

    el = await show(wireLineTest(wireDone(), { moved: true, line_count: 128 }));
    expect(el.querySelector('.stale-note')!.textContent).toContain('measured the top 64');
    const again: number[] = [];
    fixture.componentInstance.testAgain.subscribe(() => again.push(1));
    (el.querySelector('.stale-note button') as HTMLButtonElement).click();
    expect(again.length).toBe(1);
  });

  it('says when a test resumed from the picks the detector keeps (#4526)', async () => {
    let el = await show(wireLineTest(wireDone()));
    flushChecks();
    await settleZoneless(fixture);
    expect(el.querySelector('.kept-note')).toBeNull();

    const keptAt = new Date(2026, 9, 5, 12, 0).getTime() / 1000;
    el = await show(wireLineTest(wireDone({ kept_at: keptAt })));
    expect(el.querySelector('.kept-note')!.textContent).toContain('Kept from your test of 2026-10-05');
    expect(el.querySelector('.kept-note')!.textContent).toContain('its 45 picks still hold');
  });

  it('says there is nothing to test', async () => {
    const el = await show(wireLineTest(wireTest({ phase: 'nothing', picks: [] })));
    expect(el.querySelector('.section-title')!.textContent).toContain('Nothing to test');
    expect(el.querySelector('.headline-value')).toBeNull();
  });
});
