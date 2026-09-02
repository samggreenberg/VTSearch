import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { FindStatsModalComponent } from './find-stats-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { ActiveContextService } from '../../../services/active-context.service';
import type { DatasetRegistryEntry } from '../../../models/api.models';

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
    precision: 0.85,
    inclusion: 0,
    sweep: [
      { inclusion: -10, false_pos: 1, false_neg: 9 },
      { inclusion: 0, false_pos: 5, false_neg: 5 },
      { inclusion: 10, false_pos: 9, false_neg: 1 },
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
    expect(fixture.nativeElement.querySelector('.fpfn-chart')).toBeTruthy();
    expect(fixture.nativeElement.textContent).toContain('90%'); // agreement rate
  });

  it('repaints the error text on a failed load (zoneless canary)', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/find/stats').flush(
      { error: 'no find run' },
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
    precision: 0.85,
    inclusion: 0,
    sweep: [{ inclusion: 0, false_pos: 5, false_neg: 5 }],
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
