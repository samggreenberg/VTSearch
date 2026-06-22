import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { FindStatsModalComponent } from './find-stats-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

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

  beforeEach(async () => {
    await configureZoneless({
      imports: [FindStatsModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
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
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('no find run');
  });

  it('emits closed on close', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/find/stats').flush(mockStats);
    await settleZoneless(fixture);

    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
