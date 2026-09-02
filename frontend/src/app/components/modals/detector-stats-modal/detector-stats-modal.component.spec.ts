import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { DetectorStatsModalComponent } from './detector-stats-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('DetectorStatsModalComponent', () => {
  let component: DetectorStatsModalComponent;
  let fixture: ComponentFixture<DetectorStatsModalComponent>;
  let httpMock: HttpTestingController;

  const mockStats = {
    num_positive: 12,
    num_positive_resolved: 10,
    num_negative: 8,
    num_total: 20,
    active_dataset_name: 'My Dataset',
    media_type: 'audio',
    embedder: 'clap',
    clipper: 'whole',
    text_query: '',
    media_example: '',
    created_at: 1700000000,
    last_trained_at: 1700000100,
    created_by: 'sam',
    autofind: false,
    readers: [],
  };

  beforeEach(async () => {
    await configureZoneless({
      imports: [DetectorStatsModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DetectorStatsModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('detectorId', 'det1');
    fixture.componentRef.setInput('detectorName', 'My Detector');
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/detectors/registry/det1/stats').flush(mockStats);
    await settleZoneless(fixture);
    expect(component).toBeTruthy();
  });

  // Zoneless staleness canary: the stats land in an HTTP subscribe (an unpatched
  // callback) and repaint only because `loading`/`stats` are signals read in the
  // template. Flush the GET and assert the loaded table renders with no manual
  // `detectChanges`.
  it('repaints from loading to the loaded table (zoneless canary)', async () => {
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('.loading-text')).toBeTruthy();

    httpMock.expectOne('/api/detectors/registry/det1/stats').flush(mockStats);
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.loading-text')).toBeFalsy();
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('10 of 12 in "My Dataset"'); // resolvedSummary
  });

  it('repaints the error text on a failed load (zoneless canary)', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/detectors/registry/det1/stats').flush(
      { message: 'gone' },
      { status: 404, statusText: 'Not Found' },
    );
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('gone');
  });
});
