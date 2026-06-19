import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { DatasetStatsModalComponent } from './dataset-stats-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('DatasetStatsModalComponent', () => {
  let component: DatasetStatsModalComponent;
  let fixture: ComponentFixture<DatasetStatsModalComponent>;
  let httpMock: HttpTestingController;

  const mockStats = {
    num_items: 100,
    num_dupes: 3,
    ingest_started_at: 1700000000,
    ingest_finished_at: 1700000075,
    clipper: 'whole',
    embedder: 'clap',
    file_type_counts: { wav: 80, mp3: 20 },
    source: { importer: 'server_folder', params: { path: '/data', media_type: 'audio' } },
  };

  beforeEach(async () => {
    await configureZoneless({
      imports: [DatasetStatsModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DatasetStatsModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('datasetId', 'ds1');
    fixture.componentRef.setInput('datasetName', 'My Dataset');
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/stats').flush(mockStats);
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

    httpMock.expectOne('/api/datasets/registry/ds1/stats').flush(mockStats);
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.loading-text')).toBeFalsy();
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('server_folder'); // importerName
    expect(text).toContain('1m 15s'); // duration getter
  });

  it('repaints the error text on a failed load (zoneless canary)', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/stats').flush(
      { error: 'gone' },
      { status: 404, statusText: 'Not Found' },
    );
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('gone');
  });
});
