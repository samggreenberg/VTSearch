import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { DatasetStatsModalComponent } from './dataset-stats-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('DatasetStatsModalComponent', () => {
  let component: DatasetStatsModalComponent;
  let fixture: ComponentFixture<DatasetStatsModalComponent>;
  let httpMock: HttpTestingController;

  const mockStats = {
    name: 'My Dataset',
    media_type: 'audio',
    num_items: 100,
    num_dupes: 3,
    created_at: 1700000075,
    expires_at: null,
    created_by: 'alice',
    readers: ['bob', 'carol'],
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
      providers: [...provideHttpTesting()],
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

  it('opens the Duplicates child modal from the Duplicate-groups View button', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/stats').flush(mockStats);
    await settleZoneless(fixture);

    const viewBtn = fixture.nativeElement.querySelector('.view-dupes-btn') as HTMLButtonElement;
    expect(viewBtn).toBeTruthy();
    viewBtn.click();
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('vt-duplicates-modal')).toBeTruthy();
    // The child modal fetches the duplicate sets for the same dataset.
    httpMock.expectOne('/api/datasets/registry/ds1/duplicates').flush({ duplicate_sets: [] });
    await settleZoneless(fixture);
  });

  it('hides the View button when the dataset has no duplicates', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/stats').flush({ ...mockStats, num_dupes: 0 });
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.view-dupes-btn')).toBeFalsy();
  });

  // Issue #2698: the Stats window covers the Dashboard grid, so it has to
  // carry the grid's own columns (Type / # Items / Created / Age-Off /
  // Creator / Readers) rather than force the user to dismiss it to look.
  it('renders every Dashboard grid column', async () => {
    fixture.componentRef.setInput('isDefaultLogin', false);
    fixture.componentRef.setInput('serverSetsAgeOff', true);
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/stats').flush(mockStats);
    await settleZoneless(fixture);

    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('Type');
    expect(text).toContain('Audio'); // capitalized media_type
    expect(text).toContain('Created');
    expect(text).toContain('Age-Off');
    expect(text).toContain('Never'); // expires_at === null
    expect(text).toContain('Creator');
    expect(text).toContain('alice');
    expect(text).toContain('Readers');
    expect(text).toContain('bob, carol');
  });

  it('hides Creator/Readers under the default login, as the grid does', async () => {
    fixture.componentRef.setInput('isDefaultLogin', true);
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/stats').flush(mockStats);
    await settleZoneless(fixture);

    const text = fixture.nativeElement.textContent as string;
    expect(text).not.toContain('Creator');
    expect(text).not.toContain('Readers');
  });

  it('hides Age-Off when the server stamps no expiry and the dataset has none', async () => {
    fixture.componentRef.setInput('serverSetsAgeOff', false);
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/stats').flush(mockStats);
    await settleZoneless(fixture);

    expect(fixture.nativeElement.textContent as string).not.toContain('Age-Off');
  });

  // A stamped expiry outlives the server setting being turned back off;
  // hiding a real death date would be a lie, so the row survives.
  it('shows Age-Off for a stamped dataset even with the setting off', async () => {
    fixture.componentRef.setInput('serverSetsAgeOff', false);
    await fixture.whenStable();
    httpMock
      .expectOne('/api/datasets/registry/ds1/stats')
      .flush({ ...mockStats, expires_at: 1800000000 });
    await settleZoneless(fixture);

    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('Age-Off');
    expect(text).not.toContain('Never');
  });

  it('dots real file types but passes the unknown sentinel through verbatim', async () => {
    await fixture.whenStable();
    httpMock
      .expectOne('/api/datasets/registry/ds1/stats')
      .flush({ ...mockStats, file_type_counts: { jpg: 400, '(unknown)': 37 } });
    await settleZoneless(fixture);

    expect(component.fileTypes.map((ft) => ft.label)).toEqual(['.jpg', '(unknown)']);
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('.jpg');
    expect(text).not.toContain('.(unknown)');
  });

  it('repaints the error text on a failed load (zoneless canary)', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/stats').flush(
      { message: 'gone' },
      { status: 404, statusText: 'Not Found' },
    );
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('gone');
  });
});
