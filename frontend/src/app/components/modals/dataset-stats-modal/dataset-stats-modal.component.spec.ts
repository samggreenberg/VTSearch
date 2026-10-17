import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { Subject } from 'rxjs';
import { DatasetStatsModalComponent } from './dataset-stats-modal.component';
import type { LoadingTask } from '../../../models/api.models';
import { ProgressEventsService } from '../../../services/progress-events.service';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('DatasetStatsModalComponent', () => {
  let component: DatasetStatsModalComponent;
  let fixture: ComponentFixture<DatasetStatsModalComponent>;
  let httpMock: HttpTestingController;
  let loadingTasks$: Subject<LoadingTask[]>;

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
    coverage_atlas: null,
  };

  beforeEach(async () => {
    loadingTasks$ = new Subject<LoadingTask[]>();
    await configureZoneless({
      imports: [DatasetStatsModalComponent],
      providers: [...provideHttpTesting(), { provide: ProgressEventsService, useValue: { loadingTasks$ } }],
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

  // Issue #4995: a loaded dataset past the auto-build threshold has no
  // coverage atlas, and this row is the only way to build one from the GUI.
  describe('Coverage atlas row', () => {
    const atlas = (built: boolean, building = false) => ({ built, building, auto_build_threshold: 50000 });
    const statsUrl = '/api/datasets/registry/ds1/stats';
    const buildUrl = '/api/datasets/registry/ds1/coverage-atlas';
    const atlasTask = (status: string): LoadingTask =>
      ({ task_id: '_atlas_ds1', dataset_id: 'ds1', status, message: '', current: 0, total: 0, name: 'ds', created_at: 0 }) as LoadingTask;

    async function open(stats: object): Promise<void> {
      await fixture.whenStable();
      httpMock.expectOne(statsUrl).flush({ ...mockStats, ...stats });
      await settleZoneless(fixture);
    }

    const buildBtn = () => fixture.nativeElement.querySelector('.build-atlas-btn') as HTMLButtonElement | null;
    const text = () => fixture.nativeElement.textContent as string;

    it('is hidden when the dataset has an atlas', async () => {
      await open({ coverage_atlas: atlas(true) });
      expect(buildBtn()).toBeNull();
      expect(text()).not.toContain('Coverage atlas');
    });

    it('is hidden when the dataset is not loaded', async () => {
      await open({ coverage_atlas: null });
      expect(buildBtn()).toBeNull();
      expect(text()).not.toContain('Coverage atlas');
    });

    it('shows "Not built" with the threshold reason for a large dataset', async () => {
      await open({ num_items: 60000, coverage_atlas: atlas(false) });
      expect(text()).toContain('Coverage atlas');
      expect(text()).toContain('Not built');
      expect(text()).toContain('over 50,000 items');
      expect(buildBtn()!.disabled).toBe(false);
    });

    it('leaves the threshold reason off a dataset under it', async () => {
      // A small dataset with no atlas is one whose build was cancelled.
      await open({ coverage_atlas: atlas(false) });
      expect(text()).toContain('Not built');
      expect(text()).not.toContain('items skip it');
    });

    it('issues exactly one POST on a click and disables the button', async () => {
      await open({ num_items: 60000, coverage_atlas: atlas(false) });
      buildBtn()!.click();
      buildBtn()!.click();
      await settleZoneless(fixture);

      const posts = httpMock.match(buildUrl);
      expect(posts.length).toBe(1);
      expect(posts[0].request.method).toBe('POST');
      expect(buildBtn()!.disabled).toBe(true);
      expect(text()).toContain('Building…');

      // The task row running is no reason to re-read the stats.
      loadingTasks$.next([atlasTask('loading')]);
      posts[0].flush({ ok: true, message: 'started', task_id: '_atlas_ds1' });
      await settleZoneless(fixture);
      httpMock.expectNone(statsUrl);
      expect(buildBtn()!.disabled).toBe(true);
    });

    it('re-reads the stats when the build ends, and hides the row once built', async () => {
      await open({ num_items: 60000, coverage_atlas: atlas(false) });
      buildBtn()!.click();
      await settleZoneless(fixture);
      httpMock.expectOne(buildUrl).flush({ ok: true, message: 'started', task_id: '_atlas_ds1' });
      // The POST answered before the channel showed the task: the stats say
      // the build is still on, so the window keeps waiting.
      httpMock.expectOne(statsUrl).flush({ ...mockStats, num_items: 60000, coverage_atlas: atlas(false, true) });
      await settleZoneless(fixture);
      expect(buildBtn()!.disabled).toBe(true);

      loadingTasks$.next([atlasTask('loading')]);
      await settleZoneless(fixture);
      httpMock.expectNone(statsUrl);

      loadingTasks$.next([atlasTask('idle')]);
      await settleZoneless(fixture);
      httpMock.expectOne(statsUrl).flush({ ...mockStats, num_items: 60000, coverage_atlas: atlas(true) });
      await settleZoneless(fixture);
      expect(buildBtn()).toBeNull();
      expect(text()).not.toContain('Coverage atlas');
    });

    it('brings the button back when the build is cancelled', async () => {
      await open({ num_items: 60000, coverage_atlas: atlas(false, true) });
      expect(text()).toContain('Building…');
      expect(buildBtn()!.disabled).toBe(true);

      loadingTasks$.next([{ ...atlasTask('idle'), error: 'Cancelled' } as LoadingTask]);
      await settleZoneless(fixture);
      httpMock.expectOne(statsUrl).flush({ ...mockStats, num_items: 60000, coverage_atlas: atlas(false) });
      await settleZoneless(fixture);
      expect(text()).toContain('Not built');
      expect(buildBtn()!.disabled).toBe(false);
    });

    it('re-enables the button when the build request fails', async () => {
      await open({ coverage_atlas: atlas(false) });
      buildBtn()!.click();
      await settleZoneless(fixture);
      httpMock.expectOne(buildUrl).flush({ message: 'not loaded' }, { status: 400, statusText: 'Bad Request' });
      await settleZoneless(fixture);
      expect(buildBtn()!.disabled).toBe(false);
    });
  });
});
