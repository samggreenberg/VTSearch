import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { CombineDatasetsModalComponent } from './combine-datasets-modal.component';
import { DatasetRegistryEntry } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('CombineDatasetsModalComponent', () => {
  let component: CombineDatasetsModalComponent;
  let fixture: ComponentFixture<CombineDatasetsModalComponent>;
  let httpMock: HttpTestingController;

  const mockMediaTypes = {
    media_types: [
      { type_id: 'audio', name: 'Audio', icon: '🎵' },
      { type_id: 'image', name: 'Image', icon: '🖼' },
    ],
  };

  const ds = (id: string, name: string, type: string, num: number, pkl: string): DatasetRegistryEntry => ({
    id,
    name,
    media_type: type,
    num_items: num,
    pkl_path: pkl,
  });

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CombineDatasetsModalComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(CombineDatasetsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  // Set the datasets input, settle to run ngOnInit (builds rows + issues the
  // media-types GET), then flush it.
  async function init(datasets: DatasetRegistryEntry[]): Promise<void> {
    fixture.componentRef.setInput('datasets', datasets);
    await settleZoneless(fixture);
    httpMock.expectOne('/api/media-types').flush(mockMediaTypes);
    await settleZoneless(fixture);
  }

  it('builds rows from input datasets, filtering out entries with no pkl_path', async () => {
    await init([
      ds('a', 'Alpha', 'audio', 10, '/data/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, ''),
      ds('c', 'Charlie', 'audio', 30, '/data/c.pkl'),
    ]);

    expect(component.rows.length).toBe(2);
    expect(component.rows.map((r) => r.id)).toEqual(['a', 'c']);
    expect(component.totalItems).toBe(40);
  });

  it('canCombine is true only when ≥2 rows share a single media type', async () => {
    await init([
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, '/x/b.pkl'),
    ]);

    expect(component.canCombine).toBe(true);
    expect(component.disabledReason).toBe('');
  });

  it('canCombine is false when media types differ', async () => {
    await init([
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'image', 20, '/x/b.pkl'),
    ]);

    expect(component.canCombine).toBe(false);
    expect(component.disabledReason).toContain('share a media type');
  });

  it('canCombine is false when fewer than two rows remain after removal', async () => {
    await init([
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, '/x/b.pkl'),
    ]);

    component.removeRow('b');
    expect(component.rows.length).toBe(1);
    expect(component.canCombine).toBe(false);
    expect(component.disabledReason).toContain('at least two');
  });

  it('submit posts the pkl paths to /api/dataset/combine and emits combineStarted', async () => {
    await init([
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, '/x/b.pkl'),
    ]);

    let started = false;
    component.combineStarted.subscribe(() => { started = true; });

    component.submit();
    const req = httpMock.expectOne('/api/dataset/combine');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ datasets: ['/x/a.pkl', '/x/b.pkl'], name: 'Alpha + Bravo' });
    req.flush({ ok: true });

    expect(started).toBe(true);
    expect(component.submitting()).toBe(false);
  });

  it('submit surfaces backend errors without emitting combineStarted', async () => {
    await init([
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, '/x/b.pkl'),
    ]);

    let started = false;
    component.combineStarted.subscribe(() => { started = true; });

    component.submit();
    const req = httpMock.expectOne('/api/dataset/combine');
    req.flush({ error: 'boom' }, { status: 500, statusText: 'Server Error' });

    expect(started).toBe(false);
    expect(component.error()).toBe('boom');
    expect(component.submitting()).toBe(false);
  });

  it('submit is a no-op when canCombine is false', async () => {
    await init([ds('a', 'Alpha', 'audio', 10, '/x/a.pkl')]);

    component.submit();
    httpMock.expectNone('/api/dataset/combine');
  });
});
