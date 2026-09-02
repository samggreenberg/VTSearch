import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { CombineDatasetsModalComponent } from './combine-datasets-modal.component';
import { DatasetRegistryEntry } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

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

  const mockEmbedders = [
    { name: 'siglip', display_name: 'SigLIP' },
    { name: 'clip', display_name: 'CLIP' },
    { name: 'dinov3_patch', display_name: 'DINOv3 Patch' },
  ];

  const ds = (
    id: string,
    name: string,
    type: string,
    num: number,
    pkl: string,
    embeddersByType?: Record<string, string>,
  ): DatasetRegistryEntry => ({
    id,
    name,
    media_type: type,
    num_items: num,
    pkl_path: pkl,
    embedders_by_type: embeddersByType,
  });

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CombineDatasetsModalComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
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
    httpMock.expectOne((r) => r.url === '/api/embedders').flush({ embedders: mockEmbedders });
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

    let emitted: { taskId: string; numSources: number; totalItems: number } | undefined;
    component.combineStarted.subscribe((info) => { emitted = info; });

    component.submit();
    const req = httpMock.expectOne('/api/dataset/combine');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ datasets: ['/x/a.pkl', '/x/b.pkl'], name: 'Alpha + Bravo' });
    req.flush({ ok: true, message: 'Combining datasets...', task_id: 't-combine' });

    // The emitted payload carries the task id plus the pre-dedup source
    // counts the dashboard needs for its summary toast.
    expect(emitted).toEqual({ taskId: 't-combine', numSources: 2, totalItems: 30 });
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
    req.flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });

    expect(started).toBe(false);
    expect(component.error()).toBe('boom');
    expect(component.submitting()).toBe(false);
  });

  it('submit is a no-op when canCombine is false', async () => {
    await init([ds('a', 'Alpha', 'audio', 10, '/x/a.pkl')]);

    component.submit();
    httpMock.expectNone('/api/dataset/combine');
  });

  it('detects no conflict when datasets share the same embedder per type', async () => {
    await init([
      ds('a', 'Alpha', 'image', 10, '/x/a.pkl', { semantic: 'siglip' }),
      ds('b', 'Bravo', 'image', 20, '/x/b.pkl', { semantic: 'siglip' }),
    ]);

    expect(component.hasConflicts).toBe(false);
    expect(component.canCombine).toBe(true);
  });

  it('flags a name-clash conflict and gates Combine until resolved', async () => {
    await init([
      ds('a', 'Alpha', 'image', 10, '/x/a.pkl', { semantic: 'siglip' }),
      ds('b', 'Bravo', 'image', 20, '/x/b.pkl', { semantic: 'clip' }),
    ]);

    expect(component.hasConflicts).toBe(true);
    const conflict = component.conflicts[0];
    expect(conflict.type).toBe('semantic');
    expect(conflict.options).toEqual(['siglip', 'clip']);
    // Unresolved → cannot combine.
    expect(component.canCombine).toBe(false);
    expect(component.disabledReason).toContain('Resolve each embedder conflict');

    component.setResolution('semantic', 'reembed:clip');
    expect(component.canCombine).toBe(true);
  });

  it('flags partial coverage (some datasets lack a type) as a conflict', async () => {
    await init([
      ds('a', 'Alpha', 'image', 10, '/x/a.pkl', { semantic: 'siglip', patch_semantic: 'dinov3_patch' }),
      ds('b', 'Bravo', 'image', 20, '/x/b.pkl', { semantic: 'siglip' }),
    ]);

    const patch = component.conflicts.find((c) => c.type === 'patch_semantic');
    expect(patch).toBeTruthy();
    expect(patch!.partial).toBe(true);
    expect(patch!.options).toEqual(['dinov3_patch']);
    // Semantic agrees → not a conflict.
    expect(component.conflicts.some((c) => c.type === 'semantic')).toBe(false);
  });

  it('blocks dropping every conflicted embedder', async () => {
    await init([
      ds('a', 'Alpha', 'image', 10, '/x/a.pkl', { semantic: 'siglip' }),
      ds('b', 'Bravo', 'image', 20, '/x/b.pkl', { semantic: 'clip' }),
    ]);

    component.setResolution('semantic', 'drop');
    expect(component.canCombine).toBe(false);
    expect(component.disabledReason).toContain('At least one embedder');
  });

  it('submits resolutions in the combine body', async () => {
    await init([
      ds('a', 'Alpha', 'image', 10, '/x/a.pkl', { semantic: 'siglip', patch_semantic: 'dinov3_patch' }),
      ds('b', 'Bravo', 'image', 20, '/x/b.pkl', { semantic: 'clip' }),
    ]);

    component.setResolution('semantic', 'reembed:siglip');
    component.setResolution('patch_semantic', 'drop');

    component.submit();
    const req = httpMock.expectOne('/api/dataset/combine');
    expect(req.request.body).toEqual({
      datasets: ['/x/a.pkl', '/x/b.pkl'],
      name: 'Alpha + Bravo',
      resolutions: {
        semantic: { action: 'reembed', embedder: 'siglip' },
        patch_semantic: { action: 'drop' },
      },
    });
    req.flush({ ok: true, message: 'Combining datasets...', task_id: 't' });
  });

  it('prunes resolution choices for conflicts removed by removeRow', async () => {
    await init([
      ds('a', 'Alpha', 'image', 10, '/x/a.pkl', { semantic: 'siglip' }),
      ds('b', 'Bravo', 'image', 20, '/x/b.pkl', { semantic: 'clip' }),
      ds('c', 'Charlie', 'image', 30, '/x/c.pkl', { semantic: 'siglip' }),
    ]);

    expect(component.hasConflicts).toBe(true);
    component.setResolution('semantic', 'reembed:clip');
    // Removing the odd-one-out clears the conflict; its choice is pruned.
    component.removeRow('b');
    expect(component.hasConflicts).toBe(false);
    expect(component.resolutionChoices()['semantic']).toBeUndefined();
    expect(component.canCombine).toBe(true);
  });
});
