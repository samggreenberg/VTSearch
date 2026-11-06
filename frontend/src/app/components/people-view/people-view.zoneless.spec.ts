import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpRequest } from '@angular/common/http';
import { HttpTestingController, TestRequest } from '@angular/common/http/testing';
import { ActivatedRoute, Router, convertToParamMap, provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { PeopleViewComponent } from './people-view.component';
import type { PeopleList } from '../../generated/api-client/models/people-list';
import type { Person } from '../../generated/api-client/models/person';
import type { SharedPhotos } from '../../generated/api-client/models/shared-photos';
import { ActiveContextService } from '../../services/active-context.service';
import { SourcePhotoService } from '../../services/source-photo.service';
import { ToastService } from '../../services/toast.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

// The reads here are rxResources, which keep a zoneless app unstable while a
// request is out, so each step drains with `settleResource` (a macrotask, then
// a tick) rather than `whenStable`, which would wait on the request answered.

const READY: PeopleList = {
  status: 'ready',
  partition: 'p1',
  people: [
    { id: 1, face_count: 3, photo_count: 2, representative: 10 },
    { id: 2, face_count: 2, photo_count: 2, representative: 20 },
    { id: 3, face_count: 1, photo_count: 1, representative: 30 },
  ],
  unplaced: 1,
  unplaced_ids: [40],
  cutoff: 0.44,
  compat: 'oss-arcface-mbf:1',
};

const PERSON_1: Person = {
  partition: 'p1',
  id: 1,
  representative: 10,
  face_ids: [10, 11, 12],
  photos: [
    { photo: 0, face_ids: [10] },
    { photo: 1, face_ids: [11] },
  ],
  appears_with: [{ id: 2, shared: 2, representative: 20 }],
};

const SHARED_1_2: SharedPhotos = {
  partition: 'p1',
  id: 1,
  other: 2,
  photos: [
    {
      photo: 0,
      source_dataset_id: 'photos',
      source_media_id: 5,
      faces: [
        { media_id: 10, box: [0.1, 0.1, 0.3, 0.4], person: 1 },
        { media_id: 20, box: [0.5, 0.1, 0.7, 0.4], person: 2 },
        { media_id: 30, box: [0.8, 0.1, 0.9, 0.3], person: 3 },
      ],
    },
  ],
};

/**
 * The People view (#5590): a Face dataset's people, a person's faces, who they
 * appear with and the photos they share, outlined by whose face is whose. Run
 * under the zoneless oracle, so every assertion reads the rendered DOM after
 * the app's own notification path, never a forced render.
 */
describe('PeopleViewComponent (zoneless)', () => {
  let fixture: ComponentFixture<PeopleViewComponent>;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    configureZoneless({
      imports: [PeopleViewComponent],
      providers: [
        ...provideHttpTesting(),
        provideRouter([]),
        { provide: ActivatedRoute, useValue: { paramMap: of(convertToParamMap({ datasetId: 'faces' })) } },
      ],
    });
    httpMock = TestBed.inject(HttpTestingController);
    TestBed.inject(ActiveContextService).setActivePair('faces', '');
    vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);
  });

  afterEach(() => {
    fixture?.destroy();
    drainOthers();
    httpMock.verify();
  });

  /** Answer whatever the page asked for that is not a People route (metadata
   *  batches, the media-type registry, settings) so `verify()` passes. */
  function drainOthers(): void {
    for (const req of httpMock.match((r) => !r.url.startsWith('/api/people'))) {
      if (!req.cancelled) req.flush(req.request.method === 'POST' ? [] : {});
    }
  }

  function text(): string {
    return (fixture.nativeElement.textContent ?? '').replace(/\s+/g, ' ');
  }

  function el<T extends Element = HTMLElement>(selector: string): T | null {
    return (fixture.nativeElement as HTMLElement).querySelector<T>(selector);
  }

  function all(selector: string): HTMLElement[] {
    return Array.from((fixture.nativeElement as HTMLElement).querySelectorAll<HTMLElement>(selector));
  }

  function people(path: string): (req: HttpRequest<unknown>) => boolean {
    return (req) => req.url === path;
  }

  async function answer(path: string, body: object, status = 200): Promise<TestRequest> {
    const req = httpMock.expectOne(people(path));
    req.flush(body, { status, statusText: status === 200 ? 'OK' : 'Error' });
    await settleResource();
    await settleResource();
    return req;
  }

  /** Open the view and answer the list, the first person and their shared photos. */
  async function openReady(): Promise<void> {
    fixture = TestBed.createComponent(PeopleViewComponent);
    await settleResource();
    await answer('/api/people', READY);
    const person = await answer('/api/people/1', PERSON_1);
    expect(person.request.params.get('partition')).toBe('p1');
    const shared = await answer('/api/people/1/with/2', SHARED_1_2);
    expect(shared.request.params.get('partition')).toBe('p1');
  }

  it('lists the people with two or more faces, then the unplaced, and opens on the first', async () => {
    await openReady();

    const names = all('.people-list .people-row-name').map((n) => n.textContent?.trim());
    expect(names).toEqual(['Person 1', 'Person 2', 'Unplaced']);
    expect(text()).toContain('3 faces · 2 photos');
    expect(text()).toContain('3 people · 1 unplaced face');
    // The one-face tail is folded away, and one click lists it.
    expect(el('.people-folded')?.textContent).toContain('+1 with fewer');
    el<HTMLButtonElement>('.people-folded')!.click();
    await settleResource();
    expect(all('.people-list .people-row-name').map((n) => n.textContent?.trim())).toEqual([
      'Person 1',
      'Person 2',
      'Person 3',
      'Unplaced',
    ]);

    // The centre: the person, their face shown large, the grid of their faces.
    expect(el('.people-person-name')?.textContent?.trim()).toBe('Person 1');
    expect(el<HTMLImageElement>('.people-face-img')?.getAttribute('src')).toContain('/api/medias/10/image');
    expect(el('.people-row.picked .people-row-name')?.textContent?.trim()).toBe('Person 1');
  });

  it('shows who they appear with, and the shared photo outlined by whose face is whose', async () => {
    await openReady();

    const co = all('.people-with-row');
    expect(co.map((c) => c.querySelector('.people-row-name')?.textContent?.trim())).toEqual(['Person 2']);
    expect(co[0].querySelector('.people-row-counts')?.textContent?.trim()).toBe('2 shared photos');
    expect(co[0].getAttribute('aria-pressed')).toBe('true');

    const img = el<HTMLImageElement>('.people-photo-img');
    expect(img?.getAttribute('src')).toBe('/api/medias/5/thumbnail?dataset_id=photos');
    expect(all('.people-photo-box--primary')).toHaveLength(1);
    expect(all('.people-photo-box--secondary')).toHaveLength(1);
    expect(all('.people-photo-box--other')).toHaveLength(1);
    // Colour is never the only cue: each box names whose it is.
    expect(el('.people-photo-box--primary')?.getAttribute('title')).toBe('Person 1');
    expect(el('.people-photo-box--other')?.getAttribute('title')).toBe('Person 3, also in this photo');

    // A photo opens in Show in photo with every face outlined by role.
    el<HTMLButtonElement>('.people-photo')!.click();
    const request = TestBed.inject(SourcePhotoService).request();
    expect(request?.mediaId).toBe(10);
    expect(request?.fromName).toBe('People');
    expect(request?.boxes?.map((b) => b.role)).toEqual(['primary', 'secondary', 'other']);
  });

  it('picking another person reads them under the same partition', async () => {
    await openReady();
    all('.people-list .people-row')[1].click();
    await settleResource();
    const req = await answer('/api/people/2', {
      ...PERSON_1,
      id: 2,
      representative: 20,
      face_ids: [20, 21],
      appears_with: [{ id: 1, shared: 2, representative: 10 }],
    });
    expect(req.request.params.get('partition')).toBe('p1');
    await answer('/api/people/2/with/1', { ...SHARED_1_2, id: 2, other: 1 });
    expect(el('.people-person-name')?.textContent?.trim()).toBe('Person 2');
    // The roles follow the pick: Person 2 is in focus now.
    expect(el('.people-photo-box--primary')?.getAttribute('title')).toBe('Person 2');
  });

  it('a stale person id reads the list again and picks the same person anew', async () => {
    fixture = TestBed.createComponent(PeopleViewComponent);
    await settleResource();
    await answer('/api/people', READY);
    await answer(
      '/api/people/1',
      { message: 'The people were found again since this partition was read.', error_code: 'people_changed' },
      409,
    );

    // The list is read again; the person whose representative is face 10 is
    // ranked second now, and is the one picked, under the new partition.
    await answer('/api/people', {
      ...READY,
      partition: 'p2',
      people: [
        { id: 1, face_count: 4, photo_count: 3, representative: 50 },
        { id: 2, face_count: 3, photo_count: 2, representative: 10 },
      ],
    });
    const again = await answer('/api/people/2', { ...PERSON_1, partition: 'p2', id: 2, appears_with: [] });
    expect(again.request.params.get('partition')).toBe('p2');
    expect(el('.people-person-name')?.textContent?.trim()).toBe('Person 2');
    expect(TestBed.inject(ToastService).toasts.some((t) => t.message.includes('found again'))).toBe(true);
    expect(text()).toContain('shares no photo with anyone else here');
  });

  it("offers to load the photos' dataset when it is not loaded, then shows them", async () => {
    fixture = TestBed.createComponent(PeopleViewComponent);
    await settleResource();
    await answer('/api/people', READY);
    await answer('/api/people/1', PERSON_1);
    await answer(
      '/api/people/1/with/2',
      { message: 'not loaded', error_code: 'source_not_loaded', dataset_id: 'photos' },
      409,
    );

    expect(text()).toContain("isn't loaded");
    const load = all('.people-shared .btn').find((b) => b.textContent?.includes('Load'));
    expect(load).toBeTruthy();
    load!.click();
    const loadReq = httpMock.expectOne('/api/datasets/registry/photos/load');
    loadReq.flush({ task_id: '' });
    await settleResource();
    await answer('/api/people/1/with/2', SHARED_1_2);
    expect(all('.people-photo-box')).toHaveLength(3);
  });

  it('the unplaced faces show in the centre, with no one appearing with them', async () => {
    await openReady();
    all('.people-list .people-row').find((r) => r.textContent?.includes('Unplaced'))!.click();
    await settleResource();
    expect(el('.people-person-name')?.textContent?.trim()).toBe('Unplaced faces');
    expect(el<HTMLImageElement>('.people-face-img')?.getAttribute('src')).toContain('/api/medias/40/image');
    expect(text()).toContain('no one appears with them');
    httpMock.expectNone(people('/api/people/unplaced'));
  });

  it('says why there are no people for an uncalibrated face embedder', async () => {
    fixture = TestBed.createComponent(PeopleViewComponent);
    await settleResource();
    await answer('/api/people', {
      status: 'not_calibrated',
      compat: 'ext:facenet',
      message: 'People is not calibrated for this face embedder (ext:facenet).',
    });
    expect(text()).toContain('No people to show.');
    expect(text()).toContain('People is not calibrated for this face embedder (ext:facenet).');
    expect(all('.people-row')).toHaveLength(0);
    expect(el<HTMLButtonElement>('.people-rebuild')?.disabled).toBe(true);
  });

  it('shows a fit in progress and polls to the people', async () => {
    fixture = TestBed.createComponent(PeopleViewComponent);
    await settleResource();
    vi.useFakeTimers();
    try {
      httpMock
        .expectOne(people('/api/people'))
        .flush({ status: 'building', message: 'Finding people among 40 faces…', current: 1, total: 4 });
      await vi.advanceTimersByTimeAsync(0);
      await fixture.whenStable();
      expect(text()).toContain('Finding people…');
      expect(text()).toContain('people among 40 faces');

      await vi.advanceTimersByTimeAsync(1000);
      httpMock.expectOne(people('/api/people')).flush(READY);
    } finally {
      vi.useRealTimers();
    }
    await settleResource();
    await answer('/api/people/1', PERSON_1);
    await answer('/api/people/1/with/2', SHARED_1_2);
    expect(el('.people-person-name')?.textContent?.trim()).toBe('Person 1');
  });

  it('Rebuild people fits again, and lands back on the same person', async () => {
    await openReady();
    el<HTMLButtonElement>('.people-rebuild')!.click();
    const build = httpMock.expectOne(people('/api/people/build'));
    expect(build.request.body).toEqual({ force: true });
    build.flush({ ...READY, partition: 'p9' });
    await settleResource();
    const again = await answer('/api/people/1', { ...PERSON_1, partition: 'p9' });
    expect(again.request.params.get('partition')).toBe('p9');
    await answer('/api/people/1/with/2', { ...SHARED_1_2, partition: 'p9' });
    expect(el('.people-person-name')?.textContent?.trim()).toBe('Person 1');
  });

  it('a failed read offers to try again', async () => {
    fixture = TestBed.createComponent(PeopleViewComponent);
    await settleResource();
    await answer('/api/people', { message: 'People are found in Face datasets only.', error_code: 'no_people' }, 409);
    expect(text()).toContain('People are found in Face datasets only.');
    all('.people-status .btn')[0].click();
    await settleResource();
    await answer('/api/people', READY);
    await answer('/api/people/1', PERSON_1);
    await answer('/api/people/1/with/2', SHARED_1_2);
    expect(all('.people-list .people-row').length).toBeGreaterThan(0);
  });
});
