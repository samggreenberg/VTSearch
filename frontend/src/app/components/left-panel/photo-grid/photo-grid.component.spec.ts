import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController, TestRequest } from '@angular/common/http/testing';

import { PHOTO_PAGE, PhotoGridComponent } from './photo-grid.component';
import type { CollapsedFace } from '../../../generated/api-client/models/collapsed-face';
import type { SortCollapsedResponse } from '../../../generated/api-client/models/sort-collapsed-response';
import { ActiveContextService } from '../../../services/active-context.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleResource } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

/**
 * Collapse to photos' grid (#4751): the ranking's faces, grouped by photo on
 * the server, drawn as photos with every matching face outlined; a face is its
 * card, so votes land on faces; the photo dataset is offered a load when the
 * server says it is not loaded.
 */
// The client generator renders a nullable enum's null as the string 'null'.
const NO_VOTE = null as unknown as CollapsedFace['vote'];

describe('PhotoGridComponent', () => {
  let fixture: ComponentFixture<PhotoGridComponent>;
  let httpMock: HttpTestingController;

  /** Photo 10 holds faces 1 and 3, photo 11 face 2; face 4's photo is not in the dataset. */
  const PAGE: SortCollapsedResponse = {
    results: [
      {
        source_dataset_id: 'photos',
        source_media_id: 10,
        score: 0.9,
        faces: [
          { media_id: 1, score: 0.9, box: [0.1, 0.2, 0.3, 0.5], vote: NO_VOTE },
          { media_id: 3, score: 0.1, box: [0.6, 0.6, 0.8, 0.9], vote: NO_VOTE },
        ],
      },
      {
        source_dataset_id: 'photos',
        source_media_id: 11,
        score: 0.7,
        faces: [{ media_id: 2, score: 0.7, box: [0, 0, 0.5, 0.5], vote: 'good' }],
      },
      {
        source_dataset_id: null,
        source_media_id: null,
        score: 0.5,
        faces: [{ media_id: 4, score: 0.5, box: null, vote: NO_VOTE }],
      },
    ],
    offset: 0,
    limit: PHOTO_PAGE,
    total: 3,
    face_total: 4,
    has_more: false,
  };

  beforeEach(async () => {
    configureZoneless({
      imports: [PhotoGridComponent],
      providers: [...provideHttpTesting()],
    });
    httpMock = TestBed.inject(HttpTestingController);
    TestBed.inject(ActiveContextService).setActivePair('faces', 'face-det');
    TestBed.inject(DatasetStateService).refresh();
    httpMock.expectOne('/api/datasets/registry').flush({
      datasets: [
        { id: 'faces', name: 'Party – Face', media_type: 'face', import_group: 'g1' },
        { id: 'photos', name: 'Party – Image', media_type: 'image', import_group: 'g1' },
      ],
    });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });

    fixture = TestBed.createComponent(PhotoGridComponent);
    fixture.componentRef.setInput('threshold', 0.6);
  });

  afterEach(() => {
    fixture.destroy();
    httpMock.verify();
  });

  function el(): HTMLElement {
    return fixture.nativeElement as HTMLElement;
  }

  function text(): string {
    return (el().textContent ?? '').replace(/\s+/g, ' ');
  }

  function expectPage(): TestRequest {
    return httpMock.expectOne((r) => r.url === '/api/sort/collapsed');
  }

  async function open(token = 'tok-1', page: SortCollapsedResponse = PAGE): Promise<void> {
    fixture.componentRef.setInput('token', token);
    await settleResource();
    const req = expectPage();
    expect(req.request.params.get('token')).toBe(token);
    expect(req.request.params.get('offset')).toBe('0');
    req.flush(page);
    await settleResource();
  }

  function cards(): HTMLElement[] {
    return Array.from(el().querySelectorAll<HTMLElement>('.photo-card'));
  }

  function boxes(card: HTMLElement): HTMLButtonElement[] {
    return Array.from(card.querySelectorAll<HTMLButtonElement>('.face-box'));
  }

  it('asks for nothing before there is a ranking', async () => {
    await settleResource();
    httpMock.expectNone((r) => r.url === '/api/sort/collapsed');
    expect(text()).toContain('sort first');
  });

  it('draws each photo once, out of the photo dataset, with every matching face outlined', async () => {
    fixture.componentRef.setInput('token', 'tok-1');
    await settleResource();
    const req = expectPage();
    expect(req.request.params.get('limit')).toBe(String(PHOTO_PAGE));
    req.flush(PAGE);
    await settleResource();
    const [first, second, orphan] = cards();
    expect(cards().length).toBe(3);

    const photo = first.querySelector<HTMLImageElement>('img.photo')!;
    // Read by the photo dataset's id alone: never with the face detector.
    expect(photo.getAttribute('src')).toBe('/api/medias/10/thumbnail?dataset_id=photos');
    const [a, b] = boxes(first);
    expect([a.style.left, a.style.top, a.style.width, a.style.height]).toEqual(['10%', '20%', '20%', '30%']);
    expect(b.style.left).toBe('60%');
    expect(first.querySelectorAll('.face-chip').length).toBe(2);
    expect(first.querySelector<HTMLImageElement>('.face-chip img')!.getAttribute('src')).toBe(
      '/api/medias/1/thumbnail?dataset_id=faces&detector_id=face-det',
    );
    expect(boxes(second).length).toBe(1);

    // A face whose photo is not there stands in for it.
    expect(orphan.querySelector<HTMLImageElement>('img.photo')!.getAttribute('src')).toBe(
      '/api/medias/4/thumbnail?dataset_id=faces&detector_id=face-det',
    );
    expect(orphan.textContent).toContain('photo not found');
    expect(text()).toContain('4 faces in 3 photos');
    expect(text()).toContain('counts and votes are per face');
  });

  it('draws the line before the first photo whose best face is below it', async () => {
    await open();
    const children = Array.from(el().querySelector('.photo-grid')!.children);
    const line = children.findIndex((c) => c.classList.contains('photo-threshold-line'));
    const firstBelow = children.indexOf(cards()[2]);
    expect(line).toBe(firstBelow - 1);
  });

  it('colours faces from the live votes, not from the page', async () => {
    fixture.componentRef.setInput('goodVotes', new Set([3]));
    fixture.componentRef.setInput('badVotes', new Set([1]));
    await open();
    const [a, b] = boxes(cards()[0]);
    expect(a.classList).toContain('bad');
    expect(b.classList).toContain('good');
    // The page said face 2 was good; the host's sets say it is not voted now.
    expect(boxes(cards()[1])[0].classList).not.toContain('good');
  });

  it('treats an outlined face as its card in click focus: select, right-click for the menu', async () => {
    await open();
    const selected: number[] = [];
    const menus: number[] = [];
    const votes: unknown[] = [];
    fixture.componentInstance.mediaSelect.subscribe((id) => selected.push(id));
    fixture.componentInstance.mediaContextRequest.subscribe((e) => menus.push(e.id));
    fixture.componentInstance.mediaVote.subscribe((v) => votes.push(v));

    boxes(cards()[0])[1].click();
    boxes(cards()[0])[1].dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(selected).toEqual([3]);
    expect(menus).toEqual([3]);
    expect(votes).toEqual([]);

    // A click on the photo away from every face picks its best face.
    cards()[0].querySelector<HTMLElement>('.photo-wrap')!.click();
    expect(selected).toEqual([3, 1]);
  });

  it('votes the face, not the photo, in hover focus', async () => {
    fixture.componentRef.setInput('focusMode', 'hover');
    await open();
    const selected: number[] = [];
    const votes: unknown[] = [];
    fixture.componentInstance.mediaSelect.subscribe((id) => selected.push(id));
    fixture.componentInstance.mediaVote.subscribe((v) => votes.push(v));

    const face3 = boxes(cards()[0])[1];
    face3.dispatchEvent(new MouseEvent('mouseenter'));
    face3.click();
    face3.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    cards()[0].querySelectorAll<HTMLButtonElement>('.face-chip')[0].click();
    expect(selected).toEqual([3]);
    expect(votes).toEqual([
      { id: 3, vote: 'bad' },
      { id: 3, vote: 'good' },
      { id: 1, vote: 'bad' },
    ]);
  });

  it('leaves hidden faces out, and a photo with none left', async () => {
    fixture.componentRef.setInput('hiddenIds', new Set([1, 2]));
    await open();
    // Photo 11 had only face 2; photo 10 keeps face 3, now scored 0.1.
    expect(cards().length).toBe(2);
    expect(boxes(cards()[0]).length).toBe(1);
    const children = Array.from(el().querySelector('.photo-grid')!.children);
    const line = children.findIndex((c) => c.classList.contains('photo-threshold-line'));
    expect(line).toBe(children.indexOf(cards()[0]) - 1);
  });

  it('offers to load the photo dataset when it is not loaded, then asks again', async () => {
    fixture.componentRef.setInput('token', 'tok-1');
    await settleResource();
    expectPage().flush(
      { code: 409, message: 'Load the source dataset', error_code: 'source_not_loaded', dataset_id: 'photos' },
      { status: 409, statusText: 'Conflict' },
    );
    await settleResource();
    expect(text()).toContain("The photos are in Party – Image, which isn't loaded.");

    const load = Array.from(el().querySelectorAll('button')).find((b) => b.textContent?.trim() === 'Load Party – Image')!;
    load.click();
    await settleResource();
    httpMock.expectOne('/api/datasets/registry/photos/load').flush({ ok: true, task_id: '' });
    await settleResource();
    expectPage().flush(PAGE);
    await settleResource();
    expect(cards().length).toBe(3);
  });

  it('says so when the ranking has expired on the server', async () => {
    fixture.componentRef.setInput('token', 'tok-1');
    await settleResource();
    expectPage().flush({ code: 404, message: 'Unknown or expired sort token' }, { status: 404, statusText: 'Not Found' });
    await settleResource();
    expect(text()).toContain('no longer held on the server');
  });

  it('pages in more photos, and asks a new ranking for as many as were showing', async () => {
    await open('tok-1', { ...PAGE, has_more: true, total: 400 });
    // Fewer than a screenful showing: the grid pages in on its own.
    const more = expectPage();
    expect(more.request.params.get('offset')).toBe('3');
    more.flush({ ...PAGE, results: [{ ...PAGE.results[1], source_media_id: 12 }], offset: 3, has_more: false });
    await settleResource();
    expect(cards().length).toBe(4);

    fixture.componentRef.setInput('token', 'tok-2');
    await settleResource();
    const again = expectPage();
    expect(again.request.params.get('token')).toBe('tok-2');
    expect(again.request.params.get('offset')).toBe('0');
    expect(again.request.params.get('limit')).toBe(String(PHOTO_PAGE));
    again.flush(PAGE);
    await settleResource();
    expect(cards().length).toBe(3);
  });
});
