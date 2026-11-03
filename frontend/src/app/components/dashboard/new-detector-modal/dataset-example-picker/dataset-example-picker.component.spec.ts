import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController, TestRequest } from '@angular/common/http/testing';

import {
  DatasetExamplePickerComponent,
  SEARCH_DEBOUNCE_MS,
  THUMB_SETTLE_MS,
} from './dataset-example-picker.component';
import { ActiveContextService } from '../../../../services/active-context.service';
import { activeContextInterceptor } from '../../../../interceptors/active-context.interceptor';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../../testing/test-providers';
import { settleResource, settleZoneless } from '../../../../testing/settle-resource';

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** New Detector's grid of the dataset's own items (#5205, #5465). */
describe('DatasetExamplePickerComponent', () => {
  let fixture: ComponentFixture<DatasetExamplePickerComponent>;
  let httpMock: HttpTestingController;
  let el: HTMLElement;

  const range = (n: number) => Array.from({ length: n }, (_, i) => i + 1);
  const listing = (ids: number[], order = 'dataset', total = ids.length) => ({ ids, total, order });

  /** Answer the media-types request and hand back the listing request. */
  function answerOpen(): TestRequest {
    httpMock.expectOne('/api/media-types').flush({
      media_types: [
        { type_id: 'face', name: 'Face', has_thumbnail: true },
        { type_id: 'text', name: 'Text', has_thumbnail: false },
      ],
    });
    return httpMock.expectOne((req) => req.url === '/api/medias/picker');
  }

  /** The size buttons load the settings when the toolbar renders. */
  async function answerSettings(sizes: Record<string, string> = {}): Promise<void> {
    httpMock.expectOne('/api/settings').flush({ grid_icon_size_picker: sizes });
    await settleResource();
  }

  async function open(
    mediaType: string,
    body: ReturnType<typeof listing>,
    sizes: Record<string, string> = {},
  ): Promise<void> {
    fixture.componentRef.setInput('datasetId', 'faces-ds');
    fixture.componentRef.setInput('mediaType', mediaType);
    TestBed.tick();
    answerOpen().flush(body);
    await settleResource();
    if (body.total > 0) await answerSettings(sizes);
    await settleZoneless(fixture);
  }

  /** Let the scroll "rest" so the tiles on screen load. */
  async function rest(): Promise<void> {
    await wait(THUMB_SETTLE_MS + 30);
    await settleZoneless(fixture);
  }

  /** Answer the names batch the grid asks for once tiles load. */
  function flushNames(): number[] {
    const req = httpMock.expectOne('/api/medias/batch');
    expect(req.request.headers.get('X-Dataset-Id')).toBe('faces-ds');
    const asked: number[] = req.request.body.ids;
    req.flush(asked.map((id) => ({ id, media_type: 'face', filename: `face_${id}.png`, md5: '' })));
    return asked;
  }

  const tiles = () => [...el.querySelectorAll<HTMLButtonElement>('.pick-tile')];
  const rowEls = () => [...el.querySelectorAll<HTMLElement>('.pick-row')];

  beforeEach(async () => {
    // jsdom has no Element.scrollTo; the CDK viewport calls it when a new
    // search scrolls the grid back to the top.
    if (!Element.prototype.scrollTo) {
      (Element.prototype as unknown as { scrollTo: () => void }).scrollTo = () => {};
    }
    await TestBed.configureTestingModule({
      imports: [DatasetExamplePickerComponent],
      providers: [...provideZoneless(), ...provideHttpTesting(activeContextInterceptor)],
    }).compileComponents();
    // Some other pair is active: the grid must read the dataset it was given,
    // and never send the active detector along with it.
    TestBed.inject(ActiveContextService).setActive('other-ds', 'other-det');
    fixture = TestBed.createComponent(DatasetExamplePickerComponent);
    httpMock = TestBed.inject(HttpTestingController);
    el = fixture.nativeElement;
  });

  afterEach(() => httpMock.verify());

  it('lists the given dataset, not the active one, without the active detector', async () => {
    fixture.componentRef.setInput('datasetId', 'faces-ds');
    fixture.componentRef.setInput('mediaType', 'face');
    TestBed.tick();
    const req = answerOpen();
    expect(req.request.headers.get('X-Dataset-Id')).toBe('faces-ds');
    expect(req.request.headers.has('X-Detector-Id')).toBe(false);
    expect(req.request.params.has('q')).toBe(false);
    req.flush(listing(range(3)));
    await settleResource();
    await answerSettings();
    await settleZoneless(fixture);
    await rest();
    flushNames();
  });

  it('lays the items out in rows, in the server’s order', async () => {
    // At the default size (M, 80px tiles) the fallback 640px width holds 7.
    await open('face', listing([9, 3, 5, 1, 2, 4, 6, 8, 7], 'similarity'));
    expect(rowEls().map((row) => row.querySelectorAll('.pick-tile').length)).toEqual([7, 2]);
    await rest();
    flushNames();
    await settleZoneless(fixture);
    expect(tiles().map((t) => t.getAttribute('title'))).toEqual(
      [9, 3, 5, 1, 2, 4, 6, 8, 7].map((id) => `face_${id}.png`),
    );
    expect(el.textContent).toContain('Click one of its 9 items');
    expect(el.textContent).toContain('Look-alikes are listed together.');
  });

  it('loads a tile’s thumbnail only once the scroll rests on it', async () => {
    await open('face', listing(range(3)));
    // On screen, but the scroll hasn't rested yet: no request queued.
    expect(el.querySelector('img')).toBeNull();
    await rest();
    expect(flushNames()).toEqual([1, 2, 3]);
    await settleZoneless(fixture);
    const img = tiles()[0].querySelector('img')!;
    expect(img.getAttribute('src')).toBe('/api/medias/1/thumbnail?dataset_id=faces-ds');
    expect(tiles()[0].getAttribute('title')).toBe('face_1.png');
    expect(el.textContent).not.toContain('Look-alikes');
  });

  it('sizes the tiles from the picker’s own thumbnail-size setting', async () => {
    // XL is 200px: three fit across the fallback 640px.
    await open('face', listing(range(4)), { face: 'XL' });
    expect(rowEls().map((row) => row.querySelectorAll('.pick-tile').length)).toEqual([3, 1]);
    expect(rowEls()[0].style.getPropertyValue('--pick-cell')).toBe('200px');
    expect(el.querySelector('vt-view-controls')).toBeTruthy();
    await rest();
    flushNames();
  });

  it('searches by name on the server, once typing pauses', async () => {
    await open('face', listing(range(3), 'similarity'));
    await rest();
    flushNames();
    const box = el.querySelector<HTMLInputElement>('.pick-search')!;
    box.value = 'beach';
    box.dispatchEvent(new Event('input'));
    box.value = 'beach party ';
    box.dispatchEvent(new Event('input'));
    await wait(SEARCH_DEBOUNCE_MS + 30);
    TestBed.tick();
    const req = httpMock.expectOne((r) => r.url === '/api/medias/picker');
    expect(req.request.params.get('q')).toBe('beach party');
    expect(req.request.headers.get('X-Dataset-Id')).toBe('faces-ds');
    req.flush(listing([2], 'similarity', 3));
    await settleResource();
    await settleZoneless(fixture);
    expect(tiles().length).toBe(1);
    expect(el.textContent).toContain('1 of its 3 items match');
    // Its name was already fetched: no second batch.
    await rest();
    expect(tiles()[0].getAttribute('title')).toBe('face_2.png');
  });

  it('keeps the search box when nothing matches', async () => {
    await open('face', listing(range(2)));
    await rest();
    flushNames();
    const box = el.querySelector<HTMLInputElement>('.pick-search')!;
    box.value = 'nobody';
    box.dispatchEvent(new Event('input'));
    await wait(SEARCH_DEBOUNCE_MS + 30);
    TestBed.tick();
    httpMock.expectOne((r) => r.url === '/api/medias/picker').flush(listing([], 'dataset', 2));
    await settleResource();
    await settleZoneless(fixture);
    expect(tiles().length).toBe(0);
    expect(el.querySelector('.pick-search')).toBe(box);
    expect(el.textContent).toContain('No item’s name holds “nobody”.');
  });

  it('emits the clicked item, and holds picks while the host is busy', async () => {
    await open('face', listing(range(3)));
    await rest();
    flushNames();
    await settleZoneless(fixture);
    const picked: number[] = [];
    fixture.componentInstance.picked.subscribe((id) => picked.push(id));

    tiles()[1].click();
    expect(picked).toEqual([2]);

    fixture.componentRef.setInput('busy', true);
    await settleZoneless(fixture);
    expect(tiles()[2].disabled).toBe(true);
    fixture.componentInstance.pick(3);
    expect(picked).toEqual([2]);
  });

  it('names the tiles of a media type with no thumbnail', async () => {
    await open('text', listing([7]));
    await rest();
    flushNames();
    await settleZoneless(fixture);
    const tile = tiles()[0];
    expect(tile.querySelector('img')).toBeNull();
    expect(tile.textContent?.trim()).toBe('face_7.png');
  });

  it('falls back to the name when a thumbnail fails', async () => {
    await open('face', listing([1]));
    await rest();
    flushNames();
    await settleZoneless(fixture);
    el.querySelector('img')!.dispatchEvent(new Event('error'));
    await settleZoneless(fixture);
    expect(el.querySelector('img')).toBeNull();
    expect(el.querySelector('.pick-name')?.textContent).toBe('face_1.png');
  });

  it('says so when the dataset is empty', async () => {
    await open('face', listing([]));
    expect(el.textContent).toContain('This dataset has no items.');
    expect(el.querySelector('.pick-search')).toBeNull();
  });

  it('shows the server’s reason when the listing fails', async () => {
    fixture.componentRef.setInput('datasetId', 'faces-ds');
    TestBed.tick();
    answerOpen().flush({ message: 'Dataset faces-ds is not loaded' }, { status: 409, statusText: 'Conflict' });
    await settleResource();
    await settleZoneless(fixture);
    expect(el.querySelector('.error-text')?.textContent).toContain('Dataset faces-ds is not loaded');
  });
});
