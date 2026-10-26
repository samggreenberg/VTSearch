import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { DatasetExamplePickerComponent, PICKER_PAGE } from './dataset-example-picker.component';
import { ActiveContextService } from '../../../../services/active-context.service';
import { activeContextInterceptor } from '../../../../interceptors/active-context.interceptor';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../../testing/test-providers';
import { settleResource, settleZoneless } from '../../../../testing/settle-resource';

/** New Detector's grid of the dataset's own items (#5205). */
describe('DatasetExamplePickerComponent', () => {
  let fixture: ComponentFixture<DatasetExamplePickerComponent>;
  let httpMock: HttpTestingController;
  let el: HTMLElement;

  const ids = (n: number) => Array.from({ length: n }, (_, i) => ({ id: i + 1, media_type: 'face' }));

  async function open(mediaType: string, items: { id: number; media_type: string }[]): Promise<void> {
    fixture.componentRef.setInput('datasetId', 'faces-ds');
    fixture.componentRef.setInput('mediaType', mediaType);
    TestBed.tick();
    httpMock.expectOne('/api/media-types').flush({
      media_types: [
        { type_id: 'face', name: 'Face', has_thumbnail: true },
        { type_id: 'text', name: 'Text', has_thumbnail: false },
      ],
    });
    httpMock.expectOne('/api/medias/ids').flush(items);
    await settleResource();
    await settleZoneless(fixture);
  }

  /** Answer the names batch the grid asks for once a page is on screen. */
  function flushNames(): number[] {
    const req = httpMock.expectOne('/api/medias/batch');
    expect(req.request.headers.get('X-Dataset-Id')).toBe('faces-ds');
    const asked: number[] = req.request.body.ids;
    req.flush(asked.map((id) => ({ id, media_type: 'face', filename: `face_${id}.png`, md5: '' })));
    return asked;
  }

  beforeEach(async () => {
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
    httpMock.expectOne('/api/media-types').flush({ media_types: [] });
    const req = httpMock.expectOne('/api/medias/ids');
    expect(req.request.headers.get('X-Dataset-Id')).toBe('faces-ds');
    expect(req.request.headers.has('X-Detector-Id')).toBe(false);
    req.flush(ids(3));
    await settleResource();
    await settleZoneless(fixture);
    flushNames();
  });

  it('shows the first page of thumbnails out of that dataset, then the next on Show More', async () => {
    await open('face', ids(PICKER_PAGE + 5));
    expect(flushNames()).toEqual(ids(PICKER_PAGE).map((m) => m.id));
    await settleZoneless(fixture);

    const tiles = () => el.querySelectorAll<HTMLButtonElement>('.pick-tile');
    expect(tiles().length).toBe(PICKER_PAGE);
    const img = tiles()[0].querySelector('img')!;
    expect(img.getAttribute('src')).toBe('/api/medias/1/thumbnail?dataset_id=faces-ds');
    expect(tiles()[0].getAttribute('title')).toBe('face_1.png');
    expect(el.textContent).toContain(`Click one of its ${PICKER_PAGE + 5} items`);
    expect(el.textContent).toContain(`Showing ${PICKER_PAGE} of ${PICKER_PAGE + 5}`);

    const more = [...el.querySelectorAll<HTMLButtonElement>('.pick-more button')][0];
    more.click();
    await settleZoneless(fixture);
    // Only the new page's names are asked for.
    expect(flushNames()).toEqual([121, 122, 123, 124, 125]);
    await settleZoneless(fixture);
    expect(tiles().length).toBe(PICKER_PAGE + 5);
    expect(el.querySelector('.pick-more')).toBeNull();
  });

  it('emits the clicked item, and holds picks while the host is busy', async () => {
    await open('face', ids(3));
    flushNames();
    await settleZoneless(fixture);
    const picked: number[] = [];
    fixture.componentInstance.picked.subscribe((id) => picked.push(id));

    el.querySelectorAll<HTMLButtonElement>('.pick-tile')[1].click();
    expect(picked).toEqual([2]);

    fixture.componentRef.setInput('busy', true);
    await settleZoneless(fixture);
    const tile = el.querySelectorAll<HTMLButtonElement>('.pick-tile')[2];
    expect(tile.disabled).toBe(true);
    fixture.componentInstance.pick(3);
    expect(picked).toEqual([2]);
  });

  it('names the tiles of a media type with no thumbnail', async () => {
    await open('text', [{ id: 7, media_type: 'text' }]);
    flushNames();
    await settleZoneless(fixture);
    const tile = el.querySelector('.pick-tile')!;
    expect(tile.querySelector('img')).toBeNull();
    expect(tile.textContent?.trim()).toBe('face_7.png');
  });

  it('falls back to the name when a thumbnail fails', async () => {
    await open('face', ids(1));
    flushNames();
    await settleZoneless(fixture);
    el.querySelector('img')!.dispatchEvent(new Event('error'));
    await settleZoneless(fixture);
    expect(el.querySelector('img')).toBeNull();
    expect(el.querySelector('.pick-name')?.textContent).toBe('face_1.png');
  });

  it('says so when the dataset is empty', async () => {
    await open('face', []);
    expect(el.textContent).toContain('This dataset has no items.');
  });

  it('shows the server’s reason when the listing fails', async () => {
    fixture.componentRef.setInput('datasetId', 'faces-ds');
    TestBed.tick();
    httpMock.expectOne('/api/media-types').flush({ media_types: [] });
    httpMock
      .expectOne('/api/medias/ids')
      .flush({ message: 'Dataset faces-ds is not loaded' }, { status: 409, statusText: 'Conflict' });
    await settleResource();
    await settleZoneless(fixture);
    expect(el.querySelector('.error-text')?.textContent).toContain('Dataset faces-ds is not loaded');
  });
});
