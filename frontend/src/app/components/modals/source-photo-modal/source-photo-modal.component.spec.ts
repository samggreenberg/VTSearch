import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { SourcePhotoModalComponent } from './source-photo-modal.component';
import { ImageViewerComponent } from '../../center-panel/image-viewer/image-viewer.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { DATASET_OVERRIDE } from '../../../interceptors/active-context.interceptor';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleResource } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

// While a lookup is in flight the resource keeps the zoneless app unstable, so
// the steps here drain with `settleResource` (a macrotask, then a tick) rather
// than `whenStable`, which would wait on the very request the test answers.

/**
 * Show in photo's overlay (#4750): the photo out of the Face dataset's Image
 * sibling with the face outlined, the offer to load that sibling when it is
 * not loaded, and Back to the ranking. The active pair is never touched.
 */
describe('SourcePhotoModalComponent', () => {
  let fixture: ComponentFixture<SourcePhotoModalComponent>;
  let httpMock: HttpTestingController;
  let activeContext: ActiveContextService;

  beforeEach(async () => {
    configureZoneless({
      imports: [SourcePhotoModalComponent],
      providers: [...provideHttpTesting()],
    });
    httpMock = TestBed.inject(HttpTestingController);
    activeContext = TestBed.inject(ActiveContextService);
    activeContext.setActivePair('faces', 'face-det');

    TestBed.inject(DatasetStateService).refresh();
    httpMock.expectOne('/api/datasets/registry').flush({
      datasets: [
        { id: 'faces', name: 'Photos – Face', media_type: 'face', import_group: 'g1' },
        { id: 'photos', name: 'Photos – Image', media_type: 'image', import_group: 'g1' },
      ],
    });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });

    fixture = TestBed.createComponent(SourcePhotoModalComponent);
    fixture.componentRef.setInput('request', { mediaId: 4, fromName: 'Photos – Face' });
    await settleResource();
  });

  afterEach(() => {
    fixture.destroy();
    httpMock.verify();
  });

  function text(): string {
    return (fixture.nativeElement.textContent ?? '').replace(/\s+/g, ' ');
  }

  async function answerSource(body: object, status = 200): Promise<void> {
    const statusText = status === 200 ? 'OK' : 'Error';
    httpMock.expectOne('/api/medias/4/source').flush(body, { status, statusText });
    await settleResource();
  }

  async function answerPhoto(): Promise<void> {
    const req = httpMock.expectOne('/api/medias/batch');
    expect(req.request.body).toEqual({ ids: [2] });
    expect(req.request.context.get(DATASET_OVERRIDE)).toBe('photos');
    req.flush([{ id: 2, media_type: 'image', filename: 'b.jpg', md5: 'x', custom_metadata: {} }]);
    await settleResource();
  }

  it('shows the photo out of the sibling, with the face box outlined', async () => {
    expect(text()).toContain('Finding the photo');
    await answerSource({ dataset_id: 'photos', media_id: 2, box: [0.1, 0.2, 0.3, 0.4] });
    await answerPhoto();

    const viewer = fixture.debugElement.query((d) => d.componentInstance instanceof ImageViewerComponent)
      .componentInstance as ImageViewerComponent;
    expect(viewer.media().id).toBe(2);
    expect(viewer.datasetId()).toBe('photos');
    expect(viewer.outlineBox()).toEqual([0.1, 0.2, 0.3, 0.4]);
    expect(viewer.regionDraw()).toBe(false);
    expect(viewer.imageSrc()).toBe('/api/medias/2/image?dataset_id=photos');
    expect(text()).toContain('Photos – Image');
    expect(text()).toContain('b.jpg');
  });

  it("gives the photo the centre panel's toolbar and zoom/rotate keys (#5227)", async () => {
    await answerSource({ dataset_id: 'photos', media_id: 2, box: [0.1, 0.2, 0.3, 0.4] });
    await answerPhoto();
    const viewer = fixture.debugElement.query((d) => d.componentInstance instanceof ImageViewerComponent)
      .componentInstance as ImageViewerComponent;
    const toolbar = fixture.nativeElement.querySelector('.image-view-controls') as HTMLElement;
    const control = (label: string) => toolbar.querySelector(`[aria-label="${label}"]`) as HTMLElement;

    for (const label of ['Rotate image left', 'Rotate image right', 'Zoom out', 'Zoom in', 'Zoom level', 'Reset image view']) {
      expect(control(label)).toBeTruthy();
    }
    // Nothing is voted on here, so no Marquee or Highlight to lead the strip.
    expect(toolbar.querySelector('.ivc-btn-toggle')).toBeNull();
    expect(toolbar.querySelector('.ivc-sep')).toBeNull();

    control('Zoom in').click();
    control('Rotate image right').click();
    expect(viewer.zoom()).toBeGreaterThan(1);
    expect(viewer.rotation()).toBe(90);
    control('Reset image view').click();
    TestBed.tick();
    expect(viewer.zoom()).toBe(1);
    expect(viewer.rotation()).toBe(0);
    expect(toolbar.querySelector('.ivc-zoom-label')!.textContent).toBe('1×');

    // The ranking's shortcuts are off under a modal; the overlay answers these.
    document.dispatchEvent(new KeyboardEvent('keydown', { key: '+' }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: '[' }));
    expect(viewer.zoom()).toBeGreaterThan(1);
    expect(viewer.rotation()).toBe(-90);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: '-' }));
    expect(viewer.zoom()).toBeLessThan(1.15);
    // Ctrl/Cmd with -/+ is the browser's page zoom; leave it alone.
    const pageZoom = new KeyboardEvent('keydown', { key: '-', ctrlKey: true, cancelable: true });
    const before = viewer.zoom();
    document.dispatchEvent(pageZoom);
    expect(viewer.zoom()).toBe(before);
    expect(pageZoom.defaultPrevented).toBe(false);
  });

  it('never moves the active pair', async () => {
    await answerSource({ dataset_id: 'photos', media_id: 2, box: null });
    await answerPhoto();
    expect(activeContext.datasetId).toBe('faces');
    expect(activeContext.modelId).toBe('face-det');
    expect(text()).toContain('no box was recorded');
  });

  it('offers to load an unloaded sibling, then asks again', async () => {
    await answerSource(
      { code: 409, message: 'Load the source dataset', error_code: 'source_not_loaded', dataset_id: 'photos' },
      409,
    );
    expect(text()).toContain("The photo is in Photos – Image, which isn't loaded.");
    const load = Array.from(fixture.nativeElement.querySelectorAll('button') as NodeListOf<HTMLButtonElement>).find(
      (b) => b.textContent?.trim() === 'Load Photos – Image',
    )!;
    load.click();
    await settleResource();

    // A load with no background task (already loaded) asks again at once.
    httpMock.expectOne('/api/datasets/registry/photos/load').flush({ ok: true, task_id: '' });
    await settleResource();
    await answerSource({ dataset_id: 'photos', media_id: 2, box: [0.1, 0.2, 0.3, 0.4] });
    await answerPhoto();
    expect(text()).toContain('b.jpg');
  });

  it('says so when the item has no source photo', async () => {
    await answerSource({ code: 404, message: 'This item was not made by a converter.', error_code: 'not_derived' }, 404);
    expect(text()).toContain('There is no photo to show for this item.');
    expect(text()).toContain('This item was not made by a converter.');
  });

  it('goes Back to the ranking it came from', async () => {
    await answerSource({ code: 404, message: 'x' }, 404);
    let closed = 0;
    fixture.componentInstance.closed.subscribe(() => closed++);
    const back = fixture.nativeElement.querySelector('.back-btn') as HTMLButtonElement;
    expect(back.textContent).toContain('Back to Photos – Face');
    back.click();
    expect(closed).toBe(1);
  });
});
