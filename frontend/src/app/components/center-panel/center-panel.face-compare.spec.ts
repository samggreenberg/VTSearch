import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { signal } from '@angular/core';
import { By } from '@angular/platform-browser';
import { Observable, Subject, of } from 'rxjs';

import { CenterPanelComponent } from './center-panel.component';
import { FaceReferenceComponent } from './face-reference/face-reference.component';
import { ImageViewerComponent } from './image-viewer/image-viewer.component';
import { Media } from '../../models/api.models';
import { ActiveContextService } from '../../services/active-context.service';
import { FaceReference, FaceReferenceService } from '../../services/face-reference.service';
import { KeyboardService } from '../../services/keyboard.service';
import { MediaTypeCapabilityService } from '../../services/media-type-capability.service';
import { SourceLookup, SourcePhotoService } from '../../services/source-photo.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * The Face compare layout in the centre panel (#5642): a reference pane beside
 * a face while a detector is active, the toggle that turns it off, and the key
 * that steps the reference. And its Wider toggle (#5645), which shows both
 * faces from their photos.
 */
describe('CenterPanelComponent Face compare', () => {
  let fixture: ComponentFixture<CenterPanelComponent>;
  let httpMock: HttpTestingController;
  let activeContext: ActiveContextService;
  const pool = signal<FaceReference[]>([]);
  const wider = signal(false);
  const offered = signal(false);
  const photosLoaded = signal(false);
  /** Each face's answer to the source lookup, by id; none is a 404. */
  let lookups: Record<number, Observable<SourceLookup>>;

  const LANDMARKS = [
    [0.4, 0.4],
    [0.6, 0.4],
    [0.5, 0.5],
    [0.45, 0.6],
    [0.55, 0.6],
  ];
  const face: Media = {
    id: 4,
    media_type: 'face',
    filename: 'a.jpg→face_0.jpg',
    md5: 'm',
    custom_metadata: {},
    face_landmarks: LANDMARKS,
  };
  const photo: Media = { id: 5, media_type: 'image', filename: 'a.jpg', md5: 'n', custom_metadata: {} };

  beforeEach(() => {
    pool.set([
      { kind: 'media', id: 1 },
      { kind: 'media', id: 2 },
    ]);
    wider.set(false);
    offered.set(false);
    photosLoaded.set(false);
    lookups = {};
    configureZoneless({
      imports: [CenterPanelComponent],
      providers: [
        ...provideHttpTesting(),
        provideRouter([]),
        {
          provide: FaceReferenceService,
          useValue: {
            pool,
            pinned: signal(null),
            pin: vi.fn(),
            unpin: vi.fn(),
            loadQualities: vi.fn(),
            wider,
            toggleWider: () => wider.update((on) => !on),
          },
        },
        // A face has no viewer of its own; it is shown as a still.
        { provide: MediaTypeCapabilityService, useValue: { usesThumbnails: () => true, ensureLoaded: vi.fn() } },
        {
          provide: SourcePhotoService,
          useValue: {
            offered,
            photosLoaded,
            request: signal(null),
            lookup: (id: number) => lookups[id] ?? of({ kind: 'none', message: 'No photo' }),
            open: vi.fn(),
            close: vi.fn(),
          },
        },
      ],
    });
    activeContext = TestBed.inject(ActiveContextService);
    activeContext.setActivePair('faces', 'det');
    httpMock = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(CenterPanelComponent);
  });

  afterEach(() => {
    fixture.destroy();
    httpMock.match(() => true);
  });

  async function show(media: Media): Promise<void> {
    fixture.componentRef.setInput('media', media);
    await settleZoneless(fixture);
  }

  /** Start the panel as its host does, answering the loads that sets off. */
  async function init(settings: object): Promise<void> {
    fixture.componentInstance.init();
    await new Promise<void>((resolve) => setTimeout(resolve));
    TestBed.tick();
    httpMock.match((r) => r.url.includes('settings')).forEach((req) => req.flush(settings));
    httpMock.match((r) => r.url.includes('embedders')).forEach((req) => req.flush({ embedders: [] }));
    await settleZoneless(fixture);
  }

  function reference(): FaceReferenceComponent | null {
    const found = fixture.debugElement.query(By.directive(FaceReferenceComponent));
    return found ? (found.componentInstance as FaceReferenceComponent) : null;
  }

  function button(label: string): HTMLButtonElement | null {
    const all = Array.from(fixture.nativeElement.querySelectorAll('button')) as HTMLButtonElement[];
    return all.find((b) => b.textContent?.trim() === label) ?? null;
  }

  function toggle(): HTMLButtonElement | null {
    return button('Compare');
  }

  it('puts a reference beside a face, and leaves the candidate the panel\'s own viewer', async () => {
    await show(face);
    expect(reference()!.candidateId()).toBe(4);
    expect(toggle()!.getAttribute('aria-pressed')).toBe('true');
    // Zoom, rotate and the vote's region still address the face being voted on.
    expect(fixture.componentInstance.imageViewer()!.media().id).toBe(4);
    expect(fixture.debugElement.queryAll(By.directive(ImageViewerComponent)).length).toBe(2);
  });

  // #5647: the two viewers zoom and pan together, the faces matched in size and
  // place by their landmarks; the candidate's leads.
  it('links the reference\'s viewer to the candidate\'s, which leads with its landmarks', async () => {
    await show(face);
    const panel = fixture.componentInstance;
    const candidate = panel.imageViewer()!;
    expect(candidate.link()).toBe(panel.faceLink);
    expect(candidate.linkAs()).toBe('lead');
    expect(candidate.landmarks()).toBe(LANDMARKS);
    expect(panel.faceLink.lead()).toBe(candidate);

    const viewers = fixture.debugElement.queryAll(By.directive(ImageViewerComponent));
    const ref = viewers.map((v) => v.componentInstance as ImageViewerComponent).find((v) => v !== candidate)!;
    expect(ref.link()).toBe(panel.faceLink);
    expect(ref.linkAs()).toBe('follow');
  });

  it('takes the candidate off the link when the layout is turned off', async () => {
    await show(face);
    toggle()!.click();
    await settleZoneless(fixture);
    expect(fixture.componentInstance.imageViewer()!.link()).toBeNull();
    expect(fixture.componentInstance.faceLink.lead()).toBeNull();
    httpMock.match((r) => r.url === '/api/settings');
  });

  it('offers nothing on an image, or on a face with no detector active', async () => {
    await show(photo);
    expect(reference()).toBeNull();
    expect(toggle()).toBeNull();

    activeContext.setActivePair('faces', '');
    await show(face);
    expect(reference()).toBeNull();
    expect(toggle()).toBeNull();
  });

  it('turns off from the toggle, saves the choice, and turns back on', async () => {
    await show(face);
    toggle()!.click();
    await settleZoneless(fixture);
    expect(reference()).toBeNull();
    expect(toggle()!.getAttribute('aria-pressed')).toBe('false');
    const off = httpMock.expectOne((r) => r.method === 'PUT' && r.url === '/api/settings');
    expect(off.request.body).toEqual({ face_compare: false });
    off.flush({ face_compare: false });
    await settleZoneless(fixture);

    toggle()!.click();
    await settleZoneless(fixture);
    expect(reference()).not.toBeNull();
    const on = httpMock.expectOne((r) => r.method === 'PUT' && r.url === '/api/settings');
    expect(on.request.body).toEqual({ face_compare: true });
  });

  it('stays off for a user who turned it off, with the toggle still there', async () => {
    await init({ face_compare: false });
    await show(face);
    expect(reference()).toBeNull();
    expect(toggle()!.getAttribute('aria-pressed')).toBe('false');
  });

  it('steps the reference on the R key, and the candidate stays', async () => {
    await init({});
    await show(face);
    expect(reference()!.current()).toEqual({ kind: 'media', id: 1 });

    TestBed.inject(KeyboardService).action$.next({ type: 'reference' });
    await settleZoneless(fixture);
    expect(reference()!.current()).toEqual({ kind: 'media', id: 2 });
    expect(fixture.componentInstance.imageViewer()!.media().id).toBe(4);
  });

  describe('Wider', () => {
    const found = (mediaId: number, box: [number, number, number, number]): Observable<SourceLookup> =>
      of({ kind: 'found', datasetId: 'photos', mediaId, box });

    beforeEach(() => {
      offered.set(true);
      photosLoaded.set(true);
      lookups = { 4: found(40, [0.4, 0.4, 0.6, 0.6]), 1: found(10, [0.1, 0.2, 0.3, 0.4]) };
    });

    function referenceViewer(): ImageViewerComponent {
      const viewers = fixture.debugElement.queryAll(By.directive(ImageViewerComponent));
      const candidate = fixture.componentInstance.imageViewer();
      return viewers.map((v) => v.componentInstance as ImageViewerComponent).find((v) => v !== candidate)!;
    }

    async function turnOn(): Promise<void> {
      button('Wider')!.click();
      await settleZoneless(fixture);
    }

    it('is offered beside Compare once the photos are loaded, and only in the layout', async () => {
      photosLoaded.set(false);
      await show(face);
      expect(button('Wider')).toBeNull();

      photosLoaded.set(true);
      await settleZoneless(fixture);
      expect(button('Wider')!.getAttribute('aria-pressed')).toBe('false');

      toggle()!.click();
      await settleZoneless(fixture);
      expect(button('Wider')).toBeNull();
      httpMock.match((r) => r.url === '/api/settings');
    });

    it('is not offered when the lookup says the photos are not loaded after all', async () => {
      lookups[4] = of({ kind: 'not-loaded', datasetId: 'photos' });
      await show(face);
      expect(button('Wider')).toBeNull();
    });

    it('shows both faces from their photos, framed twice as wide as their crops', async () => {
      await show(face);
      await turnOn();
      expect(button('Wider')!.getAttribute('aria-pressed')).toBe('true');

      const candidate = fixture.componentInstance.imageViewer()!;
      expect(candidate.media().id).toBe(40);
      expect(candidate.datasetId()).toBe('photos');
      const box = candidate.fitBox()!;
      [0.3, 0.3, 0.7, 0.7].forEach((v, i) => expect(box[i]).toBeCloseTo(v, 9));

      const ref = referenceViewer();
      expect(ref.media().id).toBe(10);
      expect(ref.datasetId()).toBe('photos');
      const refBox = ref.fitBox()!;
      [0, 0.1, 0.4, 0.5].forEach((v, i) => expect(refBox[i]).toBeCloseTo(v, 9));
      // The landmarks were measured on the levelled crop, not on the photo.
      expect(candidate.landmarks()).toBeNull();
    });

    it('goes back to the crops when turned off', async () => {
      await show(face);
      await turnOn();
      await turnOn();
      const candidate = fixture.componentInstance.imageViewer()!;
      expect(candidate.media().id).toBe(4);
      expect(candidate.datasetId()).toBe('');
      expect(candidate.fitBox()).toBeNull();
      expect(referenceViewer().media().id).toBe(1);
    });

    it('keeps a face whose photo cannot be found on its crop, and the toggle with it', async () => {
      delete lookups[4];
      await show(face);
      await turnOn();
      const candidate = fixture.componentInstance.imageViewer()!;
      expect(candidate.media().id).toBe(4);
      expect(candidate.fitBox()).toBeNull();
      expect(button('Wider')).not.toBeNull();
      // The reference's photo is found, so it still goes wider.
      expect(referenceViewer().media().id).toBe(10);
    });

    it('shows nothing for the next face until its photo is found, rather than its crop', async () => {
      await show(face);
      await turnOn();
      const pending = new Subject<SourceLookup>();
      lookups[6] = pending;
      // Not `show`: `whenStable` would wait out the lookup this means to catch.
      fixture.componentRef.setInput('media', { ...face, id: 6 });
      await settleResource();

      const host = (): HTMLElement =>
        fixture.debugElement
          .queryAll(By.directive(ImageViewerComponent))
          .find((v) => v.componentInstance === fixture.componentInstance.imageViewer())!.nativeElement;
      expect(host().classList.contains('wider-pending')).toBe(true);

      pending.next({ kind: 'found', datasetId: 'photos', mediaId: 60, box: [0, 0, 0.5, 0.5] });
      pending.complete();
      await settleZoneless(fixture);
      expect(host().classList.contains('wider-pending')).toBe(false);
      expect(fixture.componentInstance.imageViewer()!.media().id).toBe(60);
    });
  });
});
