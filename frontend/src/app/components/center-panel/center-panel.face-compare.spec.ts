import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { signal } from '@angular/core';
import { By } from '@angular/platform-browser';
import { of } from 'rxjs';

import { CenterPanelComponent } from './center-panel.component';
import { FaceReferenceComponent } from './face-reference/face-reference.component';
import { ImageViewerComponent } from './image-viewer/image-viewer.component';
import { Media } from '../../models/api.models';
import { ActiveContextService } from '../../services/active-context.service';
import { FaceReference, FaceReferenceService } from '../../services/face-reference.service';
import { KeyboardService } from '../../services/keyboard.service';
import { MediaTypeCapabilityService } from '../../services/media-type-capability.service';
import { SourcePhotoService } from '../../services/source-photo.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * The Face compare layout in the centre panel (#5642): a reference pane beside
 * a face while a detector is active, the toggle that turns it off, and the key
 * that steps the reference.
 */
describe('CenterPanelComponent Face compare', () => {
  let fixture: ComponentFixture<CenterPanelComponent>;
  let httpMock: HttpTestingController;
  let activeContext: ActiveContextService;
  const pool = signal<FaceReference[]>([]);

  const face: Media = { id: 4, media_type: 'face', filename: 'a.jpg→face_0.jpg', md5: 'm', custom_metadata: {} };
  const photo: Media = { id: 5, media_type: 'image', filename: 'a.jpg', md5: 'n', custom_metadata: {} };

  beforeEach(() => {
    pool.set([
      { kind: 'media', id: 1 },
      { kind: 'media', id: 2 },
    ]);
    configureZoneless({
      imports: [CenterPanelComponent],
      providers: [
        ...provideHttpTesting(),
        provideRouter([]),
        {
          provide: FaceReferenceService,
          useValue: { pool, pinned: signal(null), pin: vi.fn(), unpin: vi.fn(), loadQualities: vi.fn() },
        },
        // A face has no viewer of its own; it is shown as a still.
        { provide: MediaTypeCapabilityService, useValue: { usesThumbnails: () => true, ensureLoaded: vi.fn() } },
        {
          provide: SourcePhotoService,
          useValue: { offered: signal(false), request: signal(null), lookup: () => of(null), open: vi.fn(), close: vi.fn() },
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

  function toggle(): HTMLButtonElement | null {
    const all = Array.from(fixture.nativeElement.querySelectorAll('button')) as HTMLButtonElement[];
    return all.find((b) => b.textContent?.trim() === 'Compare') ?? null;
  }

  it('puts a reference beside a face, and leaves the candidate the panel\'s own viewer', async () => {
    await show(face);
    expect(reference()!.candidateId()).toBe(4);
    expect(toggle()!.getAttribute('aria-pressed')).toBe('true');
    // Zoom, rotate and the vote's region still address the face being voted on.
    expect(fixture.componentInstance.imageViewer()!.media().id).toBe(4);
    expect(fixture.debugElement.queryAll(By.directive(ImageViewerComponent)).length).toBe(2);
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
});
