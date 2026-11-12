import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { By } from '@angular/platform-browser';
import { Observable, of } from 'rxjs';

import { FaceReferenceComponent } from './face-reference.component';
import { ImageViewerComponent } from '../image-viewer/image-viewer.component';
import { FaceReference, FaceReferenceService } from '../../../services/face-reference.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { SourceLookup, SourcePhotoService } from '../../../services/source-photo.service';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

/**
 * The reference pane of the Face compare layout (#5642): which reference it
 * shows for a candidate, stepping through the set, and the pin. And, under the
 * layout's Wider toggle (#5645), the reference from its photo.
 */
describe('FaceReferenceComponent', () => {
  let fixture: ComponentFixture<FaceReferenceComponent>;
  const pool = signal<FaceReference[]>([]);
  const pinned = signal<FaceReference | null>(null);
  let refs: {
    pool: typeof pool;
    pinned: typeof pinned;
    pin: ReturnType<typeof vi.fn<(ref: FaceReference) => void>>;
    unpin: ReturnType<typeof vi.fn<() => void>>;
    loadQualities: ReturnType<typeof vi.fn<() => void>>;
  };

  const media = (id: number): FaceReference => ({ kind: 'media', id });
  let lookup: ReturnType<typeof vi.fn<(id: number) => Observable<SourceLookup>>>;

  beforeEach(async () => {
    pool.set([media(1), media(2), media(3)]);
    pinned.set(null);
    refs = { pool, pinned, pin: vi.fn((ref) => pinned.set(ref)), unpin: vi.fn(() => pinned.set(null)), loadQualities: vi.fn() };
    lookup = vi.fn((id: number) => of<SourceLookup>({ kind: 'found', datasetId: 'photos', mediaId: id * 10, box: [0.2, 0.2, 0.4, 0.4] }));
    configureZoneless({
      imports: [FaceReferenceComponent],
      providers: [
        ...provideHttpTesting(),
        { provide: FaceReferenceService, useValue: refs },
        { provide: SourcePhotoService, useValue: { lookup } },
        {
          provide: MediaMetadataCacheService,
          useValue: {
            version: signal(0),
            get: (id: number) => (id === 2 ? { id, media_type: 'face', filename: 'two.jpg', custom_metadata: {} } : undefined),
          },
        },
      ],
    });
    fixture = TestBed.createComponent(FaceReferenceComponent);
    await candidate(10);
  });

  afterEach(() => fixture.destroy());

  async function candidate(id: number): Promise<void> {
    fixture.componentRef.setInput('candidateId', id);
    await settleZoneless(fixture);
  }

  function viewer(): ImageViewerComponent | null {
    const found = fixture.debugElement.query(By.directive(ImageViewerComponent));
    return found ? (found.componentInstance as ImageViewerComponent) : null;
  }

  function button(label: string): HTMLButtonElement {
    const all = Array.from(fixture.nativeElement.querySelectorAll('button')) as HTMLButtonElement[];
    return all.find((b) => b.textContent?.trim() === label)!;
  }

  function text(): string {
    return (fixture.nativeElement.textContent ?? '').replace(/\s+/g, ' ');
  }

  it('shows the best reference, and asks for the pile\'s qualities', () => {
    expect(viewer()!.media().id).toBe(1);
    expect(text()).toContain('Reference 1 of 3');
    expect(refs.loadQualities).toHaveBeenCalled();
  });

  it('steps to the next reference and wraps, without a new candidate', async () => {
    button('Next').click();
    await settleZoneless(fixture);
    expect(viewer()!.media().id).toBe(2);
    expect(text()).toContain('Reference 2 of 3');
    expect(text()).toContain('two.jpg');

    fixture.componentInstance.next();
    fixture.componentInstance.next();
    await settleZoneless(fixture);
    expect(viewer()!.media().id).toBe(1);
  });

  it('starts the next candidate on the best reference again', async () => {
    fixture.componentInstance.next();
    await candidate(11);
    expect(viewer()!.media().id).toBe(1);
  });

  it('holds the reference stepped to while the pile re-ranks around it', async () => {
    fixture.componentInstance.next();
    pool.set([media(3), media(2), media(1)]);
    await settleZoneless(fixture);
    expect(viewer()!.media().id).toBe(2);
  });

  it('never offers the candidate as its own reference', async () => {
    await candidate(1);
    expect(viewer()!.media().id).toBe(2);
    expect(text()).toContain('Reference 1 of 2');
  });

  it('starts every candidate on the pinned reference, and still steps from it', async () => {
    fixture.componentInstance.next();
    fixture.componentInstance.next();
    await settleZoneless(fixture);
    button('Pin').click();
    await settleZoneless(fixture);
    expect(refs.pin).toHaveBeenCalledWith(media(3));
    expect(button('Pinned').getAttribute('aria-pressed')).toBe('true');

    await candidate(11);
    expect(viewer()!.media().id).toBe(3);
    expect(text()).toContain('Reference 1 of 3');

    fixture.componentInstance.next();
    await settleZoneless(fixture);
    expect(viewer()!.media().id).toBe(1);
    expect(button('Pin').getAttribute('aria-pressed')).toBe('false');
  });

  it('unpins from the pinned reference', async () => {
    pinned.set(media(2));
    await settleZoneless(fixture);
    button('Pinned').click();
    expect(refs.unpin).toHaveBeenCalled();
  });

  it('shows an example out of its own file, full size', async () => {
    pool.set([{ kind: 'example', filename: 'my face.jpg' }]);
    await settleZoneless(fixture);
    expect(viewer()!.srcOverride()).toBe('/api/server-media-files/my%20face.jpg/image');
    expect(viewer()!.imageSrc()).toBe('/api/server-media-files/my%20face.jpg/image');
    expect(text()).toContain('Your example');
    expect(button('Next').disabled).toBe(true);
  });

  it('says so when there is no reference yet', async () => {
    pool.set([]);
    await settleZoneless(fixture);
    expect(viewer()).toBeNull();
    expect(text()).toContain('No reference face yet');
  });

  describe('Wider', () => {
    async function wider(on: boolean): Promise<void> {
      fixture.componentRef.setInput('wider', on);
      await settleZoneless(fixture);
    }

    it('asks for nothing while Wider is off', () => {
      expect(lookup).not.toHaveBeenCalled();
      expect(viewer()!.fitBox()).toBeNull();
    });

    it('shows the reference from its photo, framed wide on it, and keeps its caption', async () => {
      await wider(true);
      expect(lookup).toHaveBeenCalledWith(1);
      expect(viewer()!.media().id).toBe(10);
      expect(viewer()!.datasetId()).toBe('photos');
      const box = viewer()!.fitBox()!;
      [0.1, 0.1, 0.5, 0.5].forEach((v, i) => expect(box[i]).toBeCloseTo(v, 9));
      expect(text()).toContain('Reference 1 of 3');

      await wider(false);
      expect(viewer()!.media().id).toBe(1);
      expect(viewer()!.fitBox()).toBeNull();
    });

    it('looks up a reference once, however many candidates start on it', async () => {
      await wider(true);
      await candidate(11);
      await candidate(12);
      expect(lookup).toHaveBeenCalledTimes(1);

      fixture.componentInstance.next();
      await settleZoneless(fixture);
      expect(lookup).toHaveBeenLastCalledWith(2);
      expect(viewer()!.media().id).toBe(20);
    });

    it('keeps an example picture as it is, since no photo is behind it', async () => {
      pool.set([{ kind: 'example', filename: 'my face.jpg' }]);
      await wider(true);
      expect(lookup).not.toHaveBeenCalled();
      expect(viewer()!.srcOverride()).toBe('/api/server-media-files/my%20face.jpg/image');
      expect(viewer()!.fitBox()).toBeNull();
    });

    it('keeps a reference whose photo is not loaded on its crop', async () => {
      lookup.mockReturnValue(of<SourceLookup>({ kind: 'not-loaded', datasetId: 'photos' }));
      await wider(true);
      expect(viewer()!.media().id).toBe(1);
      expect(viewer()!.fitBox()).toBeNull();
    });
  });
});
