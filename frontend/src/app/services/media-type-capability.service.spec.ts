import { TestBed } from '@angular/core/testing';
import { Observable, Subject, of } from 'rxjs';
import {
  FALLBACK_THUMBNAIL_TYPE_IDS,
  MediaTypeCapabilityService,
} from './media-type-capability.service';
import { DatasetsListingsApiService } from './datasets-listings-api.service';
import type { MediaTypeInfo } from '../models/api.models';

type MediaTypesResponse = { media_types: MediaTypeInfo[] };

describe('MediaTypeCapabilityService', () => {
  let service: MediaTypeCapabilityService;
  let getMediaTypes: ReturnType<typeof vi.fn>;

  const infos: MediaTypeInfo[] = [
    { type_id: 'image', name: 'Image', has_thumbnail: true },
    { type_id: 'video', name: 'Video', has_thumbnail: true },
    { type_id: 'audio', name: 'Audio', has_thumbnail: true },
    { type_id: 'document', name: 'Document', has_thumbnail: true },
    { type_id: 'text', name: 'Text', has_thumbnail: false },
    { type_id: 'point_cloud', name: 'Point Cloud', has_thumbnail: true }, // a new thumbnail type
    { type_id: 'legacy', name: 'Legacy' }, // has_thumbnail undefined ⇒ no thumbnail
  ];

  beforeEach(() => {
    getMediaTypes = vi.fn((): Observable<MediaTypesResponse> => of({ media_types: infos }));
    TestBed.configureTestingModule({
      providers: [
        MediaTypeCapabilityService,
        { provide: DatasetsListingsApiService, useValue: { getMediaTypes } },
      ],
    });
    service = TestBed.inject(MediaTypeCapabilityService);
  });

  it('is side-effect-free on injection (no fetch, unloaded)', () => {
    expect(getMediaTypes).not.toHaveBeenCalled();
    expect(service.loaded).toBe(false);
    expect(service.infos()).toBeNull();
  });

  it('answers from the fallback set before the registry loads', () => {
    for (const t of FALLBACK_THUMBNAIL_TYPE_IDS) {
      expect(service.usesThumbnails(t)).toBe(true);
    }
    expect(service.usesThumbnails('text')).toBe(false);
    // A served-only new type is unknown until the registry loads.
    expect(service.usesThumbnails('point_cloud')).toBe(false);
    expect(service.usesThumbnails('')).toBe(false);
  });

  it('ensureLoaded data-drives usesThumbnails from the served has_thumbnail field', () => {
    service.ensureLoaded();

    expect(service.loaded).toBe(true);
    for (const t of ['image', 'video', 'audio', 'document']) {
      expect(service.usesThumbnails(t)).toBe(true);
    }
    // A new thumbnail type flips on purely from the served field — no frontend edit.
    expect(service.usesThumbnails('point_cloud')).toBe(true);
    // Non-thumbnail / absent-flag / unknown types are false.
    expect(service.usesThumbnails('text')).toBe(false);
    expect(service.usesThumbnails('legacy')).toBe(false);
    expect(service.usesThumbnails('mystery')).toBe(false);
  });

  it('ensureLoaded is idempotent (fetches at most once)', () => {
    service.ensureLoaded();
    service.ensureLoaded();
    expect(getMediaTypes).toHaveBeenCalledTimes(1);
  });

  it('on error marks the registry loaded and falls back to the served-empty set', () => {
    getMediaTypes.mockReturnValueOnce(
      new Observable<MediaTypesResponse>((o) => o.error(new Error('500'))),
    );
    service.ensureLoaded();
    expect(service.loaded).toBe(true);
    expect(service.infos()).toEqual([]);
    // With an empty served registry, nothing is a thumbnail type.
    expect(service.usesThumbnails('image')).toBe(false);
  });

  it('does not refetch while a load is still in flight', () => {
    const pending = new Subject<MediaTypesResponse>();
    getMediaTypes.mockReturnValue(pending.asObservable());
    service.ensureLoaded();
    service.ensureLoaded();
    expect(getMediaTypes).toHaveBeenCalledTimes(1);
    pending.next({ media_types: infos });
    expect(service.loaded).toBe(true);
  });
});
