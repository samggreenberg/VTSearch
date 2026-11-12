import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { Observable, Subject } from 'rxjs';

import { FACE_QUALITY_KEY, FaceReferenceService, orderByQuality } from './face-reference.service';
import { ActiveDatasetService } from './active-dataset.service';
import { ActiveDetectorService } from './active-detector.service';
import { MediaMetadataCacheService } from './media-metadata-cache.service';
import { SortingApiService } from './sorting-api.service';
import { VoteStateService } from './vote-state.service';
import { configureZoneless } from '../testing/zoneless-testbed';

describe('orderByQuality', () => {
  it('puts the best quality first, which leaves the hard views last', () => {
    // 14 is the hard-view cut of the space these are read in: 9 and 3 are hard views.
    const quality: Record<number, number> = { 1: 9, 2: 31, 3: 22, 4: 3 };
    expect(orderByQuality([1, 2, 3, 4], (id) => quality[id])).toEqual([2, 3, 1, 4]);
  });

  it('puts a face with no quality read after every face with one, in id order', () => {
    const quality: Record<number, unknown> = { 5: 2, 7: 'n/a' };
    expect(orderByQuality([9, 7, 5, 8], (id) => quality[id])).toEqual([5, 7, 8, 9]);
  });

  it('breaks a tie by id, so the order holds between reads', () => {
    expect(orderByQuality([3, 1, 2], () => 20)).toEqual([1, 2, 3]);
  });
});

/**
 * The reference set of the Face compare layout (#5642): the detector's
 * examples until there is a Good pile, then the pile by Face Quality; and the
 * pin, which lasts as long as what it names.
 */
describe('FaceReferenceService', () => {
  let service: FaceReferenceService;
  const goods = signal(new Set<number>());
  const version = signal(0);
  const datasetId = signal('faces');
  const detectorId = signal('det');
  const detector = signal<{ examples?: Array<Record<string, unknown>> } | null>(null);
  let quality: Record<number, number>;
  let hard: Record<number, boolean>;
  let ensureLoaded: ReturnType<typeof vi.fn<(ids: number[]) => void>>;
  /** The server's answers about each example picture, sent when a test says. */
  let verdicts: Record<string, Subject<boolean | null>>;
  let exampleFaceHardView: ReturnType<typeof vi.fn<(filename: string) => Observable<boolean | null>>>;

  beforeEach(() => {
    goods.set(new Set());
    version.set(0);
    datasetId.set('faces');
    detectorId.set('det');
    detector.set({
      examples: [
        { type: 'text', value: 'a person' },
        { type: 'media', value: 'a.jpg' },
        { type: 'media', value: 'seed.jpg', labeled: false },
        { type: 'media', value: 'b.jpg', labeled: true },
        { type: 'media', value: 'a.jpg' },
      ],
    });
    quality = {};
    hard = {};
    ensureLoaded = vi.fn();
    verdicts = {};
    exampleFaceHardView = vi.fn((filename: string) => (verdicts[filename] = new Subject<boolean | null>()));
    configureZoneless({
      providers: [
        { provide: VoteStateService, useValue: { get goodVotes() { return goods(); } } },
        {
          provide: MediaMetadataCacheService,
          useValue: {
            version,
            ensureLoaded,
            get: (id: number) =>
              id in quality ? { id, custom_metadata: { [FACE_QUALITY_KEY]: quality[id] }, face_hard_view: hard[id] } : undefined,
          },
        },
        { provide: SortingApiService, useValue: { exampleFaceHardView } },
        { provide: ActiveDatasetService, useValue: { activeId: datasetId } },
        { provide: ActiveDetectorService, useValue: { activeId: detectorId, detector } },
      ],
    });
    service = TestBed.inject(FaceReferenceService);
  });

  it('offers the labeled example pictures before any vote, once each, and no seed', () => {
    expect(service.pool()).toEqual([
      { kind: 'example', filename: 'a.jpg' },
      { kind: 'example', filename: 'b.jpg' },
    ]);
  });

  it('offers the Good pile in place of the examples once there is one', () => {
    quality = { 4: 30, 6: 5, 9: 18 };
    goods.set(new Set([6, 4, 9]));
    expect(service.pool()).toEqual([
      { kind: 'media', id: 4 },
      { kind: 'media', id: 9 },
      { kind: 'media', id: 6 },
    ]);
  });

  it('re-ranks the pile when its metadata arrives', () => {
    goods.set(new Set([1, 2]));
    expect(service.pool().map((r) => r.kind === 'media' && r.id)).toEqual([1, 2]);
    quality = { 1: 10, 2: 40 };
    version.set(1);
    expect(service.pool().map((r) => r.kind === 'media' && r.id)).toEqual([2, 1]);
  });

  it('asks for the metadata of the pile, and for nothing with no pile', () => {
    service.loadQualities();
    expect(ensureLoaded).not.toHaveBeenCalled();
    goods.set(new Set([3, 8]));
    service.loadQualities();
    expect(ensureLoaded).toHaveBeenCalledWith([3, 8]);
  });

  it('keeps a pin until it is unpinned', () => {
    goods.set(new Set([3, 8]));
    service.pin({ kind: 'media', id: 8 });
    expect(service.pinned()).toEqual({ kind: 'media', id: 8 });
    service.unpin();
    expect(service.pinned()).toBeNull();
  });

  it('drops a pin whose face left the Good pile', () => {
    goods.set(new Set([3, 8]));
    service.pin({ kind: 'media', id: 8 });
    goods.set(new Set([3]));
    expect(service.pinned()).toBeNull();
  });

  it('keeps a pinned example while the Good pile grows past it', () => {
    service.pin({ kind: 'example', filename: 'a.jpg' });
    goods.set(new Set([3]));
    expect(service.pinned()).toEqual({ kind: 'example', filename: 'a.jpg' });
    detector.set({ examples: [] });
    expect(service.pinned()).toBeNull();
  });

  describe('hard views (#5643)', () => {
    it('reads a face of the dataset\'s verdict off its metadata, and asks the server nothing', () => {
      quality = { 4: 9, 6: 30, 7: 20 };
      hard = { 4: true, 6: false };
      expect(service.hardView({ kind: 'media', id: 4 })).toBe(true);
      expect(service.hardView({ kind: 'media', id: 6 })).toBe(false);
      // No verdict stamped, and no metadata yet.
      expect(service.hardView({ kind: 'media', id: 7 })).toBeNull();
      expect(service.hardView({ kind: 'media', id: 8 })).toBeNull();
      service.loadHardView({ kind: 'media', id: 4 });
      expect(exampleFaceHardView).not.toHaveBeenCalled();
    });

    it('asks the server about an example picture once, and has no verdict until it answers', () => {
      const ref = { kind: 'example', filename: 'a.jpg' } as const;
      service.loadHardView(ref);
      service.loadHardView(ref);
      expect(exampleFaceHardView).toHaveBeenCalledTimes(1);
      expect(exampleFaceHardView).toHaveBeenCalledWith('a.jpg');
      expect(service.hardView(ref)).toBeNull();

      verdicts['a.jpg'].next(true);
      expect(service.hardView(ref)).toBe(true);
      expect(service.hardView({ kind: 'example', filename: 'b.jpg' })).toBeNull();
      service.loadHardView(ref);
      expect(exampleFaceHardView).toHaveBeenCalledTimes(1);
    });
  });

  it('keeps one pin per dataset and detector', () => {
    goods.set(new Set([3]));
    service.pin({ kind: 'media', id: 3 });
    detectorId.set('other');
    expect(service.pinned()).toBeNull();
    detectorId.set('det');
    expect(service.pinned()).toEqual({ kind: 'media', id: 3 });
  });
});
