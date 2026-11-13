import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { Observable, Subject } from 'rxjs';

import { FACE_QUALITY_KEY, FaceReferenceService, VIEW_STEP, orderByQuality, orderByView } from './face-reference.service';
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

describe('orderByView (#5670)', () => {
  // A pile already in quality order, and how far each head is turned.
  const ranked = [1, 2, 3, 4, 5];
  const turns: Record<number, number | null> = { 1: 0, 2: 0.55, 3: -0.5, 4: 0.6, 5: null };
  const turn = (id: number) => turns[id] ?? null;

  it('puts the faces turned like the candidate first, the clearest of them first', () => {
    // 2 and 4 are turned the candidate's way; 1 looks ahead, and 3 is turned the other way.
    expect(orderByView(ranked, 0.6, turn)).toEqual([2, 4, 1, 3, 5]);
    expect(orderByView(ranked, -0.45, turn)).toEqual([3, 1, 2, 4, 5]);
  });

  it('counts faces less than a step apart as one view, and keeps their quality order', () => {
    expect(orderByView([1, 2], 0.1, (id) => (id === 1 ? 0.1 + VIEW_STEP * 0.9 : 0.1))).toEqual([1, 2]);
    expect(orderByView([1, 2], 0.1, (id) => (id === 1 ? 0.1 + VIEW_STEP * 1.1 : 0.1))).toEqual([2, 1]);
  });

  it('puts a face with no turn to read last, and keeps the quality order with no candidate turn', () => {
    expect(orderByView([5, 1], 0, turn)).toEqual([1, 5]);
    expect(orderByView(ranked, null, turn)).toEqual(ranked);
  });
});

/**
 * The reference set of the Face compare layout (#5642): the detector's
 * examples until there is a Good pile, then the pile by Face Quality, or for a
 * candidate turned like it first (#5670); and the pin, which lasts as long as
 * what it names.
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
  let landmarks: Record<number, number[][]>;
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
    landmarks = {};
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
              id in quality
                ? { id, custom_metadata: { [FACE_QUALITY_KEY]: quality[id] }, face_hard_view: hard[id], face_landmarks: landmarks[id] }
                : undefined,
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

  it('orders the pile for a candidate by which way each head is turned (#5670)', () => {
    // Eyes level, mouth below them, and the nose moved off the line between
    // them by *nose*: toward the image's left is a turn that way.
    const turned = (nose: number) => [[0.4, 0.4], [0.6, 0.4], [0.5 + nose, 0.5], [0.45, 0.6], [0.55, 0.6]];
    quality = { 4: 30, 6: 5, 9: 18 };
    landmarks = { 4: turned(0), 6: turned(-0.08), 9: turned(0.08) };
    goods.set(new Set([6, 4, 9]));
    const ids = (turn: number | null) => service.pool(turn).map((r) => r.kind === 'media' && r.id);
    // With no turn for the candidate, the quality order.
    expect(ids(null)).toEqual([4, 9, 6]);
    expect(ids(-0.7)).toEqual([6, 4, 9]);
    expect(ids(0.7)).toEqual([9, 4, 6]);
  });

  it('keeps the examples\' order for any candidate, since they have no landmarks here', () => {
    expect(service.pool(0.7)).toEqual(service.pool());
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
