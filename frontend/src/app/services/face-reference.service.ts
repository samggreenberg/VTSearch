import { Injectable, computed, inject, signal } from '@angular/core';
import { ActiveDatasetService } from './active-dataset.service';
import { ActiveDetectorService } from './active-detector.service';
import { MediaMetadataCacheService } from './media-metadata-cache.service';
import { VoteStateService } from './vote-state.service';

/**
 * A face to compare the candidate against (#5642): an item of the active
 * dataset voted Good, or one of the detector's example pictures, which is a
 * file of the user's and no dataset's item.
 */
export type FaceReference =
  | { readonly kind: 'media'; readonly id: number }
  | { readonly kind: 'example'; readonly filename: string };

/** The `custom_metadata` key a face's converter stamps its quality under
 *  (`QUALITY_METADATA_KEY` in `vtscore/media/face/imagekit_bridge.py`). */
export const FACE_QUALITY_KEY = 'Face Quality';

/** A string that names *ref* among the references of one dataset and detector. */
export function referenceKey(ref: FaceReference): string {
  return ref.kind === 'media' ? `m:${ref.id}` : `e:${ref.filename}`;
}

/**
 * The Good pile as references, best Face Quality first.
 *
 * Hard views come last by the same sort: a hard view is a face whose quality
 * is under its space's cut (`HARD_VIEW_QUALITY`), and a dataset's faces share
 * one space, so every hard view already ranks below every clear one. A face
 * with no quality read yet (its metadata is still on its way) goes after
 * both. Ties keep id order, so the list does not shuffle between reads.
 */
export function orderByQuality(ids: Iterable<number>, quality: (id: number) => unknown): number[] {
  const rank = (id: number): number => {
    const q = quality(id);
    return typeof q === 'number' && Number.isFinite(q) ? q : -Infinity;
  };
  return [...ids].sort((a, b) => rank(b) - rank(a) || a - b);
}

/**
 * The reference faces of the Face compare layout (#5642), and the one the user
 * pinned.
 *
 * The set is the Good pile once there is one, and the detector's example
 * pictures before that: the examples are what the user said the person looks
 * like until their own votes say it better. An unlabeled seed is left out; it
 * steers a sort and was never claimed to be the person.
 *
 * The pin is kept here, in memory, for as long as the page lives. It is not
 * written to the detector (the issue's decision): a pin is a convenience of
 * one sitting, and saving it would add a field to detector JSON for that.
 */
@Injectable({ providedIn: 'root' })
export class FaceReferenceService {
  private readonly voteState = inject(VoteStateService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly activeDataset = inject(ActiveDatasetService);
  private readonly activeDetector = inject(ActiveDetectorService);

  /** The pinned reference per `<dataset>::<detector>`. */
  private readonly pins = signal<Readonly<Record<string, FaceReference>>>({});

  private readonly contextKey = computed(() => `${this.activeDataset.activeId()}::${this.activeDetector.activeId()}`);

  /** The active detector's labeled example pictures, in the order it lists them. */
  private readonly examples = computed<FaceReference[]>(() => {
    const out: FaceReference[] = [];
    const seen = new Set<string>();
    for (const ex of this.activeDetector.detector()?.examples ?? []) {
      const filename = ex['type'] === 'media' && typeof ex['value'] === 'string' ? ex['value'] : '';
      if (!filename || ex['labeled'] === false || seen.has(filename)) continue;
      seen.add(filename);
      out.push({ kind: 'example', filename });
    }
    return out;
  });

  /** The Good pile, best Face Quality first (see {@link orderByQuality}). */
  private readonly goods = computed<FaceReference[]>(() => {
    // Media ids, and so the cache's rows, are per dataset.
    this.activeDataset.activeId();
    // A batch of metadata landing re-ranks the pile.
    this.metadataCache.version();
    const ids = orderByQuality(
      this.voteState.goodVotes,
      (id) => this.metadataCache.get(id)?.custom_metadata?.[FACE_QUALITY_KEY],
    );
    return ids.map((id) => ({ kind: 'media', id }));
  });

  /** The references to step through: the Good pile, or the examples before it. */
  readonly pool = computed<FaceReference[]>(() => {
    const goods = this.goods();
    return goods.length ? goods : this.examples();
  });

  /**
   * The pinned reference, or null. A pin lapses with what it named: a face
   * voted off the Good pile, or an example taken off the detector.
   */
  readonly pinned = computed<FaceReference | null>(() => {
    const pin = this.pins()[this.contextKey()];
    if (!pin) return null;
    const key = referenceKey(pin);
    const live =
      pin.kind === 'media' ? this.voteState.goodVotes.has(pin.id) : this.examples().some((ex) => referenceKey(ex) === key);
    return live ? pin : null;
  });

  /** Ask for the Good pile's metadata, which carries each face's quality. */
  loadQualities(): void {
    const ids = [...this.voteState.goodVotes];
    if (ids.length) this.metadataCache.ensureLoaded(ids);
  }

  pin(ref: FaceReference): void {
    this.pins.update((pins) => ({ ...pins, [this.contextKey()]: ref }));
  }

  unpin(): void {
    const key = this.contextKey();
    this.pins.update((pins) => {
      const rest = { ...pins };
      delete rest[key];
      return rest;
    });
  }
}
