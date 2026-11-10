import { Injectable, inject } from '@angular/core';
import { ActiveContextService } from './active-context.service';
import { MAX_THUMBS, ThumbStore } from '../components/browse-canvas/thumb-store';

/** What the Browse canvas painting from the store lends it. */
export interface ThumbHost {
  /** Repaint: a thumbnail on screen has landed (or failed). */
  onLoaded(): void;
  /** The live accent colour waveform masks are tinted with. */
  accent(): string;
}

/** How many thumbnails outlive the canvas that loaded them. */
export const RETAINED_THUMBS = MAX_THUMBS / 2;

/**
 * The Browse canvas's thumbnail cache, kept from one visit to the map to the
 * next (#5637).
 *
 * Thumbnails reach the canvas in batches (see `ThumbStore`), as Blob URLs, so
 * the browser no longer reuses the images it decoded for the last visit: the
 * batch responses come back from its HTTP cache, but every thumbnail is new to
 * it and decoded again, which put half a second of grey bins on a revisit that
 * used to paint at once. Keeping the decoded images here, rather than on the
 * canvas, makes a revisit paint at once again, without asking for anything.
 *
 * The cache holds one dataset's thumbnails (media ids are per dataset); a map
 * of another dataset drops it. When the canvas goes away the cache is trimmed
 * to the {@link RETAINED_THUMBS} most recently painted, which covers an
 * opening view, so leaving Browse doesn't hold a long pan session's worth.
 */
@Injectable({ providedIn: 'root' })
export class BrowseThumbsService {
  private activeContext = inject(ActiveContextService);
  /** The canvas painting from the store now, if any. */
  private host: ThumbHost | null = null;
  /** The dataset the cached thumbnails belong to. */
  private datasetId: string | null = null;

  readonly store = new ThumbStore({
    mediaUrl: (path) => this.activeContext.mediaUrl(path),
    batchUrl: (ids) => this.activeContext.mediaUrl('/api/medias/thumbnails', { ids: ids.join(',') }),
    onLoaded: () => this.host?.onLoaded(),
    accent: () => this.host?.accent() ?? '',
  });

  /** A canvas starts painting from the store. */
  attach(host: ThumbHost): ThumbStore {
    this.host = host;
    // The theme may have flipped while no canvas was watching it.
    this.store.clearTinted();
    return this.store;
  }

  /** A canvas is going away. A newer one may already have attached, and
   *  keeps the store as it is. */
  detach(host: ThumbHost): void {
    if (this.host !== host) return;
    this.host = null;
    this.store.trim(RETAINED_THUMBS);
  }

  /** Hold thumbnails for this dataset only: switching to another drops them. */
  useDataset(datasetId: string): void {
    if (datasetId === this.datasetId) return;
    this.datasetId = datasetId;
    this.store.clear();
  }
}
