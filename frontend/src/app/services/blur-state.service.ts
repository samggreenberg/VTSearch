import { DestroyRef, Injectable, WritableSignal, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subscription } from 'rxjs';
import { map } from 'rxjs/operators';

import { adaptivePoll } from './adaptive-poll';
import { BlurApiService } from './blur-api.service';
import { SettingsStateService } from './settings-state.service';

/** Poll cadence while a dataset's blurred set keeps changing, and once it has settled. */
const POLL_FAST_MS = 5_000;
const POLL_SLOW_MS = 30_000;

/**
 * How long a dataset nobody shows keeps its poll. A popup that mounts and
 * unmounts its thumbnails on hover (Browse's bin popup) would otherwise stop
 * and restart the poll, with a fresh GET, every time the pointer crossed a cell.
 */
const RELEASE_GRACE_MS = 10_000;

/** A Blur / Unblur click, laid over the server's set until a read reflects it. */
interface Overlay {
  /** The state the click asked for, or, once confirmed, the one the server recorded. */
  readonly blurred: boolean;
  /** Which write the overlay belongs to; a newer click on the same item replaces it. */
  readonly write: number;
  /**
   * `null` while the PUT is in flight. Once the server has confirmed it, the
   * newest read issued by then: a read issued *after* that already reflects
   * the write and retires the overlay, while one issued before it may predate
   * the write and must not undo it.
   */
  readonly confirmedAfterRead: number | null;
}

/** What a dataset's images show: the server's set, with the user's clicks over it. */
interface BlurView {
  /** The ids the server says show blurred; `null` until the first read lands. */
  readonly ids: ReadonlySet<number> | null;
  readonly overlays: ReadonlyMap<number, Overlay>;
}

/** Per-dataset bookkeeping. Only `view` is rendered state, so only it is a signal. */
interface DatasetBlur {
  readonly view: WritableSignal<BlurView>;
  /** How many on-screen sites show this dataset's images ({@link BlurStateService.retain}). */
  refs: number;
  poll: Subscription | null;
  stopTimer: ReturnType<typeof setTimeout> | null;
  /** Issue order of reads, and the newest one applied. */
  readSeq: number;
  appliedRead: number;
}

const NO_OVERLAYS: ReadonlyMap<number, Overlay> = new Map();

function sameIds(a: ReadonlySet<number> | null, b: readonly number[]): boolean {
  if (a === null || a.size !== b.length) return false;
  return b.every((id) => a.has(id));
}

/**
 * Which images show blurred, per dataset (#5689).
 *
 * The server decides each item's effective state, its Blur / Unblur override
 * or else the `blur_rules` (`GET /api/medias/blurred`); this store holds the
 * answer for every dataset an image site has {@link retain}ed, polls it so
 * other clients' clicks arrive, and lays this tab's own clicks over it
 * optimistically ({@link setBlurred}). It is keyed by dataset rather than
 * following the active one, because several sites show a sibling dataset's
 * items (a face's source photo).
 *
 * **It fails closed.** While the instance may have blur rules (they are set,
 * or the settings have not loaded yet) and a dataset's set has not arrived,
 * every image of that dataset reads as blurred: otherwise the first paint
 * would show what the rules are there to hide. With no rules, an item is
 * blurred only by an override, so the wait shows images sharp.
 *
 * Sites read it through the `vtBlur` directive, which retains for them; a
 * site that cannot carry the directive (the Browse canvas) calls
 * {@link retain} and {@link isBlurred} itself.
 */
@Injectable({ providedIn: 'root' })
export class BlurStateService {
  private readonly api = inject(BlurApiService);
  private readonly settings = inject(SettingsStateService);
  private readonly destroyRef = inject(DestroyRef);

  private readonly datasets = new Map<string, DatasetBlur>();
  private writeSeq = 0;

  /**
   * Whether an image whose state is not known yet must be drawn blurred:
   * the instance has blur rules, or the settings that would say so have not
   * arrived.
   */
  private readonly failClosed = computed(() => {
    const settings = this.settings.settingsSignal();
    return settings === null || (settings.blur_rules?.length ?? 0) > 0;
  });

  constructor() {
    this.destroyRef.onDestroy(() => {
      for (const entry of this.datasets.values()) {
        entry.poll?.unsubscribe();
        if (entry.stopTimer !== null) clearTimeout(entry.stopTimer);
      }
    });
  }

  /**
   * Whether item *id* of *datasetId* shows blurred. A signal read: a template,
   * `computed` or `effect` that calls it follows every change.
   *
   * It only answers from what has been loaded, and nothing is loaded for a
   * dataset no site has {@link retain}ed, so on an instance with rules such a
   * dataset reads as wholly blurred.
   */
  isBlurred(datasetId: string, id: number): boolean {
    if (!datasetId) return this.failClosed();
    const view = this.entry(datasetId).view();
    const overlay = view.overlays.get(id);
    if (overlay) return overlay.blurred;
    if (view.ids === null) return this.failClosed();
    return view.ids.has(id);
  }

  /**
   * Blur or unblur item *id* of *datasetId* for good: the Blur / Unblur
   * button. Every view of the item follows at once; the server's answer then
   * settles it. A failed write is rolled back (and toasted by the error
   * interceptor), and the set read again.
   */
  setBlurred(datasetId: string, id: number, blurred: boolean): void {
    if (!datasetId) return;
    const entry = this.entry(datasetId);
    const write = ++this.writeSeq;
    this.putOverlay(entry, id, { blurred, write, confirmedAfterRead: null });
    this.api
      .setBlurredIn(datasetId, id, blurred)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (resp) => {
          // A later click on the item owns it now; its own answer settles it.
          if (entry.view().overlays.get(id)?.write !== write) return;
          this.putOverlay(entry, id, { blurred: resp.blurred, write, confirmedAfterRead: entry.readSeq });
        },
        error: () => {
          if (entry.view().overlays.get(id)?.write === write) this.dropOverlay(entry, id);
          this.refresh(datasetId, entry);
        },
      });
  }

  /**
   * Say that a site is showing images of *datasetId*: its blurred set is
   * loaded, and polled until every site that retained it has released it.
   * Returns the release, which is safe to call more than once.
   */
  retain(datasetId: string): () => void {
    if (!datasetId) return () => {};
    const entry = this.entry(datasetId);
    entry.refs++;
    if (entry.stopTimer !== null) {
      clearTimeout(entry.stopTimer);
      entry.stopTimer = null;
    }
    if (!entry.poll) this.startPoll(datasetId, entry);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      if (--entry.refs > 0) return;
      entry.stopTimer = setTimeout(() => {
        entry.stopTimer = null;
        entry.poll?.unsubscribe();
        entry.poll = null;
      }, RELEASE_GRACE_MS);
    };
  }

  private entry(datasetId: string): DatasetBlur {
    let entry = this.datasets.get(datasetId);
    if (!entry) {
      entry = {
        view: signal<BlurView>({ ids: null, overlays: NO_OVERLAYS }),
        refs: 0,
        poll: null,
        stopTimer: null,
        readSeq: 0,
        appliedRead: 0,
      };
      this.datasets.set(datasetId, entry);
    }
    return entry;
  }

  private startPoll(datasetId: string, entry: DatasetBlur): void {
    // adaptivePoll runs one read at a time, eases to a heartbeat once the set
    // stops changing, pauses while the tab is hidden, and absorbs a failed
    // read: the set stays as it was (still unknown, so still blurred, before
    // the first one lands) until a later read succeeds.
    entry.poll = adaptivePoll(
      () => {
        const seq = ++entry.readSeq;
        return this.api.getBlurredIn(datasetId).pipe(map((ids) => ({ ids, seq })));
      },
      { fastMs: POLL_FAST_MS, slowMs: POLL_SLOW_MS, signature: ({ ids }) => ids.join(',') },
    )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(({ ids, seq }) => this.applyRead(entry, ids, seq));
  }

  /** One read outside the poll, after a failed write. */
  private refresh(datasetId: string, entry: DatasetBlur): void {
    const seq = ++entry.readSeq;
    this.api
      .getBlurredIn(datasetId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({ next: (ids) => this.applyRead(entry, ids, seq), error: () => {} });
  }

  /**
   * Adopt the server's set from read *seq*, unless a newer read already
   * landed. It retires each confirmed click the read was issued after; an
   * in-flight click, or one confirmed after the read was issued, stays over it.
   */
  private applyRead(entry: DatasetBlur, ids: readonly number[], seq: number): void {
    if (seq <= entry.appliedRead) return;
    entry.appliedRead = seq;
    const view = entry.view();
    let overlays = view.overlays;
    for (const [id, overlay] of view.overlays) {
      if (overlay.confirmedAfterRead !== null && overlay.confirmedAfterRead < seq) {
        if (overlays === view.overlays) overlays = new Map(view.overlays);
        (overlays as Map<number, Overlay>).delete(id);
      }
    }
    // Leave the signal alone when nothing moved: every image of the dataset
    // re-reads it, and most polls change nothing.
    if (overlays === view.overlays && sameIds(view.ids, ids)) return;
    entry.view.set({ ids: sameIds(view.ids, ids) ? view.ids : new Set(ids), overlays });
  }

  private putOverlay(entry: DatasetBlur, id: number, overlay: Overlay): void {
    const view = entry.view();
    entry.view.set({ ids: view.ids, overlays: new Map(view.overlays).set(id, overlay) });
  }

  private dropOverlay(entry: DatasetBlur, id: number): void {
    const view = entry.view();
    const overlays = new Map(view.overlays);
    overlays.delete(id);
    entry.view.set({ ids: view.ids, overlays });
  }
}
