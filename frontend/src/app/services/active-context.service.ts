import { Injectable } from '@angular/core';
import { BehaviorSubject, combineLatest, Observable } from 'rxjs';
import { distinctUntilChanged, map } from 'rxjs/operators';

/**
 * Tracks which dataset and model the user is currently working with.
 *
 * Split into two layers to fix the H25 race where the HTTP interceptor
 * would otherwise tag outgoing requests with an id the backend hadn't
 * finished loading (cascade of 409 `dataset_not_loaded`):
 *
 *  - **intent**: what the user has selected (pulldown click, deep-link
 *    URL).  Updates immediately so UI affordances like the pulldown
 *    highlight reflect the selection without delay.
 *  - **active**: the pair currently loaded into the backend, against
 *    which API requests are dispatched.  The HTTP interceptor
 *    (`activeContextInterceptor`) attaches `X-Dataset-Id` /
 *    `X-Detector-Id` from this layer, so it lags `intent` until
 *    `ContextSwitchService` has finished any required dataset / detector
 *    load and explicitly promotes the pair via `setActive()`.
 *
 * `setActivePair()` writes BOTH layers; used by cleanup paths (the
 * registry watcher, `clear()`) where there is nothing to wait for.  UI
 * code that wants to *change* the pair should go through
 * `ContextSwitchService.switchTo()` so loads run first.
 */
@Injectable({ providedIn: 'root' })
export class ActiveContextService {
  // --- ACTIVE layer (what the HTTP interceptor reads) ---------------------
  private readonly datasetIdSubject = new BehaviorSubject<string>('');
  private readonly modelIdSubject = new BehaviorSubject<string>('');
  private readonly pairSubject = new BehaviorSubject<{ datasetId: string; modelId: string }>({
    datasetId: '',
    modelId: '',
  });

  // --- INTENT layer (what the user picked) -------------------------------
  private readonly intentDatasetIdSubject = new BehaviorSubject<string>('');
  private readonly intentModelIdSubject = new BehaviorSubject<string>('');
  private readonly intentPairSubject = new BehaviorSubject<{
    datasetId: string;
    modelId: string;
  }>({
    datasetId: '',
    modelId: '',
  });

  private requestCounter = 0;

  readonly datasetId$ = this.datasetIdSubject.asObservable();
  readonly modelId$ = this.modelIdSubject.asObservable();
  readonly intentDatasetId$ = this.intentDatasetIdSubject.asObservable();
  readonly intentModelId$ = this.intentModelIdSubject.asObservable();

  /**
   * Emits whenever either half of the *active* pair changes.  Use this
   * when a view needs to react to the actually-loaded pair (e.g. media
   * URLs, cache invalidation).  Subscribing to the two halves separately
   * would fire twice for an atomic `setActivePair()` call.
   */
  readonly pair$: Observable<{ datasetId: string; modelId: string }> = this.pairSubject.pipe(
    distinctUntilChanged((a, b) => a.datasetId === b.datasetId && a.modelId === b.modelId),
  );

  /**
   * Emits whenever either half of the user's *intent* changes, i.e.
   * the moment a pulldown row is clicked, before any load has run.
   * Use this for UI affordances that should reflect the user's pick
   * without waiting (pulldown highlight, "you picked X" labels).
   */
  readonly intentPair$: Observable<{ datasetId: string; modelId: string }> =
    this.intentPairSubject.pipe(
      distinctUntilChanged((a, b) => a.datasetId === b.datasetId && a.modelId === b.modelId),
    );

  /** Active-pair tuple as a joined key (`<datasetId>::<modelId>`). */
  readonly pairKey$: Observable<string> = combineLatest([this.datasetId$, this.modelId$]).pipe(
    map(([d, m]) => `${d}::${m}`),
    distinctUntilChanged(),
  );

  /** Intent-pair tuple as a joined key. */
  readonly intentPairKey$: Observable<string> = combineLatest([
    this.intentDatasetId$,
    this.intentModelId$,
  ]).pipe(
    map(([d, m]) => `${d}::${m}`),
    distinctUntilChanged(),
  );

  get datasetId(): string {
    return this.datasetIdSubject.value;
  }

  get modelId(): string {
    return this.modelIdSubject.value;
  }

  get intentDatasetId(): string {
    return this.intentDatasetIdSubject.value;
  }

  get intentModelId(): string {
    return this.intentModelIdSubject.value;
  }

  /**
   * Set the user's intent.  Called by `ContextSwitchService` the moment
   * a switch begins so UI affordances (pulldown highlight) reflect the
   * selection immediately, while the active pair stays pinned to the
   * still-loaded backend state.  Active is promoted separately via
   * `setActive()` once any required load finishes.
   */
  setIntent(datasetId: string, modelId: string): void {
    const dsChanged = this.intentDatasetIdSubject.value !== datasetId;
    const mChanged = this.intentModelIdSubject.value !== modelId;
    if (!dsChanged && !mChanged) return;
    if (dsChanged) this.intentDatasetIdSubject.next(datasetId);
    if (mChanged) this.intentModelIdSubject.next(modelId);
    this.intentPairSubject.next({ datasetId, modelId });
  }

  /**
   * Promote a pair to active (what the HTTP interceptor will tag onto
   * outgoing requests).  Called by `ContextSwitchService` once required
   * loads have finished (or by `setActivePair`/cleanup paths).
   */
  setActive(datasetId: string, modelId: string): void {
    const dsChanged = this.datasetIdSubject.value !== datasetId;
    const mChanged = this.modelIdSubject.value !== modelId;
    if (!dsChanged && !mChanged) return;
    if (dsChanged) this.datasetIdSubject.next(datasetId);
    if (mChanged) this.modelIdSubject.next(modelId);
    this.pairSubject.next({ datasetId, modelId });
  }

  /**
   * Set both intent and active atomically.  Used by cleanup paths that
   * are not gated on a load: the registry watcher when an active id
   * disappears, the `clear()` shortcut, and tests.  Code that wants to
   * *switch* to a new pair should go through
   * `ContextSwitchService.switchTo()` instead so the dataset/detector
   * load runs before active flips.
   */
  setActivePair(datasetId: string, modelId: string): void {
    this.setIntent(datasetId, modelId);
    this.setActive(datasetId, modelId);
  }

  clear(): void {
    this.setActivePair('', '');
  }

  /**
   * Allocate a fresh request id for a switcher-driven prep flow. Latest
   * caller wins: when a prep step completes, the caller compares the id
   * it captured at start to the current id and discards the result if
   * they differ. Cancellation of in-flight work is best-effort; this
   * request-id check is the correctness guarantee.
   */
  nextRequestId(): number {
    this.requestCounter += 1;
    return this.requestCounter;
  }

  get currentRequestId(): number {
    return this.requestCounter;
  }

  /**
   * Build a media URL with context query params so browser-native requests
   * (`<img src>`, `<audio src>`, `<video src>`) resolve the correct dataset.
   *
   * Uses the *active* pair so the id matches what the backend has
   * loaded; falling back to intent would point at an in-flight,
   * not-yet-resolved id.
   *
   * Angular HttpClient requests use the interceptor to send headers, but
   * native element `src` attributes bypass it entirely.
   */
  mediaUrl(path: string): string {
    const params: string[] = [];
    const ds = this.datasetIdSubject.value;
    if (ds) params.push(`dataset_id=${encodeURIComponent(ds)}`);
    const model = this.modelIdSubject.value;
    if (model) params.push(`detector_id=${encodeURIComponent(model)}`);
    return params.length ? `${path}?${params.join('&')}` : path;
  }
}
