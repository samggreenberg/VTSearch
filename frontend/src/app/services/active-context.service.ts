import { Injectable } from '@angular/core';
import { BehaviorSubject, combineLatest, Observable } from 'rxjs';
import { distinctUntilChanged, map } from 'rxjs/operators';

/**
 * Tracks which dataset and model the user is currently working with.
 *
 * The HTTP interceptor (`activeContextInterceptor`) reads from this
 * service and attaches `X-Dataset-Id` / `X-Detector-Id` headers to every
 * outgoing API request so the backend resolves the correct context
 * per-request — no server-side "active" state needed.
 */
@Injectable({ providedIn: 'root' })
export class ActiveContextService {
  private readonly datasetIdSubject = new BehaviorSubject<string>('');
  private readonly modelIdSubject = new BehaviorSubject<string>('');
  private readonly pairSubject = new BehaviorSubject<{ datasetId: string; modelId: string }>({
    datasetId: '',
    modelId: '',
  });
  private requestCounter = 0;

  readonly datasetId$ = this.datasetIdSubject.asObservable();
  readonly modelId$ = this.modelIdSubject.asObservable();

  /**
   * Emits whenever either half of the pair changes. Use this when a view
   * needs to react to a swap — subscribing to the two halves separately
   * would fire twice for an atomic `setActivePair()` call.
   */
  readonly pair$: Observable<{ datasetId: string; modelId: string }> = this.pairSubject.pipe(
    distinctUntilChanged((a, b) => a.datasetId === b.datasetId && a.modelId === b.modelId),
  );

  /**
   * Emits whenever the (datasetId, modelId) tuple changes, as a joined key
   * (`<datasetId>::<modelId>`). Useful as a `switchMap` trigger.
   */
  readonly pairKey$: Observable<string> = combineLatest([this.datasetId$, this.modelId$]).pipe(
    map(([d, m]) => `${d}::${m}`),
    distinctUntilChanged(),
  );

  get datasetId(): string {
    return this.datasetIdSubject.value;
  }

  get modelId(): string {
    return this.modelIdSubject.value;
  }

  setDatasetId(id: string): void {
    if (this.datasetIdSubject.value === id) return;
    this.datasetIdSubject.next(id);
    this.pairSubject.next({ datasetId: id, modelId: this.modelIdSubject.value });
  }

  setModelId(id: string): void {
    if (this.modelIdSubject.value === id) return;
    this.modelIdSubject.next(id);
    this.pairSubject.next({ datasetId: this.datasetIdSubject.value, modelId: id });
  }

  /**
   * Set both halves of the active pair in a single change. Avoids the
   * transient mismatched-pair window that would result from two separate
   * setter calls fighting through the HTTP interceptor.
   */
  setActivePair(datasetId: string, modelId: string): void {
    const dsChanged = this.datasetIdSubject.value !== datasetId;
    const mChanged = this.modelIdSubject.value !== modelId;
    if (!dsChanged && !mChanged) return;
    if (dsChanged) this.datasetIdSubject.next(datasetId);
    if (mChanged) this.modelIdSubject.next(modelId);
    this.pairSubject.next({ datasetId, modelId });
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
   *
   * See `docs/plans/active-context-switcher.md` § "Cancel-and-replace
   * on rapid re-click".
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
