import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

/**
 * Tracks which dataset and model the user is currently working with.
 *
 * The HTTP interceptor (`activeContextInterceptor`) reads from this
 * service and attaches `X-Dataset-Id` / `X-Model-Id` headers to every
 * outgoing API request so the backend resolves the correct context
 * per-request — no server-side "active" state needed.
 */
@Injectable({ providedIn: 'root' })
export class ActiveContextService {
  private readonly datasetIdSubject = new BehaviorSubject<string>('');
  private readonly modelIdSubject = new BehaviorSubject<string>('');

  readonly datasetId$ = this.datasetIdSubject.asObservable();
  readonly modelId$ = this.modelIdSubject.asObservable();

  get datasetId(): string {
    return this.datasetIdSubject.value;
  }

  get modelId(): string {
    return this.modelIdSubject.value;
  }

  setDatasetId(id: string): void {
    this.datasetIdSubject.next(id);
  }

  setModelId(id: string): void {
    this.modelIdSubject.next(id);
  }

  clear(): void {
    this.datasetIdSubject.next('');
    this.modelIdSubject.next('');
  }
}
