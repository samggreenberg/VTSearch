import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';

type PulldownKind = 'dataset' | 'detector';

/**
 * Cross-component signal bus so non-pulldown UI (e.g. the
 * `vt-incompatible-pair-explainer`) can ask a specific pulldown to open
 * its menu programmatically.
 *
 * Each pulldown instance subscribes to `openSignal$(kind)` matching its
 * own `kind` and calls `openMenu()` when a signal arrives. The
 * explainer's "Pick a compatible detector/dataset" button fires
 * `requestOpen(kind)`.
 *
 * Kept intentionally tiny — no state, no behaviour beyond signal
 * dispatch — so it's safe to inject anywhere without coupling to
 * pulldown internals.
 */
@Injectable({ providedIn: 'root' })
export class PulldownControlService {
  private readonly datasetSubject = new Subject<void>();
  private readonly detectorSubject = new Subject<void>();

  openSignal$(kind: PulldownKind): Observable<void> {
    return kind === 'dataset'
      ? this.datasetSubject.asObservable()
      : this.detectorSubject.asObservable();
  }

  requestOpen(kind: PulldownKind): void {
    if (kind === 'dataset') this.datasetSubject.next();
    else this.detectorSubject.next();
  }
}
