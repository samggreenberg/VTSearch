import { Injectable } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';

export type SelectionKind = 'dataset' | 'detector';

/**
 * Bridges the Dashboard's table row-selection to the top-bar
 * `vt-context-pulldown`s so the blue bar reflects what the user has
 * highlighted, not merely what is loaded into the backend.
 *
 * Two selection concepts coexist:
 *  - **active/loaded context** (`ActiveContextService`) — the pair the
 *    backend has loaded, which the pulldown shows on the label/find/browse
 *    views (there are no tables there to select from).
 *  - **table selection** (the Dashboard's `selectedDatasetIds` /
 *    `selectedDetectorIds`) — the highlighted rows that drive Train / Find /
 *    Combine / Delete.
 *
 * While the Dashboard is on screen the pulldown mirrors the *table
 * selection* (this service); once the user navigates away the Dashboard is
 * destroyed, `dashboardVisible` flips to false, and the pulldown falls back
 * to the active/loaded context. That view-awareness rides on the Dashboard's
 * component lifecycle rather than the router, so it needs no URL parsing.
 */
@Injectable({ providedIn: 'root' })
export class DashboardSelectionService {
  /** True while the Dashboard component is mounted. Set in its
   *  `ngOnInit` / `ngOnDestroy`. */
  private readonly dashboardVisibleSubject = new BehaviorSubject<boolean>(false);
  readonly dashboardVisible$ = this.dashboardVisibleSubject.asObservable();

  /** Currently-selected dataset / detector ids, mirrored from the
   *  Dashboard's selection Sets (filtered to ids that still exist in the
   *  registry, so counts stay accurate). */
  private readonly datasetIdsSubject = new BehaviorSubject<string[]>([]);
  private readonly detectorIdsSubject = new BehaviorSubject<string[]>([]);
  readonly datasetIds$ = this.datasetIdsSubject.asObservable();
  readonly detectorIds$ = this.detectorIdsSubject.asObservable();

  /** Pulldown → Dashboard: request a plain (non-additive) single-select of
   *  one id, exactly as if the user had clicked that table row. */
  private readonly selectRequestSubject = new Subject<{ kind: SelectionKind; id: string }>();
  readonly selectRequest$ = this.selectRequestSubject.asObservable();

  get dashboardVisible(): boolean {
    return this.dashboardVisibleSubject.value;
  }

  get datasetIds(): string[] {
    return this.datasetIdsSubject.value;
  }

  get detectorIds(): string[] {
    return this.detectorIdsSubject.value;
  }

  setDashboardVisible(visible: boolean): void {
    if (this.dashboardVisibleSubject.value !== visible) {
      this.dashboardVisibleSubject.next(visible);
    }
  }

  setDatasetIds(ids: string[]): void {
    this.datasetIdsSubject.next(ids);
  }

  setDetectorIds(ids: string[]): void {
    this.detectorIdsSubject.next(ids);
  }

  requestSelect(kind: SelectionKind, id: string): void {
    this.selectRequestSubject.next({ kind, id });
  }
}
