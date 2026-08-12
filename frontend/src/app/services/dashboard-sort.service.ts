import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

import type { SortState } from '../utils/sort-rows';

/** Which of the Dashboard's two tables a sort state belongs to. */
export type DashboardTable = 'dataset' | 'detector';

const INITIAL_SORT: SortState = { column: 'name', asc: true };

/**
 * Read-side mirror of the Dashboard tables' sort state.
 *
 * `DashboardColumnsService` owns the actual `ManagedColumns` instances and
 * publishes into this service; consumers that only want to *read* the sort
 * (the top-bar context pulldowns, which order their rows to match the
 * table the user last sorted) inject this instead.
 *
 * The indirection is a bundle boundary, not decoration: the pulldowns are
 * eager (they live in the app header) while `ManagedColumns` and the
 * Dashboard's column metadata are only needed once the lazy
 * `dashboard-component` chunk loads.  Injecting the owning service from an
 * eager component pulled ~6 kB of column-management code onto the initial
 * bundle for a feature first paint never uses.
 *
 * Until the Dashboard is first constructed nothing has published, so the
 * mirror reports the same name-ascending default the tables start at.
 */
@Injectable({ providedIn: 'root' })
export class DashboardSortService {
  private readonly subjects: Record<DashboardTable, BehaviorSubject<SortState>> = {
    dataset: new BehaviorSubject<SortState>(INITIAL_SORT),
    detector: new BehaviorSubject<SortState>(INITIAL_SORT),
  };

  /** Current sort of the named table, replayed on subscribe. */
  sort$(table: DashboardTable): Observable<SortState> {
    return this.subjects[table].asObservable();
  }

  /** Publish a new sort state for the named table. Called by
   *  `DashboardColumnsService`; nothing else should write here. */
  publish(table: DashboardTable, state: SortState): void {
    this.subjects[table].next(state);
  }
}
