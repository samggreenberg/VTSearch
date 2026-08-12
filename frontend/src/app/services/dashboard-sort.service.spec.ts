import { beforeEach, describe, expect, it } from 'vitest';
import { TestBed } from '@angular/core/testing';

import { DashboardSortService } from './dashboard-sort.service';
import { DashboardColumnsService } from './dashboard-columns.service';
import { SortState } from '../utils/sort-rows';

describe('DashboardSortService', () => {
  let mirror: DashboardSortService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    mirror = TestBed.inject(DashboardSortService);
  });

  it('reports name-ascending for both tables before the Dashboard is constructed', () => {
    const seen: SortState[] = [];
    mirror.sort$('dataset').subscribe((s) => seen.push(s));
    mirror.sort$('detector').subscribe((s) => seen.push(s));

    expect(seen.length).toBe(2);
    for (const s of seen) {
      expect(s.column).toBe('name');
      expect(s.asc).toBe(true);
    }
  });

  it('keeps the two tables independent', () => {
    const columns = TestBed.inject(DashboardColumnsService);
    const datasetSeen: SortState[] = [];
    const detectorSeen: SortState[] = [];
    mirror.sort$('dataset').subscribe((s) => datasetSeen.push(s));
    mirror.sort$('detector').subscribe((s) => detectorSeen.push(s));

    columns.datasetCols.sortBy('num_items');

    expect(datasetSeen[datasetSeen.length - 1].column).toBe('num_items');
    expect(detectorSeen.length).toBe(1);
    expect(detectorSeen[0].column).toBe('name');
  });

  it('mirrors sortBy on the Dashboard tables, including the direction toggle', () => {
    const columns = TestBed.inject(DashboardColumnsService);
    const seen: SortState[] = [];
    mirror.sort$('detector').subscribe((s) => seen.push(s));

    // Replayed initial state from the freshly-constructed columns service.
    expect(seen.length).toBe(1);
    expect(seen[0]).toEqual({ column: 'name', asc: true });

    columns.detectorCols.sortBy('num_training');
    expect(seen[seen.length - 1]).toEqual({ column: 'num_training', asc: true });

    // Same-column click flips direction, and that reaches the mirror too.
    columns.detectorCols.sortBy('num_training');
    expect(seen[seen.length - 1]).toEqual({ column: 'num_training', asc: false });
  });
});
