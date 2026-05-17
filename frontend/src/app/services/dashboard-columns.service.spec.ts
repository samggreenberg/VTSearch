import { TestBed } from '@angular/core/testing';
import { DashboardColumnsService } from './dashboard-columns.service';
import { SortState } from '../utils/managed-columns';

describe('DashboardColumnsService', () => {
  let service: DashboardColumnsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(DashboardColumnsService);
  });

  it('exposes two ManagedColumns instances', () => {
    expect(service.datasetCols).toBeTruthy();
    expect(service.detectorCols).toBeTruthy();
  });

  it('starts datasets and detectors sorted by name ascending', () => {
    expect(service.datasetCols.sortColumn).toBe('name');
    expect(service.datasetCols.sortAsc).toBeTrue();
    expect(service.detectorCols.sortColumn).toBe('name');
    expect(service.detectorCols.sortAsc).toBeTrue();
  });

  it('sortState$ emits the current state on subscribe and again on sortBy', () => {
    const seen: SortState[] = [];
    const sub = service.datasetCols.sortState$.subscribe((s) => seen.push(s));

    // Replay (BehaviorSubject) gives us the initial state.
    expect(seen.length).toBe(1);
    expect(seen[0].column).toBe('name');
    expect(seen[0].asc).toBeTrue();

    service.datasetCols.sortBy('num_items');
    expect(seen.length).toBe(2);
    expect(seen[1].column).toBe('num_items');
    expect(seen[1].asc).toBeTrue();

    // Same-column toggle flips direction.
    service.datasetCols.sortBy('num_items');
    expect(seen.length).toBe(3);
    expect(seen[2].column).toBe('num_items');
    expect(seen[2].asc).toBeFalse();

    sub.unsubscribe();
  });
});
