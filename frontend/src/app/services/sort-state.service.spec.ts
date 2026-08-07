import { computed } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { SortStateService, SortedItem } from './sort-state.service';

describe('SortStateService', () => {
  let service: SortStateService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SortStateService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should have default values', () => {
    expect(service.sortMode).toBe('text');
    expect(service.selectMode).toBe('top');
    expect(service.sortOrder).toBeNull();
    expect(service.threshold).toBeNull();
    expect(service.sortBusy).toBe(false);
    expect(service.sortStatus).toBe('');
    expect(service.inclusion).toBe(0);
    expect(service.loadSortLabel).toBe('');
  });

  it('setSortMode should update sortMode', () => {
    service.setSortMode('learned');
    expect(service.sortMode).toBe('learned');
  });

  it('setSelectMode should update selectMode', () => {
    service.setSelectMode('hard');
    expect(service.selectMode).toBe('hard');
  });

  it('setSortResults should update sortOrder and threshold', () => {
    const items: SortedItem[] = [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
    ];
    service.setSortResults(items, 0.7);
    expect(service.sortOrder).toEqual(items);
    expect(service.threshold).toBe(0.7);
  });

  it('setSortResults treats the list as complete (no windowing)', () => {
    const items: SortedItem[] = [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.5 },
      { id: 3, score: 0.1 },
    ];
    service.setSortResults(items, 0.4);
    expect(service.sortTotal).toBe(3);
    expect(service.sortHasMore).toBe(false);
    expect(service.sortToken).toBeNull();
    expect(service.aboveThreshold).toBe(2); // 0.9, 0.5 >= 0.4
  });

  it('setSortWindow installs a windowed first page', () => {
    service.setSortWindow({
      items: [
        { id: 1, score: 0.9 },
        { id: 2, score: 0.8 },
      ],
      threshold: 0.5,
      total: 1000,
      hasMore: true,
      token: 'tok-1',
      aboveThreshold: 640,
    });
    expect(service.sortOrder?.map((i) => i.id)).toEqual([1, 2]);
    expect(service.threshold).toBe(0.5);
    expect(service.sortTotal).toBe(1000);
    expect(service.sortHasMore).toBe(true);
    expect(service.sortToken).toBe('tok-1');
    expect(service.aboveThreshold).toBe(640);
  });

  it('acqThreshold carries the acquisition cut when the sort supplies one', () => {
    service.setSortWindow({
      items: [{ id: 1, score: 0.9 }],
      threshold: 0.5,
      acqThreshold: 0.72,
      total: 1,
      hasMore: false,
      token: null,
      aboveThreshold: 1,
    });
    // The two jobs are separate: Autopilot's Hard/New picks sample around the
    // acquisition cut while everything the user sees stays on `threshold`.
    expect(service.acqThreshold).toBe(0.72);
    expect(service.threshold).toBe(0.5);
    expect(service.aboveThreshold).toBe(1); // counted against the reporting cut
  });

  it('acqThreshold falls back to the reporting threshold when absent', () => {
    // Text / example sorts have no detector behind them, and the load-sort
    // restore path carries no window metadata, so the picks keep the behaviour
    // they had before the two cuts were split.
    service.setSortWindow({
      items: [{ id: 1, score: 0.9 }],
      threshold: 0.5,
      total: 1,
      hasMore: false,
      token: null,
      aboveThreshold: 1,
    });
    expect(service.acqThreshold).toBe(0.5);

    service.setSortResults([{ id: 1, score: 0.9 }], 0.4);
    expect(service.acqThreshold).toBe(0.4);
  });

  it('setSortResults clears a stale acquisition cut', () => {
    service.setSortWindow({
      items: [{ id: 1, score: 0.9 }],
      threshold: 0.5,
      acqThreshold: 0.72,
      total: 1,
      hasMore: false,
      token: null,
      aboveThreshold: 1,
    });
    // A later sort with no acquisition cut must not keep sampling around the
    // previous detector's one, over a ranking it no longer describes.
    service.setSortResults([{ id: 2, score: 0.3 }], 0.2);
    expect(service.acqThreshold).toBe(0.2);
  });

  it('appendSortItems grows the loaded window and updates hasMore', () => {
    service.setSortWindow({
      items: [{ id: 1, score: 0.9 }],
      threshold: 0.5,
      total: 5,
      hasMore: true,
      token: 'tok-1',
      aboveThreshold: 3,
    });
    service.appendSortItems([{ id: 2, score: 0.8 }, { id: 3, score: 0.7 }], true);
    expect(service.sortOrder?.map((i) => i.id)).toEqual([1, 2, 3]);
    expect(service.sortHasMore).toBe(true);
    service.appendSortItems([{ id: 4, score: 0.4 }], false);
    expect(service.sortOrder?.map((i) => i.id)).toEqual([1, 2, 3, 4]);
    expect(service.sortHasMore).toBe(false);
  });

  it('setSortBusy and setSortStatus should update', () => {
    service.setSortBusy(true);
    service.setSortStatus('Sorting...');
    expect(service.sortBusy).toBe(true);
    expect(service.sortStatus).toBe('Sorting...');
  });

  it('setInclusion should update', () => {
    service.setInclusion(0.5);
    expect(service.inclusion).toBe(0.5);
  });

  it('setLoadSortLabel should update', () => {
    service.setLoadSortLabel('detector_1');
    expect(service.loadSortLabel).toBe('detector_1');
  });

  it('clear should reset all state', () => {
    service.setSortMode('learned');
    service.setSelectMode('new');
    service.setSortResults([{ id: 1, score: 0.5 }], 0.5);
    service.setSortBusy(true);
    service.setSortStatus('busy');
    service.setInclusion(0.3);
    service.setLoadSortLabel('test');

    service.clear();

    expect(service.sortMode).toBe('text');
    expect(service.selectMode).toBe('top');
    expect(service.sortOrder).toBeNull();
    expect(service.threshold).toBeNull();
    expect(service.sortBusy).toBe(false);
    expect(service.sortStatus).toBe('');
    expect(service.inclusion).toBe(0);
    expect(service.loadSortLabel).toBe('');
    expect(service.sortTotal).toBe(0);
    expect(service.sortHasMore).toBe(false);
    expect(service.sortToken).toBeNull();
    expect(service.aboveThreshold).toBe(0);
  });

  it('sortMode getter is reactive (drives a computed that reads it)', () => {
    // The state is signal-backed and exposed via value getters; a computed that
    // reads the getter must recompute when the setter writes the backing signal.
    // This is the property the zoneless template bindings rely on.
    const derived = TestBed.runInInjectionContext(() => computed(() => service.sortMode));
    expect(derived()).toBe('text');

    service.setSortMode('learned');
    expect(derived()).toBe('learned');

    service.setSortMode('load');
    expect(derived()).toBe('load');
  });
});
