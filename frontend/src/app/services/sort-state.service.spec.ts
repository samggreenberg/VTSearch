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
    expect(service.sortBusy).toBeFalse();
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

  it('setSortBusy and setSortStatus should update', () => {
    service.setSortBusy(true);
    service.setSortStatus('Sorting...');
    expect(service.sortBusy).toBeTrue();
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
    expect(service.sortBusy).toBeFalse();
    expect(service.sortStatus).toBe('');
    expect(service.inclusion).toBe(0);
    expect(service.loadSortLabel).toBe('');
  });

  it('sortMode$ should emit on change', (done) => {
    const modes: string[] = [];
    service.sortMode$.subscribe((m) => modes.push(m));

    service.setSortMode('learned');
    service.setSortMode('load');

    setTimeout(() => {
      expect(modes).toContain('learned');
      expect(modes).toContain('load');
      done();
    });
  });
});
