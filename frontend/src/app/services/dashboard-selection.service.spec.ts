import { TestBed } from '@angular/core/testing';

import { ActiveContextService } from './active-context.service';
import { DashboardSelectionService } from './dashboard-selection.service';
import { DatasetStateService } from './dataset-state.service';
import { DatasetRegistryEntry } from '../models/api.models';
import { DetectorRegistryEntry } from '../generated/api-client/models/detector-registry-entry';
import { provideZoneless } from '../testing/zoneless-testbed';
import { provideHttpTesting } from '../testing/test-providers';

/**
 * The Dashboard's table selection lives here rather than in the component, so
 * the top bar reads it directly instead of being pushed to after every
 * mutation. These specs pin the ladder both grids share, the registry
 * filtering the pulldown depends on, and the single intent mirror that
 * replaced the old `pushTopBarLabels()` call sites.
 */
describe('DashboardSelectionService', () => {
  let service: DashboardSelectionService;
  let datasetState: DatasetStateService;
  let activeContext: ActiveContextService;

  /** Push registry contents straight into the signal-backed state (no HTTP),
   *  the same way the pulldown spec seeds it. */
  function setRegistry(datasetIds: string[], detectorIds: string[] = []): void {
    const ds = datasetState as unknown as {
      _datasets: { set: (v: unknown) => void };
      _detectors: { set: (v: unknown) => void };
    };
    ds._datasets.set(datasetIds.map((id) => ({ id, name: id }) as DatasetRegistryEntry));
    ds._detectors.set(detectorIds.map((id) => ({ id, name: id }) as DetectorRegistryEntry));
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideZoneless(), ...provideHttpTesting()],
    });
    service = TestBed.inject(DashboardSelectionService);
    datasetState = TestBed.inject(DatasetStateService);
    activeContext = TestBed.inject(ActiveContextService);
    setRegistry(['d1', 'd2', 'd3'], ['m1', 'm2']);
  });

  describe('the shared toggle ladder', () => {
    it('single-selects, and toggles off when re-picking the sole selection', () => {
      service.toggle('dataset', 'd1', false);
      expect([...service.ids('dataset')]).toEqual(['d1']);
      service.toggle('dataset', 'd2', false);
      expect([...service.ids('dataset')]).toEqual(['d2']);
      service.toggle('dataset', 'd2', false);
      expect([...service.ids('dataset')]).toEqual([]);
    });

    it('replaces a multi-selection with a plain pick rather than toggling off', () => {
      service.selectOnly('dataset', ['d1', 'd2']);
      service.toggle('dataset', 'd1', false);
      expect([...service.ids('dataset')]).toEqual(['d1']);
    });

    it('adds and removes with the additive (Ctrl/Cmd, or checkbox) modifier', () => {
      service.toggle('dataset', 'd1', true);
      service.toggle('dataset', 'd2', true);
      expect([...service.ids('dataset')].sort()).toEqual(['d1', 'd2']);
      service.toggle('dataset', 'd1', true);
      expect([...service.ids('dataset')]).toEqual(['d2']);
    });

    it('runs the same ladder for detectors, independently of datasets', () => {
      service.toggle('dataset', 'd1', false);
      service.toggle('detector', 'm1', false);
      service.toggle('detector', 'm1', false);
      expect([...service.ids('dataset')]).toEqual(['d1']);
      expect([...service.ids('detector')]).toEqual([]);
    });
  });

  describe('select-all and the header tri-state', () => {
    it('reports none / some / all against the rows the grid shows', () => {
      expect(service.selectionState('dataset', 3)).toBe('none');
      service.toggle('dataset', 'd1', true);
      expect(service.selectionState('dataset', 3)).toBe('some');
      service.selectOnly('dataset', ['d1', 'd2', 'd3']);
      expect(service.selectionState('dataset', 3)).toBe('all');
    });

    it('selects exactly the visible rows, then clears when they are all picked', () => {
      // The detector grid passes only its visible tab's rows.
      service.toggleAll('detector', ['m1']);
      expect([...service.ids('detector')]).toEqual(['m1']);
      service.toggleAll('detector', ['m1']);
      expect([...service.ids('detector')]).toEqual([]);
    });
  });

  it('prunes ids that have left the registry, and keeps the rest', () => {
    service.selectOnly('dataset', ['d1', 'd2']);
    service.retain('dataset', new Set(['d2', 'd3']));
    expect([...service.ids('dataset')]).toEqual(['d2']);
  });

  it('drops one id on deselect and leaves an absent id alone', () => {
    service.selectOnly('dataset', ['d1', 'd2']);
    service.deselect('dataset', 'd1');
    service.deselect('dataset', 'd9');
    expect([...service.ids('dataset')]).toEqual(['d2']);
  });

  it('clears the detector selection when the grid tab changes, but not on a no-op switch', () => {
    service.selectOnly('detector', ['m1']);
    service.setDetectorTab('drafts');
    expect([...service.ids('detector')]).toEqual(['m1']);
    service.setDetectorTab('autorun');
    expect(service.detectorTab()).toBe('autorun');
    expect([...service.ids('detector')]).toEqual([]);
  });

  describe('the ids the top bar reads', () => {
    it('filters to rows that still exist, in registry order', () => {
      service.selectOnly('dataset', ['d3', 'd1', 'gone']);
      expect(service.datasetIds()).toEqual(['d1', 'd3']);
    });

    it('follows the registry when a selected row disappears, before any prune', () => {
      service.selectOnly('detector', ['m1', 'm2']);
      expect(service.detectorIds()).toEqual(['m1', 'm2']);
      setRegistry(['d1'], ['m2']);
      expect(service.detectorIds()).toEqual(['m2']);
    });
  });

  describe('mirroring a lone pick into the active-context intent', () => {
    it('mirrors an unambiguous single selection while the Dashboard is up', () => {
      service.setDashboardVisible(true);
      service.toggle('dataset', 'd1', false);
      service.toggle('detector', 'm2', false);
      TestBed.tick();
      expect(activeContext.intentDatasetId).toBe('d1');
      expect(activeContext.intentModelId).toBe('m2');
    });

    it('leaves a loaded pair alone on an empty or multiple selection', () => {
      activeContext.setActivePair('d3', 'm1');
      service.setDashboardVisible(true);
      service.selectOnly('dataset', ['d1', 'd2']);
      TestBed.tick();
      expect(activeContext.intentDatasetId).toBe('d3');
      expect(activeContext.intentModelId).toBe('m1');
    });

    it('does not mirror while the Dashboard is off screen', () => {
      activeContext.setActivePair('d3', 'm1');
      service.toggle('dataset', 'd1', false);
      TestBed.tick();
      expect(activeContext.intentDatasetId).toBe('d3');
    });
  });

  it('keeps the selection when the Dashboard unmounts, so a round trip restores it', () => {
    service.setDashboardVisible(true);
    service.selectOnly('dataset', ['d2']);
    service.setDetectorTab('autorun');
    service.setDashboardVisible(false);
    expect([...service.ids('dataset')]).toEqual(['d2']);
    service.setDashboardVisible(true);
    expect([...service.ids('dataset')]).toEqual(['d2']);
    expect(service.detectorTab()).toBe('autorun');
  });
});
