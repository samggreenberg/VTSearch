import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { DashboardComponent } from './dashboard.component';
import { LoadingTask } from '../../models/api.models';
import { LabelSessionService } from '../../services/label-session.service';
import { NewThingFlowsService } from '../../services/new-thing-flows.service';
import { ActiveContextService } from '../../services/active-context.service';
import { DashboardSelectionService } from '../../services/dashboard-selection.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../testing/test-providers';
import { settleResource } from '../../testing/settle-resource';

describe('DashboardComponent', () => {
  let component: DashboardComponent;
  let fixture: ComponentFixture<DashboardComponent>;
  let httpMock: HttpTestingController;
  let selection: DashboardSelectionService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [...provideZoneless(), ...provideHttpTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    selection = TestBed.inject(DashboardSelectionService);
  });

  afterEach(() => {
    // Best-effort background pollers (disk/RAM usage on a timer, embedder
    // preloads triggered by auto-selection) may have fired during the test
    // without being asserted. Drain them so verify() only fails on
    // genuinely-unexpected requests.
    drainBackgroundRequests();
    httpMock.verify();
  });

  /** Flush the fire-and-forget requests the dashboard issues on init and
   *  as a side effect of selection: the disk/RAM usage pollers
   *  (`timer(0, 10000)`) and the per-dataset embedder preload. None of
   *  these are asserted on by the selection/hint/button tests, so we just
   *  swallow whatever has accumulated. */
  function drainBackgroundRequests(): void {
    for (const req of httpMock.match('/api/dashboard/disk-usage')) {
      if (!req.cancelled) req.flush({ total: 0, used: 0, free: 0 });
    }
    for (const req of httpMock.match('/api/dashboard/ram-usage')) {
      if (!req.cancelled) req.flush({ total: 0, used: 0, free: 0 });
    }
    for (const req of httpMock.match((r) => /\/preload-embedder$/.test(r.url))) {
      if (!req.cancelled) req.flush({ ok: true, embedder: '' });
    }
  }

  function flushInitialRequests(
    datasets: any[] = [],
    detectors: any[] = [],
    importers: any[] = [],
  ): void {
    TestBed.tick();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors });
    httpMock.expectOne('/api/dataset/all-importers').flush({ importers, tabs: [] });
    // DatasetStateService is signal-backed and exposes `datasets$`/`detectors$`
    // as `toObservable` bridges, which emit on the next change-detection pass
    // (not synchronously when the signal is set). Tick once more so the
    // dashboard's `datasets$`/`detectors$` subscriptions (registry auto-select)
    // run before the test asserts on the resulting selection state.
    TestBed.tick();
    // Auto-selecting a single dataset kicks off an embedder preload; drain it
    // (and any usage polls already in flight) here so tests that don't assert
    // on them stay clean.
    drainBackgroundRequests();
  }

  it('should create', () => {
    flushInitialRequests();
    expect(component).toBeTruthy();
  });

  it('should fetch datasets and detectors on init', () => {
    const datasets = [{ id: 'd1', name: 'Test Dataset', media_type: 'audio', num_items: 10 }];
    const detectors = [{ id: 'm1', name: 'Test Detector' }];
    flushInitialRequests(datasets, detectors);
    expect(component.datasets.length).toBe(1);
    expect(component.detectors.length).toBe(1);
  });

  it('should auto-select single dataset', () => {
    const datasets = [{ id: 'd1', name: 'Only One' }];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.has('d1')).toBe(true);
  });

  it('should auto-select single model', () => {
    const models = [{ id: 'm1', name: 'Only One' }];
    flushInitialRequests([], models);
    expect(component.selectedDetectorIds.has('m1')).toBe(true);
  });

  it('should not auto-select when multiple datasets on initial load', () => {
    const datasets = [
      { id: 'd1', name: 'First' },
      { id: 'd2', name: 'Second' },
    ];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.size).toBe(0);
  });

  it('should auto-select newly added dataset', () => {
    const datasets = [{ id: 'd1', name: 'First' }];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.has('d1')).toBe(true);

    // Simulate adding a second dataset via refresh
    component.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({
      datasets: [
        { id: 'd1', name: 'First' },
        { id: 'd2', name: 'Second' },
      ],
    });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
    // Let the `datasets$` bridge deliver the new registry to the auto-select sub.
    TestBed.tick();

    expect(component.selectedDatasetIds.has('d1')).toBe(false);
    expect(component.selectedDatasetIds.has('d2')).toBe(true);
  });

  it('should auto-select newly added model', () => {
    const models = [{ id: 'm1', name: 'First' }];
    flushInitialRequests([], models);
    expect(component.selectedDetectorIds.has('m1')).toBe(true);

    // Simulate adding a second model via refresh
    component.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({
      detectors: [
        { id: 'm1', name: 'First' },
        { id: 'm2', name: 'Second' },
      ],
    });
    // Let the `detectors$` bridge deliver the new registry to the auto-select sub.
    TestBed.tick();

    // Adding items after the initial load clears the prior selection and
    // selects only the new ones (same behavior as datasets above).
    expect(component.selectedDetectorIds.has('m1')).toBe(false);
    expect(component.selectedDetectorIds.has('m2')).toBe(true);
  });

  describe('detector Drafts/AutoFind tabs', () => {
    const mixed = [
      { id: 'm1', name: 'Draft A', media_type: 'audio' },
      { id: 'm2', name: 'Frozen B', media_type: 'audio', autofind: true },
      { id: 'm3', name: 'Draft C', media_type: 'audio' },
    ];

    it('partitions detectors by autofind and defaults to the Drafts tab', () => {
      flushInitialRequests([], mixed);
      expect(component.detectorTab()).toBe('drafts');
      expect(component.draftDetectors.map((d) => d.id)).toEqual(['m1', 'm3']);
      expect(component.autofindDetectors.map((d) => d.id)).toEqual(['m2']);
      expect(component.sortedDetectors.map((d) => d.id).sort()).toEqual(['m1', 'm3']);

      component.setDetectorTab('autofind');
      expect(component.sortedDetectors.map((d) => d.id)).toEqual(['m2']);
    });

    it('clears the detector selection when switching tabs', () => {
      flushInitialRequests([], mixed);
      component.toggleDetectorCheckbox('m1');
      expect(component.selectedDetectorIds.size).toBe(1);
      component.setDetectorTab('autofind');
      expect(component.selectedDetectorIds.size).toBe(0);
    });

    it('scopes select-all to the visible tab', () => {
      flushInitialRequests([], mixed);
      component.toggleAllDetectors();
      expect([...component.selectedDetectorIds].sort()).toEqual(['m1', 'm3']);
      expect(component.detectorSelectionState).toBe('all');

      component.setDetectorTab('autofind');
      component.toggleAllDetectors();
      expect([...component.selectedDetectorIds]).toEqual(['m2']);
    });

    it('moves a detector between tabs via the autofind endpoint and refreshes', () => {
      flushInitialRequests([], mixed);
      component.setDetectorAutofind(component.detectors[0], true);
      const put = httpMock.expectOne('/api/detectors/registry/m1/autofind');
      expect(put.request.method).toBe('PUT');
      expect(put.request.body).toEqual({ autofind: true });
      put.flush({ id: 'm1', autofind: true });
      // The move triggers a registry refresh so the row hops tabs.
      httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
      httpMock.expectOne('/api/detectors/registry').flush({
        detectors: [
          { id: 'm1', name: 'Draft A', media_type: 'audio', autofind: true },
          { id: 'm2', name: 'Frozen B', media_type: 'audio', autofind: true },
          { id: 'm3', name: 'Draft C', media_type: 'audio' },
        ],
      });
      TestBed.tick();
      expect(component.autofindDetectors.map((d) => d.id).sort()).toEqual(['m1', 'm2']);
      expect(component.selectedDetectorIds.has('m1')).toBe(false);
    });

    it('switches back to the Drafts tab when a new detector is registered', () => {
      flushInitialRequests([], mixed);
      component.setDetectorTab('autofind');

      component.refresh();
      httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
      httpMock.expectOne('/api/detectors/registry').flush({
        detectors: [...mixed, { id: 'm4', name: 'Fresh Draft', media_type: 'audio' }],
      });
      TestBed.tick();

      expect(component.detectorTab()).toBe('drafts');
      expect(component.selectedDetectorIds.has('m4')).toBe(true);
    });

    it('disables Train for a frozen detector with an explanatory hint', () => {
      flushInitialRequests(
        [{ id: 'd1', name: 'Data', media_type: 'audio' }],
        [{ id: 'm2', name: 'Frozen B', media_type: 'audio', num_training: 5, autofind: true }],
      );
      component.setDetectorTab('autofind');
      component.toggleDetectorSelection('m2', new MouseEvent('click'));
      expect(component.selectedDetectorIds.has('m2')).toBe(true);
      expect(component.labelEnabled).toBe(false);
      expect(component.labelHint).toBe('Frozen: move to Drafts to retrain');
      // Test (read-only scoring) stays available.
      expect(component.findEnabled).toBe(true);
    });

    describe('Test follows the visible tab only (#4228)', () => {
      const dataset = [{ id: 'd1', name: 'Data', media_type: 'audio' }];
      const loneAutofind = [
        { id: 'm2', name: 'Frozen B', media_type: 'audio', num_training: 5, autofind: true },
      ];

      /** Answer a registry refresh, triggering one first unless the code
       *  under test already has. The rows are copied: re-flushing the same
       *  array would leave the registry signal unchanged, so `detectors$`
       *  would never emit and the refresh would test nothing. */
      function refreshRegistry(detectors: unknown[], trigger = true): void {
        if (trigger) component.refresh();
        httpMock.expectOne('/api/datasets/registry').flush({ datasets: [...dataset] });
        httpMock.expectOne('/api/detectors/registry').flush({ detectors: [...detectors] });
        TestBed.tick();
        drainBackgroundRequests();
      }

      it('does not auto-select a lone AutoFind detector while Drafts is showing', () => {
        flushInitialRequests(dataset, loneAutofind);
        expect(component.detectorTab()).toBe('drafts');
        expect(component.selectedDetectorIds.size).toBe(0);
        expect(component.findEnabled).toBe(false);
        expect(component.findHint).toBe('Select a detector in the table above');
      });

      it('leaves Drafts empty after selecting on AutoFind and switching back, across refreshes', () => {
        flushInitialRequests(dataset, loneAutofind);
        component.setDetectorTab('autofind');
        // With the grid showing it, the lone detector is auto-selected again.
        refreshRegistry(loneAutofind);
        expect(component.selectedDetectorIds.has('m2')).toBe(true);
        expect(component.findEnabled).toBe(true);

        component.setDetectorTab('drafts');
        expect(component.selectedDetectorIds.size).toBe(0);
        expect(component.findEnabled).toBe(false);
        // A later registry refresh used to reselect the hidden detector.
        refreshRegistry(loneAutofind);
        expect(component.selectedDetectorIds.size).toBe(0);
        expect(component.findEnabled).toBe(false);
      });

      it('leaves Drafts empty when its only detector moves to AutoFind', () => {
        flushInitialRequests(dataset, [
          { id: 'm1', name: 'Draft A', media_type: 'audio', num_training: 5 },
        ]);
        expect(component.selectedDetectorIds.has('m1')).toBe(true);

        component.setDetectorAutofind(component.detectors[0], true);
        httpMock.expectOne('/api/detectors/registry/m1/autofind').flush({ id: 'm1', autofind: true });
        httpMock.expectOne('/api/datasets/registry').flush({ datasets: dataset });
        httpMock.expectOne('/api/detectors/registry').flush({
          detectors: [{ id: 'm1', name: 'Draft A', media_type: 'audio', num_training: 5, autofind: true }],
        });
        TestBed.tick();

        expect(component.detectorTab()).toBe('drafts');
        expect(component.selectedDetectorIds.size).toBe(0);
        expect(component.findEnabled).toBe(false);
      });

      it('lands a detector created from Train on the AutoFind tab on Drafts, selected', () => {
        flushInitialRequests(dataset, loneAutofind);
        component.setDetectorTab('autofind');
        const routerSpy = vi.spyOn(component['router'], 'navigate').mockResolvedValue(true);

        // Train with no detector selected opens the new-detector modal and
        // proceeds to training once the detector exists.
        component.onLabel();
        expect(component.trainAfterModelCreation).toBe(true);
        // The created event refreshes the registry itself.
        TestBed.inject(NewThingFlowsService).emitDetectorCreated('m9');
        refreshRegistry([...loneAutofind, { id: 'm9', name: 'Fresh', media_type: 'audio' }], false);

        expect(component.detectorTab()).toBe('drafts');
        expect(component.selectedDetectorIds.has('m9')).toBe(true);
        expect(routerSpy).toHaveBeenCalledWith(['/label', 'd1', 'm9']);
      });
    });

    it('hides the Combine and Delete-selected section actions on the AutoFind tab', async () => {
      flushInitialRequests([], mixed);
      await fixture.whenStable();
      fixture.detectChanges();
      const el = fixture.nativeElement as HTMLElement;
      const detectorSection = el.querySelectorAll('.dashboard-section')[1] as HTMLElement;
      expect(detectorSection.querySelector('[aria-label="Combine selected detectors"]')).toBeTruthy();
      expect(detectorSection.querySelector('[aria-label="Delete selected"]')).toBeTruthy();

      component.setDetectorTab('autofind');
      fixture.detectChanges();
      expect(detectorSection.querySelector('[aria-label="Combine selected detectors"]')).toBeNull();
      expect(detectorSection.querySelector('[aria-label="Delete selected"]')).toBeNull();
      drainBackgroundRequests();
    });
  });

  it('mirrors an implicitly selected dataset into the active-context intent', () => {
    // Off the Dashboard the top-bar pulldowns read the active-context intent
    // (not the mirrored table selection), so an auto-selected import must land
    // there too or the picker forgets it the moment the Dashboard unmounts.
    const activeContext = TestBed.inject(ActiveContextService);
    const datasets = [{ id: 'd1', name: 'Only One' }];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.has('d1')).toBe(true);
    expect(activeContext.intentDatasetId).toBe('d1');
  });

  it('mirrors an implicitly selected model into the active-context intent', () => {
    const activeContext = TestBed.inject(ActiveContextService);
    const models = [{ id: 'm1', name: 'Only One' }];
    flushInitialRequests([], models);
    expect(component.selectedDetectorIds.has('m1')).toBe(true);
    expect(activeContext.intentModelId).toBe('m1');
  });

  it('does not blank out the intent when the selection is empty or multiple', () => {
    // A 0- or multi-selection is ambiguous, so it must leave a previously
    // loaded pair's intent alone rather than snapping the picker to a
    // placeholder.
    const activeContext = TestBed.inject(ActiveContextService);
    activeContext.setActivePair('d0', 'm0');
    const datasets = [
      { id: 'd1', name: 'First' },
      { id: 'd2', name: 'Second' },
    ];
    flushInitialRequests(datasets);
    // Multiple datasets on initial load → nothing auto-selected, so the
    // empty-selection mirror leaves the loaded pair's intent untouched.
    expect(component.selectedDatasetIds.size).toBe(0);
    expect(activeContext.intentDatasetId).toBe('d0');
    // Select both at once → an ambiguous multi-selection also leaves it alone.
    component.toggleAllDatasets();
    expect(component.selectedDatasetIds.size).toBe(2);
    expect(activeContext.intentDatasetId).toBe('d0');
    expect(activeContext.intentModelId).toBe('m0');
  });

  it('keeps the selection across a Dashboard round trip', () => {
    // The selection lives in `DashboardSelectionService`, not in the
    // component, so leaving for the label view and coming back leaves the
    // same rows highlighted (and the same name in the top bar) instead of
    // resetting to whatever the registry auto-select would pick.
    const datasets = [
      { id: 'd1', name: 'First' },
      { id: 'd2', name: 'Second' },
    ];
    const detectors = [
      { id: 'm1', name: 'Draft', media_type: 'audio' },
      { id: 'm2', name: 'Frozen', media_type: 'audio', autofind: true },
    ];
    flushInitialRequests(datasets, detectors);
    selection.selectOnly('dataset', ['d2']);
    component.setDetectorTab('autofind');
    selection.selectOnly('detector', ['m2']);

    fixture.destroy();
    expect(selection.dashboardVisible()).toBe(false);
    // The pulldown keeps reading the service off the Dashboard, so the ids
    // survive the unmount rather than being reset by it.
    expect(selection.datasetIds()).toEqual(['d2']);

    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
    flushInitialRequests(datasets, detectors);

    expect(component.selectedDatasetIds.has('d2')).toBe(true);
    // The tab travels with the selection it scopes: a returning user must not
    // land on Drafts with a hidden AutoFind row still feeding the actions.
    expect(component.detectorTab()).toBe('autofind');
    expect(component.selectedDetectorIds.has('m2')).toBe(true);
  });

  it('should toggle dataset selection on click', () => {
    flushInitialRequests();
    const event = new MouseEvent('click');
    component.toggleDatasetSelection('d1', event);
    expect(component.isDatasetSelected('d1')).toBe(true);
    component.toggleDatasetSelection('d1', event);
    expect(component.isDatasetSelected('d1')).toBe(false);
  });

  it('should support multi-select with ctrl key', () => {
    flushInitialRequests();
    const ctrlEvent = new MouseEvent('click', { ctrlKey: true });
    component.toggleDatasetSelection('d1', new MouseEvent('click'));
    component.toggleDatasetSelection('d2', ctrlEvent);
    expect(component.isDatasetSelected('d1')).toBe(true);
    expect(component.isDatasetSelected('d2')).toBe(true);
  });

  it('should replace selection without ctrl key', () => {
    flushInitialRequests();
    component.toggleDatasetSelection('d1', new MouseEvent('click'));
    component.toggleDatasetSelection('d2', new MouseEvent('click'));
    expect(component.isDatasetSelected('d1')).toBe(false);
    expect(component.isDatasetSelected('d2')).toBe(true);
  });

  it('should sort datasets by column', () => {
    const datasets = [
      { id: 'd1', name: 'Bravo', num_items: 5 },
      { id: 'd2', name: 'Alpha', num_items: 10 },
    ];
    flushInitialRequests(datasets);

    // 'name' is the initial sort column (ascending), so the list starts
    // sorted ascending; clicking it toggles to descending, then back.
    expect(component.sortedDatasets[0].name).toBe('Alpha');

    component.datasetCols.sortBy('name');
    expect(component.sortedDatasets[0].name).toBe('Bravo');

    component.datasetCols.sortBy('name');
    expect(component.sortedDatasets[0].name).toBe('Alpha');
  });

  it('should sort models by column', () => {
    const models = [
      { id: 'm1', name: 'Zeta', num_training: 5 },
      { id: 'm2', name: 'Alpha', num_training: 10 },
    ];
    flushInitialRequests([], models);

    // 'name' is the initial sort column (ascending) \u2192 Alpha first.
    expect(component.sortedDetectors[0].name).toBe('Alpha');
    // Toggling it flips to descending \u2192 Zeta first.
    component.detectorCols.sortBy('name');
    expect(component.sortedDetectors[0].name).toBe('Zeta');
  });

  it('should show sort indicators', () => {
    flushInitialRequests();
    // 'name' starts as the active ascending sort, so it already shows \u25B2.
    expect(component.datasetCols.sortIndicator('name')).toContain('\u25B2');
    expect(component.datasetCols.isSortActive('name')).toBe(true);
    // Clicking the active column toggles to descending \u2192 \u25BC.
    component.datasetCols.sortBy('name');
    expect(component.datasetCols.sortIndicator('name')).toContain('\u25BC');
    expect(component.datasetCols.isSortActive('other')).toBe(false);
  });

  describe('button state', () => {
    it('should disable Label when nothing selected', () => {
      flushInitialRequests();
      selection.clear('dataset');
      selection.clear('detector');
      expect(component.labelEnabled).toBe(false);
    });

    it('should enable Label with 1 dataset + 1 model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      // Auto-selected since only 1 each
      expect(component.labelEnabled).toBe(true);
    });

    it('should disable Label on media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.labelEnabled).toBe(false);
    });

    it('should disable Label when model media_type is "any" (not a valid type)', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'any' }];
      flushInitialRequests(datasets, models);
      expect(component.labelEnabled).toBe(false);
    });

    it('should disable Test with no selections', () => {
      flushInitialRequests();
      selection.clear('dataset');
      selection.clear('detector');
      expect(component.findEnabled).toBe(false);
    });

    it('should enable Test with matching media types', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      // Test requires the detector to have training labels.
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 5 }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBe(true);
    });

    it('should disable Test on media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBe(false);
    });

    it('should disable Test when multiple datasets have different media types', () => {
      const datasets = [
        { id: 'd1', name: 'DS1', media_type: 'audio' },
        { id: 'd2', name: 'DS2', media_type: 'image' },
      ];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      selection.toggle('dataset', 'd2', true);
      expect(component.findEnabled).toBe(false);
    });

    it('should enable Test when all selected items share media type', () => {
      const datasets = [
        { id: 'd1', name: 'DS1', media_type: 'image' },
        { id: 'd2', name: 'DS2', media_type: 'image' },
      ];
      const models = [
        { id: 'm1', name: 'M1', media_type: 'image', num_training: 5 },
        { id: 'm2', name: 'M2', media_type: 'image', num_training: 5 },
      ];
      flushInitialRequests(datasets, models);
      selection.toggle('dataset', 'd2', true);
      selection.toggle('detector', 'm2', true);
      expect(component.findEnabled).toBe(true);
    });

    it('should disable Test when the selected model has 0 training', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 0 }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBe(false);
    });

    it('should enable Test for a model with training', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 5 }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBe(true);
    });
  });

  describe('parallel-task gating (isContextSwitching / isNavBusy)', () => {
    // Regression guard for the GRID parallelism fix. Background work
    // (dataset imports, card-initiated loads, browse-prep of another row)
    // sets `datasetState.loading`, but that must NOT freeze independent
    // dashboard actions — on a big machine you can saturate the import
    // slots and still start another import, create/delete a detector, or
    // change the selection. Only an in-flight *active-pair switch* gates
    // them, via `isContextSwitching`. Train/Test additionally wait out a
    // browse-prep (whose completion fires a competing /browse navigation),
    // via `isNavBusy`.

    it('keeps independent actions live while only a background load runs', () => {
      flushInitialRequests();
      vi.spyOn(component.datasetState, 'loading', 'get').mockReturnValue(true);
      // A background import/load is the ONLY thing in flight.
      expect(component.isContextSwitching).toBe(false);
      expect(component.isNavBusy).toBe(false);
    });

    it('gates the whole dashboard during an active-pair context switch', () => {
      flushInitialRequests();
      vi.spyOn(component['contextSwitch'], 'switching', 'get').mockReturnValue(true);
      expect(component.isContextSwitching).toBe(true);
      expect(component.isNavBusy).toBe(true);
    });

    it('gates on a Train or Test click intent', () => {
      flushInitialRequests();
      component.trainLoading.set(true);
      expect(component.isContextSwitching).toBe(true);
      component.trainLoading.set(false);
      component.findLoading.set(true);
      expect(component.isContextSwitching).toBe(true);
    });

    it('gates Train/Test during browse-prep but leaves independent actions live', () => {
      flushInitialRequests();
      vi.spyOn(component.browsePrep, 'preparing', 'get').mockReturnValue(true);
      expect(component.isContextSwitching).toBe(false);
      expect(component.isNavBusy).toBe(true);
    });
  });

  describe('label hints', () => {
    it('should hint about missing dataset', () => {
      flushInitialRequests();
      selection.clear('dataset');
      expect(component.labelHint).toBe('Select a dataset in the table above');
    });

    it('should hint about missing model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      flushInitialRequests(datasets);
      // Single dataset is auto-selected; no detector selected.
      selection.clear('detector');
      expect(component.labelHint).toBe('Create a new detector, then train it on the selected dataset');
    });

    it('should hint about multiple datasets', () => {
      const datasets = [
        { id: 'd1', name: 'DS1', media_type: 'audio' },
        { id: 'd2', name: 'DS2', media_type: 'audio' },
      ];
      flushInitialRequests(datasets);
      selection.toggle('dataset', 'd1', true);
      selection.toggle('dataset', 'd2', true);
      expect(component.labelHint).toBe('Select exactly 1 dataset');
    });

    it('should hint about multiple models', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [
        { id: 'm1', name: 'M1', media_type: 'audio' },
        { id: 'm2', name: 'M2', media_type: 'audio' },
      ];
      flushInitialRequests(datasets, models);
      selection.selectOnly('dataset', ['d1']);
      selection.selectOnly('detector', ['m1', 'm2']);
      expect(component.labelHint).toBe('Select exactly 1 detector');
    });

    it('should hint about media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.labelHint).toBe('Media type mismatch');
    });

    it('should return empty hint when label is enabled', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      expect(component.labelHint).toBe('Open the Train view with the selected dataset and detector');
    });
  });

  describe('Test button hints (findHint)', () => {
    it('should hint about missing dataset and model', () => {
      flushInitialRequests();
      selection.clear('dataset');
      selection.clear('detector');
      expect(component.findHint).toBe('Select a dataset and detector above');
    });

    it('should hint about missing dataset', () => {
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests([], models);
      // Single detector is auto-selected; no dataset selected.
      selection.clear('dataset');
      expect(component.findHint).toBe('Select a dataset in the table above');
    });

    it('should hint about missing model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      flushInitialRequests(datasets);
      // Single dataset is auto-selected; no detector selected.
      selection.clear('detector');
      expect(component.findHint).toBe('Select a detector in the table above');
    });

    it('should hint about media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.findHint).toBe('Media type mismatch');
    });

    it('should return the test hint when Test is enabled', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 5 }];
      flushInitialRequests(datasets, models);
      expect(component.findHint).toBe(
        "Open the Test view to score the selected dataset and test the selected detector's line on it",
      );
    });

    it('should hint about untrained model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 0 }];
      flushInitialRequests(datasets, models);
      expect(component.findHint).toBe('Detector has no training labels');
    });

    // Test opens one view on one pair, so it takes exactly one of each, as
    // Train does, rather than quietly testing the first ticked pair; Find
    // takes any number.
    it('should hint about multiple datasets, leaving Find enabled', () => {
      const datasets = [
        { id: 'd1', name: 'DS1', media_type: 'audio' },
        { id: 'd2', name: 'DS2', media_type: 'audio' },
      ];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 5 }];
      flushInitialRequests(datasets, models);
      selection.selectOnly('dataset', ['d1', 'd2']);
      selection.selectOnly('detector', ['m1']);
      expect(component.findEnabled).toBe(false);
      expect(component.findHint).toBe('Select exactly 1 dataset');
      expect(component.autofindEnabled).toBe(true);
    });

    it('should hint about multiple detectors, leaving Find enabled', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [
        { id: 'm1', name: 'M1', media_type: 'audio', num_training: 5 },
        { id: 'm2', name: 'M2', media_type: 'audio', num_training: 5 },
      ];
      flushInitialRequests(datasets, models);
      selection.selectOnly('dataset', ['d1']);
      selection.selectOnly('detector', ['m1', 'm2']);
      expect(component.findEnabled).toBe(false);
      expect(component.findHint).toBe('Select exactly 1 detector');
      expect(component.autofindEnabled).toBe(true);
    });
  });

  describe('Find button (#4529; AutoRun until #4525)', () => {
    const audioDatasets = [
      { id: 'd1', name: 'DS1', media_type: 'audio', loaded: true },
      { id: 'd2', name: 'DS2', media_type: 'audio', loaded: true },
    ];
    const audioDetectors = [
      { id: 'm1', name: 'M1', media_type: 'audio', num_training: 5 },
      { id: 'm2', name: 'M2', media_type: 'audio', num_training: 5 },
    ];

    function autofindButton(): HTMLButtonElement {
      fixture.detectChanges();
      const buttons = Array.from(
        (fixture.nativeElement as HTMLElement).querySelectorAll<HTMLButtonElement>('.dashboard-actions button'),
      );
      const button = buttons.find((b) => b.textContent?.trim() === 'Find');
      expect(button).toBeTruthy();
      return button!;
    }

    it('sits in the action bar after Train and Test', () => {
      flushInitialRequests();
      autofindButton();
      const labels = Array.from(
        (fixture.nativeElement as HTMLElement).querySelectorAll('.dashboard-actions button'),
      ).map((b) => b.textContent?.trim());
      expect(labels).toEqual(['Train', 'Test', 'Find']);
    });

    it('is disabled, with the hint Test gives, until a dataset and a detector are ticked', () => {
      flushInitialRequests();
      expect(component.autofindEnabled).toBe(false);
      expect(component.autofindHint).toBe('Select a dataset and detector above');
      expect(autofindButton().disabled).toBe(true);
    });

    it('is enabled for every ticked dataset and every ticked detector of one media type', () => {
      flushInitialRequests(audioDatasets, audioDetectors);
      selection.selectOnly('dataset', ['d1', 'd2']);
      selection.selectOnly('detector', ['m1', 'm2']);
      expect(component.autofindEnabled).toBe(true);
      expect(autofindButton().disabled).toBe(false);
      expect(component.autofindHint).toBe(
        'Run every selected detector on every selected dataset, as AutoFind does, and show the results as each run finishes',
      );
    });

    it('is disabled on a media type mismatch across the ticked rows', () => {
      flushInitialRequests(
        [...audioDatasets, { id: 'd3', name: 'Pics', media_type: 'image', loaded: true }],
        audioDetectors,
      );
      selection.selectOnly('dataset', ['d1', 'd3']);
      selection.selectOnly('detector', ['m1']);
      expect(component.autofindEnabled).toBe(false);
      expect(component.autofindHint).toBe('Media type mismatch');
    });

    it('is disabled while any ticked detector is untrained', () => {
      flushInitialRequests(audioDatasets, [...audioDetectors, { id: 'm3', name: 'New', media_type: 'audio', num_training: 0 }]);
      selection.selectOnly('dataset', ['d1']);
      selection.selectOnly('detector', ['m1', 'm3']);
      expect(component.autofindEnabled).toBe(false);
      expect(component.autofindHint).toBe('Detector has no training labels');
    });

    it('starts one run per ticked dataset, restricted to the ticked detectors', () => {
      flushInitialRequests(audioDatasets, audioDetectors);
      selection.selectOnly('dataset', ['d1', 'd2']);
      selection.selectOnly('detector', ['m1', 'm2']);

      component.onAutofind();

      for (const id of ['d1', 'd2']) {
        const req = httpMock.expectOne(`/api/datasets/registry/${id}/autofind`);
        expect(req.request.method).toBe('POST');
        expect(req.request.body).toEqual({ detector_ids: ['m1', 'm2'] });
        req.flush({ ok: true, message: 'AutoFind started', task_id: `_autofind_${id}` });
      }
    });

    it('loads an unloaded ticked dataset before running on it', () => {
      flushInitialRequests([{ id: 'd1', name: 'Cold', media_type: 'audio', loaded: false }], [audioDetectors[0]]);

      component.onAutofind();

      // An empty task_id means the load needed no background work.
      httpMock.expectOne('/api/datasets/registry/d1/load').flush({ ok: true, message: 'Already loaded', task_id: '' });
      const req = httpMock.expectOne('/api/datasets/registry/d1/autofind');
      expect(req.request.body).toEqual({ detector_ids: ['m1'] });
      req.flush({ ok: true, message: 'AutoFind started', task_id: '_autofind_1' });
    });

    it('does nothing while disabled', () => {
      flushInitialRequests(audioDatasets, [{ id: 'm3', name: 'New', media_type: 'audio', num_training: 0 }]);
      selection.selectOnly('dataset', ['d1']);
      component.onAutofind();
      httpMock.expectNone((r) => r.url.endsWith('/autofind'));
    });

    it("leaves the row menu's Run AutoFind on the AutoFind list (no body)", () => {
      flushInitialRequests([audioDatasets[0]]);
      component.runAutofind(audioDatasets[0]);
      const req = httpMock.expectOne('/api/datasets/registry/d1/autofind');
      expect(req.request.body).toBeNull();
      req.flush({ ok: true, message: 'AutoFind started', task_id: '_autofind_1' });
    });
  });

  it('should rename a dataset', () => {
    const datasets = [{ id: 'd1', name: 'Old', media_type: 'audio' }];
    flushInitialRequests(datasets);

    component.renameDataset(datasets[0], 'New');
    const req = httpMock.expectOne('/api/datasets/registry/d1/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'New' });
    req.flush({});

    // Refresh calls
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  });

  it('should delete a dataset after confirmation', async () => {
    const datasets = [{ id: 'd1', name: 'ToDelete', media_type: 'audio' }];
    flushInitialRequests(datasets);
    selection.selectOnly('dataset', ['d1']);

    // Mock dialog confirmation
    vi.spyOn(component['dialog'], 'confirmDestructive').mockReturnValue(Promise.resolve(true));

    component.deleteDataset(datasets[0]);
    // Drain the confirm() promise continuation that issues the DELETE.
    await new Promise<void>((resolve) => setTimeout(resolve));

    const req = httpMock.expectOne('/api/datasets/registry/d1');
    expect(req.request.method).toBe('DELETE');
    req.flush({});

    expect(component.selectedDatasetIds.has('d1')).toBe(false);

    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  });

  it('clears the active context when the bulk delete removes the active dataset', async () => {
    const datasets = [
      { id: 'd1', name: 'Active', media_type: 'audio' },
      { id: 'd2', name: 'Other', media_type: 'audio' },
    ];
    flushInitialRequests(datasets);

    const activeCtx = TestBed.inject(ActiveContextService);
    activeCtx.setActivePair('d1', '');
    selection.selectOnly('dataset', ['d1', 'd2']);

    vi.spyOn(component['dialog'], 'confirmDestructive').mockReturnValue(Promise.resolve(true));

    await component.deleteSelectedDatasets();

    for (const req of httpMock.match((r) => r.method === 'DELETE')) {
      req.flush({});
    }

    // The deleted dataset must not linger as the interceptor's X-Dataset-Id.
    expect(activeCtx.datasetId).toBe('');
    expect(activeCtx.intentDatasetId).toBe('');
    expect(component.selectedDatasetIds.size).toBe(0);

    // Each delete triggers a registry refresh; drain them.
    for (const req of httpMock.match('/api/datasets/registry')) {
      if (!req.cancelled) req.flush({ datasets: [] });
    }
    for (const req of httpMock.match('/api/detectors/registry')) {
      if (!req.cancelled) req.flush({ detectors: [] });
    }
  });

  it('leaves the active context alone when the bulk delete spares the active dataset', async () => {
    const datasets = [
      { id: 'd1', name: 'Active', media_type: 'audio' },
      { id: 'd2', name: 'Other', media_type: 'audio' },
    ];
    flushInitialRequests(datasets);

    const activeCtx = TestBed.inject(ActiveContextService);
    activeCtx.setActivePair('d1', 'm1');
    selection.clear('dataset');
    selection.toggle('dataset', 'd2', true);

    vi.spyOn(component['dialog'], 'confirmDestructive').mockReturnValue(Promise.resolve(true));

    await component.deleteSelectedDatasets();

    httpMock.expectOne('/api/datasets/registry/d2').flush({});

    expect(activeCtx.datasetId).toBe('d1');
    expect(activeCtx.modelId).toBe('m1');

    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  });

  // #4380: the bulk-delete confirmation ran the question, the enumerated
  // names and the parenthetical together on one line. The names now sit on
  // their own line below the question (the detail gets its own line from
  // confirmDestructive).
  it('puts the names in a bulk delete confirmation on their own line', async () => {
    const datasets = [
      { id: 'd1', name: 'DS1', media_type: 'audio' },
      { id: 'd2', name: 'DS2', media_type: 'audio' },
    ];
    const detectors = [
      { id: 'm1', name: 'detector1', media_type: 'audio' },
      { id: 'm2', name: 'detector2', media_type: 'audio' },
    ];
    flushInitialRequests(datasets, detectors);
    selection.selectOnly('dataset', ['d1', 'd2']);
    selection.selectOnly('detector', ['m1', 'm2']);

    const confirm = vi.spyOn(component['dialog'], 'confirmDestructive').mockReturnValue(Promise.resolve(false));

    await component.deleteSelectedDetectors();
    expect(confirm).toHaveBeenLastCalledWith(
      'Delete 2 detectors?\n"detector1", "detector2"',
      '(This deletes your labels. The underlying media is unaffected.)',
    );

    await component.deleteSelectedDatasets();
    expect(confirm).toHaveBeenLastCalledWith(
      'Delete 2 datasets from your list?\n"DS1", "DS2"',
      '(Detectors are unaffected.)',
    );
  });

  it('should open and close importer modal via NewThingFlowsService', () => {
    flushInitialRequests();
    const flows = TestBed.inject(NewThingFlowsService);
    expect(component.importerModalOpen).toBe(false);
    component.openImporterModal();
    expect(component.importerModalOpen).toBe(true);
    flows.closeImporter();
    expect(component.importerModalOpen).toBe(false);
  });

  it('should render empty state when no datasets', () => {
    flushInitialRequests();
    // DatasetStateService is BehaviorSubject-backed, so the registry flush
    // updates the dashboard's view state through plain (non-signal) reads that
    // don't dirty the host under zoneless. markForCheck() marks it dirty so the
    // subsequent tick repaints over the now-settled state in one clean pass.
    fixture.changeDetectorRef.markForCheck();
    TestBed.tick();
    const el = fixture.nativeElement as HTMLElement;
    const empty = el.querySelector('.empty-state');
    expect(empty).toBeTruthy();
    expect((empty?.textContent || '').replace(/\s+/g, ' ')).toContain('No datasets yet. Click + to add one.');
  });

  describe('first-run hints (#4227)', () => {
    /** Flush the registry, then repaint (see the note in the empty-state test). */
    function renderWith(datasets: any[] = [], detectors: any[] = []): HTMLElement {
      flushInitialRequests(datasets, detectors);
      fixture.changeDetectorRef.markForCheck();
      TestBed.tick();
      return fixture.nativeElement as HTMLElement;
    }

    /** Load the user's settings with the given `show_usage_bars` mode. */
    async function loadUsageBarsSetting(mode: 'hide' | 'default' | 'view'): Promise<void> {
      TestBed.inject(SettingsStateService).load();
      TestBed.tick();
      httpMock.expectOne('/api/settings').flush({ show_usage_bars: mode });
      await settleResource();
      fixture.changeDetectorRef.markForCheck();
      TestBed.tick();
    }

    it('puts a working + in the dataset empty state, with an arrow to the header button', () => {
      const el = renderWith();
      const section = el.querySelectorAll('.dashboard-section')[0];
      const inlineAdd = section.querySelector('.empty-state .inline-add-btn') as HTMLButtonElement;
      expect(inlineAdd).toBeTruthy();
      expect(inlineAdd.textContent?.trim()).toBe('+');
      expect(section.querySelector('vt-pointer-arrow')).toBeTruthy();

      inlineAdd.click();
      expect(component.importerModalOpen).toBe(true);
    });

    it('puts a working + in the draft-detector empty state, with an arrow to the header button', () => {
      const el = renderWith([{ id: 'd1', name: 'DS', media_type: 'image' }]);
      const section = el.querySelectorAll('.dashboard-section')[1];
      const empty = section.querySelector('.empty-state');
      expect((empty?.textContent || '').replace(/\s+/g, ' ')).toContain('No draft detectors. Click + to add one.');
      expect(section.querySelector('vt-pointer-arrow')).toBeTruthy();

      (section.querySelector('.empty-state .inline-add-btn') as HTMLButtonElement).click();
      expect(component.newDetectorModalOpen).toBe(true);
    });

    it('disables both detector tabs while there are no detectors', () => {
      const el = renderWith();
      const tabs = [...el.querySelectorAll('.detector-tab-bar .tab')] as HTMLButtonElement[];
      expect(tabs.map((t) => t.disabled)).toEqual([true, true]);
    });

    it('enables the detector tabs once a detector exists', () => {
      const el = renderWith([], [{ id: 'm1', name: 'M', media_type: 'image' }]);
      const tabs = [...el.querySelectorAll('.detector-tab-bar .tab')] as HTMLButtonElement[];
      expect(tabs.map((t) => t.disabled)).toEqual([false, false]);
    });

    it('locks the grid to Drafts when there are no detectors', () => {
      selection.setDetectorTab('autofind');
      renderWith();
      expect(selection.detectorTab()).toBe('drafts');
    });

    it('leaves the AutoFind tab alone once detectors exist', () => {
      selection.setDetectorTab('autofind');
      renderWith([], [{ id: 'm1', name: 'M', media_type: 'image', autofind: true }]);
      expect(selection.detectorTab()).toBe('autofind');
    });

    it('points at Train when an empty detector and a matching dataset are selected', () => {
      const el = renderWith(
        [{ id: 'd1', name: 'DS', media_type: 'image' }],
        [{ id: 'm1', name: 'M', media_type: 'image', num_training: 0 }],
      );
      expect(component.showTrainHint).toBe(true);
      const section = el.querySelectorAll('.dashboard-section')[1];
      expect(section.querySelector('.intro-hint')?.textContent?.trim()).toBe(
        'Click Train to teach your new detector.',
      );
      expect(section.querySelector('vt-pointer-arrow')).toBeTruthy();
    });

    it('drops the Train hint once the detector has labels', () => {
      const el = renderWith(
        [{ id: 'd1', name: 'DS', media_type: 'image' }],
        [{ id: 'm1', name: 'M', media_type: 'image', num_training: 12 }],
      );
      expect(component.showTrainHint).toBe(false);
      expect(el.querySelector('.intro-hint')).toBeNull();
    });

    it('does not point at Train when Train is disabled (media type mismatch)', () => {
      renderWith(
        [{ id: 'd1', name: 'DS', media_type: 'audio' }],
        [{ id: 'm1', name: 'M', media_type: 'image', num_training: 0 }],
      );
      expect(component.labelEnabled).toBe(false);
      expect(component.showTrainHint).toBe(false);
    });

    /** Set both usage probes' readings, flagging each `low` or not. */
    function setUsage(ramLow: boolean, diskLow: boolean): void {
      const gb = 1024 ** 3;
      component.ramUsage.set({ total: 16 * gb, used: 8 * gb, free: 8 * gb, datasetBytes: gb, low: ramLow });
      component.diskUsage.set({ total: 500 * gb, used: 499 * gb, free: gb, datasetBytes: gb, low: diskLow });
      fixture.changeDetectorRef.markForCheck();
      TestBed.tick();
    }

    function shownBars(el: HTMLElement): string[] {
      return [...el.querySelectorAll('vt-usage-bar')].map((b) => (b.classList.contains('side-left') ? 'ram' : 'disk'));
    }

    it('hides the RAM / Disk bars by default while neither is low, detectors or not', () => {
      const el = renderWith([], [{ id: 'm1', name: 'M' }]);
      setUsage(false, false);
      expect(shownBars(el)).toEqual([]);
    });

    it('shows by default only the bar whose free space is low for the datasets', () => {
      const el = renderWith();
      setUsage(false, true);
      expect(shownBars(el)).toEqual(['disk']);
      setUsage(true, false);
      expect(shownBars(el)).toEqual(['ram']);
      setUsage(true, true);
      expect(shownBars(el)).toEqual(['ram', 'disk']);
    });

    it('hides the bars by default before the first usage poll lands', () => {
      const el = renderWith([], [{ id: 'm1', name: 'M' }]);
      expect(shownBars(el)).toEqual([]);
    });

    it('always shows the bars on "view" and never on "hide"', async () => {
      const el = renderWith([], [{ id: 'm1', name: 'M' }]);
      setUsage(true, true);
      await loadUsageBarsSetting('hide');
      expect(shownBars(el)).toEqual([]);

      setUsage(false, false);
      TestBed.inject(SettingsStateService).update({ show_usage_bars: 'view' }).subscribe();
      httpMock.expectOne('/api/settings').flush({ show_usage_bars: 'view' });
      TestBed.tick();
      expect(shownBars(el)).toEqual(['ram', 'disk']);
    });
  });

  it('should render dataset table when datasets exist', () => {
    const datasets = [{ id: 'd1', name: 'Test', media_type: 'audio', num_items: 5 }];
    flushInitialRequests(datasets);
    fixture.changeDetectorRef.markForCheck();
    TestBed.tick();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.dash-table')).toBeTruthy();
  });

  describe('onLabel', () => {
    // Phase 2: onLabel just navigates to the URL-encoded pair; the
    // `activeContextGuard` owns the dataset/detector load and any
    // progress polling. These tests cover the navigation contract,
    // not the load orchestration (which moved to the guard +
    // ContextSwitchService).

    it('navigates to /label/:datasetId/:detectorId for the selected pair', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);

      const routerSpy = vi.spyOn(component['router'], 'navigate').mockResolvedValue(true);
      component.onLabel();

      expect(routerSpy).toHaveBeenCalledWith(['/label', 'd1', 'm1']);
    });

    it('stores the selected model text_query in LabelSessionService before navigating', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', text_query: 'dog barking' }];
      flushInitialRequests(datasets, models);

      const session = TestBed.inject(LabelSessionService);
      vi.spyOn(component['router'], 'navigate').mockResolvedValue(true);
      component.onLabel();

      expect(session.textQuery).toBe('dog barking');
    });

    it('does nothing when no dataset is selected', () => {
      flushInitialRequests();
      selection.clear('dataset');
      const routerSpy = vi.spyOn(component['router'], 'navigate').mockResolvedValue(true);
      component.onLabel();
      expect(routerSpy).not.toHaveBeenCalled();
    });

    it('opens the new-detector modal (no navigation) when no model is selected', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      flushInitialRequests(datasets, []);
      selection.clear('detector');
      const routerSpy = vi.spyOn(component['router'], 'navigate').mockResolvedValue(true);

      component.onLabel();

      expect(routerSpy).not.toHaveBeenCalled();
      expect(component.newDetectorModalOpen).toBe(true);
      expect(component.trainAfterModelCreation).toBe(true);
    });
  });

  it('should keep refreshing after an HTTP error on the registry fetch', () => {
    flushInitialRequests();

    // Phase 2: the HTTP /api/dataset/progress poller was replaced by the
    // SSE stream, so the resilient "keep going after a transient error"
    // guarantee now lives in the registry refresh pipeline
    // (DatasetStateService catchError + retry). A failed registry fetch
    // must not wedge the dashboard: a subsequent refresh still resolves.
    component.refresh();
    // forkJoin subscribes to both registry fetches at once; erroring one
    // cancels its sibling, and DatasetStateService's catchError swallows
    // the failure so the pipeline stays alive for the next refresh.
    httpMock
      .expectOne('/api/datasets/registry')
      .error(new ProgressEvent('error'), { status: 500, statusText: 'Internal Server Error' });
    for (const req of httpMock.match('/api/detectors/registry')) {
      if (!req.cancelled) {
        req.error(new ProgressEvent('error'), { status: 500, statusText: 'Internal Server Error' });
      }
    }

    // The dashboard survives and a later refresh succeeds.
    component.refresh();
    httpMock
      .expectOne('/api/datasets/registry')
      .flush({ datasets: [{ id: 'd1', name: 'Recovered', media_type: 'audio' }] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });

    expect(component.datasets.length).toBe(1);
    expect(component.datasets[0].name).toBe('Recovered');
  });

  it('should load demo dataset on demoSelected', () => {
    flushInitialRequests();
    const flows = TestBed.inject(NewThingFlowsService);
    flows.openImporter();
    expect(component.importerModalOpen).toBe(true);
    const demo = { name: 'gtzan', label: 'GTZAN' } as any;
    flows.emitDemoSelected(demo);
    flows.closeImporter();

    expect(component.importerModalOpen).toBe(false);
    // Phase 2: the demo flow POSTs /api/dataset/load-demo and hands the
    // returned task_id to the SSE-driven loading-tasks poller. Loading
    // state and progress now live on the SSE stream
    // (DashboardLoadingTasksService), not on synchronous component fields,
    // so we only assert the request contract here.
    const req = httpMock.expectOne('/api/dataset/load-demo');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.name).toBe('gtzan');
    req.flush({ task_id: 't1' });
    // startProgressPolling subscribes to the SSE channel; no further HTTP
    // is issued until a loading-task event arrives, so nothing else to flush.
  });

  describe('loadDataset / loadDetector → active context', () => {
    // Loading from a dashboard card should make the item the active context
    // so the top-bar selector reflects it — but only *after* the load
    // settles (the SSE task reaches idle without error), never before, so we
    // don't reintroduce the H25 race where the interceptor tags requests
    // with an id the backend hasn't finished loading.

    it('promotes the loaded dataset to active only after the load settles', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'image' }];
      flushInitialRequests(datasets);

      const activeCtx = component['activeContext'];
      const setActive = vi.spyOn(activeCtx, 'setActivePair');
      // Capture the completion callback instead of running the real SSE poll.
      let onComplete: ((completed: LoadingTask[]) => void) | undefined;
      vi.spyOn(component['loadingTasksSvc'], 'startProgressPolling').mockImplementation(
        (_taskId?: string, cb?: (completed: LoadingTask[]) => void) => {
          onComplete = cb;
        },
      );

      component.loadDataset(datasets[0] as any);
      httpMock.expectOne('/api/datasets/registry/d1/load').flush({ task_id: 't1' });

      // Still not active mid-load.
      expect(setActive).not.toHaveBeenCalled();
      expect(onComplete).toBeTypeOf('function');

      onComplete!([]);
      expect(setActive).toHaveBeenCalledWith('d1', '');
    });

    it('promotes the loaded detector to active (preserving the dataset half) after settle', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'image', loaded: true }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);

      const activeCtx = component['activeContext'];
      // A dataset is already active; loading a detector must keep it.
      activeCtx.setActivePair('d1', '');
      const setActive = vi.spyOn(activeCtx, 'setActivePair');
      let onComplete: (() => void) | undefined;
      vi.spyOn(component['loadingTasksSvc'], 'startDetectorProgressPolling').mockImplementation(
        (cb?: () => void) => {
          onComplete = cb;
        },
      );

      component.loadDetector(models[0] as any);
      httpMock.expectOne('/api/detectors/registry/load').flush({ task_id: 't2' });

      expect(setActive).not.toHaveBeenCalled();
      onComplete!();
      expect(setActive).toHaveBeenCalledWith('d1', 'm1');
    });
  });

  describe('onCombineStarted → summary toast', () => {
    it('reports unique kept vs. duplicates dropped once the combine settles', () => {
      flushInitialRequests();

      const success = vi.spyOn(component['toast'], 'success');
      let onComplete: ((completed: any[]) => void) | undefined;
      vi.spyOn(component['loadingTasksSvc'], 'startProgressPolling').mockImplementation(
        (_taskId?: string, cb?: (completed: any[]) => void) => {
          onComplete = cb;
        },
      );

      component.onCombineStarted({ taskId: 'tc', numSources: 2, totalItems: 80 });
      expect(onComplete).toBeTypeOf('function');
      // No toast until the task settles.
      expect(success).not.toHaveBeenCalled();

      // Task done: it registered dataset "dc" with 50 unique items, so 30
      // of the 80 source items were duplicates.
      onComplete!([{ task_id: 'tc', status: 'idle', dataset_id: 'dc' } as any]);
      httpMock
        .expectOne('/api/datasets/registry')
        .flush({ datasets: [{ id: 'dc', name: 'Combined', media_type: 'audio', num_items: 50 }] });

      expect(success).toHaveBeenCalledWith({
        message: 'Combined 2 datasets into 1 — 50 unique kept, 30 duplicates dropped',
      });
    });

    it('skips the toast when the completed task has no dataset id', () => {
      flushInitialRequests();

      const success = vi.spyOn(component['toast'], 'success');
      let onComplete: ((completed: any[]) => void) | undefined;
      vi.spyOn(component['loadingTasksSvc'], 'startProgressPolling').mockImplementation(
        (_taskId?: string, cb?: (completed: any[]) => void) => {
          onComplete = cb;
        },
      );

      component.onCombineStarted({ taskId: 'tc', numSources: 2, totalItems: 80 });
      onComplete!([{ task_id: 'tc', status: 'idle' } as any]);

      // No dataset id → no registry fetch and no toast.
      httpMock.expectNone('/api/datasets/registry');
      expect(success).not.toHaveBeenCalled();
    });
  });
});
