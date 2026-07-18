import { Component } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter, Router } from '@angular/router';
import { BehaviorSubject } from 'rxjs';

import { ContextPulldownComponent } from './context-pulldown.component';
import { ActiveContextService } from '../../services/active-context.service';
import { ContextSwitchService } from '../../services/context-switch.service';
import { DatasetStateService } from '../../services/dataset-state.service';
import { DashboardSelectionService } from '../../services/dashboard-selection.service';
import { NewThingFlowsService } from '../../services/new-thing-flows.service';
import { PulldownControlService } from '../../services/pulldown-control.service';
import { RunningJobsService, pairKey } from '../../services/running-jobs.service';
import { DatasetRegistryEntry } from '../../models/api.models';
import { DetectorRegistryEntry } from '../../generated/api-client/models/detector-registry-entry';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Specs for the top-bar context switcher UI. The pulldown is the user-facing
 * front of the context-selection pipeline: it renders one row per registry
 * entry, dims incompatible partners, and — off the Dashboard — drives
 * `ContextSwitchService.switchTo` (which promotes the active pair the HTTP
 * interceptor tags onto requests). These tests cover row construction,
 * compatibility dimming, pick routing (switch vs Dashboard multi-select),
 * keyboard/open behavior, and the add-new auto-select handoff.
 */

/** Stub the running-jobs poller so subscribing doesn't fire background HTTP;
 *  tests push busy-pair snapshots directly. */
class RunningJobsStub {
  readonly busyPairsSubject = new BehaviorSubject<Map<string, string[]>>(new Map());
  readonly busyPairs$ = this.busyPairsSubject.asObservable();
}

/** Placeholder target for the routes the lock-behavior specs navigate to;
 *  the pulldown only reads the URL, never these components. */
@Component({ selector: 'vt-dummy-view', standalone: true, template: '' })
class DummyViewComponent {}

describe('ContextPulldownComponent', () => {
  let fixture: ComponentFixture<ContextPulldownComponent>;
  let component: ContextPulldownComponent;
  let datasetState: DatasetStateService;
  let activeContext: ActiveContextService;
  let contextSwitch: ContextSwitchService;
  let dashSelection: DashboardSelectionService;
  let newThingFlows: NewThingFlowsService;
  let pulldownControl: PulldownControlService;
  let runningJobs: RunningJobsStub;
  let router: Router;

  function makeDataset(
    id: string,
    name: string,
    overrides: Partial<DatasetRegistryEntry> = {},
  ): DatasetRegistryEntry {
    return { id, name, media_type: 'audio', loaded: true, ...overrides } as DatasetRegistryEntry;
  }

  function makeDetector(
    id: string,
    name: string,
    overrides: Partial<DetectorRegistryEntry> = {},
  ): DetectorRegistryEntry {
    return {
      id,
      name,
      media_type: 'audio',
      detector_loaded: true,
      ...overrides,
    } as DetectorRegistryEntry;
  }

  /** Push registry contents through the signal-backed state, then tick so the
   *  component's `datasets$`/`detectors$` bridges emit and rebuild the rows. */
  function setRegistry(
    datasets: DatasetRegistryEntry[],
    detectors: DetectorRegistryEntry[] = [],
  ): void {
    const ds = datasetState as unknown as {
      _datasets: { set: (v: unknown) => void };
      _detectors: { set: (v: unknown) => void };
      _loaded: { set: (v: unknown) => void };
    };
    ds._datasets.set(datasets);
    ds._detectors.set(detectors);
    ds._loaded.set(true);
    TestBed.tick();
  }

  async function createPulldown(kind: 'dataset' | 'detector' = 'dataset'): Promise<void> {
    fixture = TestBed.createComponent(ContextPulldownComponent);
    fixture.componentRef.setInput('kind', kind);
    component = fixture.componentInstance;
    // Flush the scheduled initial change detection so ngOnInit runs and wires
    // the service subscriptions.
    await settleZoneless(fixture);
  }

  beforeEach(async () => {
    await configureZoneless({
      imports: [ContextPulldownComponent],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([
          { path: 'dashboard', component: DummyViewComponent },
          { path: 'browse/:datasetId', component: DummyViewComponent },
          { path: 'find/:datasetId/:detectorId', component: DummyViewComponent },
          { path: 'label/:datasetId/:detectorId', component: DummyViewComponent },
        ]),
        { provide: RunningJobsService, useClass: RunningJobsStub },
      ],
    }).compileComponents();

    datasetState = TestBed.inject(DatasetStateService);
    activeContext = TestBed.inject(ActiveContextService);
    contextSwitch = TestBed.inject(ContextSwitchService);
    dashSelection = TestBed.inject(DashboardSelectionService);
    newThingFlows = TestBed.inject(NewThingFlowsService);
    pulldownControl = TestBed.inject(PulldownControlService);
    runningJobs = TestBed.inject(RunningJobsService) as unknown as RunningJobsStub;
    router = TestBed.inject(Router);
  });

  describe('row construction', () => {
    it('builds one row per dataset for a dataset pulldown, sorted by name', async () => {
      await createPulldown('dataset');
      setRegistry([makeDataset('d2', 'Beta'), makeDataset('d1', 'Alpha')]);

      expect(component.rows.map((r) => r.name)).toEqual(['Alpha', 'Beta']);
      expect(component.rows.map((r) => r.id)).toEqual(['d1', 'd2']);
    });

    it('builds detector rows for a detector pulldown', async () => {
      await createPulldown('detector');
      setRegistry([makeDataset('d1', 'DS')], [makeDetector('m1', 'Det One')]);

      expect(component.rows.map((r) => r.id)).toEqual(['m1']);
      expect(component.rows[0].loaded).toBe(true);
    });

    it('marks the intent dataset active and reflects it in activeName', async () => {
      await createPulldown('dataset');
      activeContext.setIntent('d1', '');
      setRegistry([makeDataset('d1', 'Alpha'), makeDataset('d2', 'Beta')]);

      const rows = new Map(component.rows.map((r) => [r.id, r]));
      expect(rows.get('d1')!.active).toBe(true);
      expect(rows.get('d2')!.active).toBe(false);
      expect(component.activeName).toBe('Alpha');
      expect(component.activeRowExists).toBe(true);
    });

    it('shows the placeholder (empty activeName) when nothing is active', async () => {
      await createPulldown('dataset');
      setRegistry([makeDataset('d1', 'Alpha')]);
      expect(component.activeName).toBe('');
      expect(component.activeRowExists).toBe(false);
    });
  });

  describe('compatibility dimming', () => {
    it('dims a dataset that is incompatible with the active detector', async () => {
      await createPulldown('dataset');
      // Active detector half is an image detector; the audio dataset can't pair.
      activeContext.setIntent('', 'm1');
      setRegistry(
        [makeDataset('d1', 'AudioDS', { media_type: 'audio' })],
        [makeDetector('m1', 'ImgDet', { media_type: 'image' })],
      );

      const row = component.rows.find((r) => r.id === 'd1')!;
      expect(row.compatibleWithOther).toBe(false);
      expect(row.incompatReason).toContain('image');
      expect(row.incompatReason).toContain('audio');
    });

    it('leaves a matching-media-type pair compatible', async () => {
      await createPulldown('dataset');
      activeContext.setIntent('', 'm1');
      setRegistry(
        [makeDataset('d1', 'AudioDS', { media_type: 'audio' })],
        [makeDetector('m1', 'AudioDet', { media_type: 'audio' })],
      );

      const row = component.rows.find((r) => r.id === 'd1')!;
      expect(row.compatibleWithOther).toBe(true);
      expect(row.incompatReason).toBe('');
    });
  });

  describe('pickRow (off the Dashboard)', () => {
    it('switches the active pair via ContextSwitchService and closes', async () => {
      await createPulldown('dataset');
      activeContext.setIntent('d2', 'm1');
      setRegistry([makeDataset('d1', 'Alpha'), makeDataset('d2', 'Beta')], [makeDetector('m1', 'Det')]);
      const switchSpy = vi.spyOn(contextSwitch, 'switchTo').mockImplementation(() => {});

      component.openMenu();
      component.pickRow(component.rows.find((r) => r.id === 'd1')!);

      // Carries the current intent detector half through.
      expect(switchSpy).toHaveBeenCalledWith('d1', 'm1');
      expect(component.open).toBe(false);
    });

    it('does not switch when the picked row is already active', async () => {
      await createPulldown('dataset');
      activeContext.setIntent('d1', '');
      setRegistry([makeDataset('d1', 'Alpha')]);
      const switchSpy = vi.spyOn(contextSwitch, 'switchTo');

      component.openMenu();
      component.pickRow(component.rows.find((r) => r.id === 'd1')!);

      expect(switchSpy).not.toHaveBeenCalled();
      expect(component.open).toBe(false);
    });
  });

  describe('pickRow (Dashboard multi-select mode)', () => {
    it('routes a pick to the Dashboard selection instead of a context switch', async () => {
      await createPulldown('dataset');
      dashSelection.setDashboardVisible(true);
      setRegistry([makeDataset('d1', 'Alpha'), makeDataset('d2', 'Beta')]);
      const selectSpy = vi.spyOn(dashSelection, 'requestSelect');
      const switchSpy = vi.spyOn(contextSwitch, 'switchTo');

      component.pickRow(component.rows.find((r) => r.id === 'd2')!);

      expect(selectSpy).toHaveBeenCalledWith('dataset', 'd2');
      expect(switchSpy).not.toHaveBeenCalled();
    });

    it('collapses a multi-selection to the "Multiple" label', async () => {
      await createPulldown('dataset');
      dashSelection.setDashboardVisible(true);
      dashSelection.setDatasetIds(['d1', 'd2']);
      setRegistry([makeDataset('d1', 'Alpha'), makeDataset('d2', 'Beta')]);

      expect(component.rows.filter((r) => r.active).length).toBe(2);
      expect(component.activeName).toBe('Multiple');
    });
  });

  describe('open / focus behavior', () => {
    it('focuses the first compatible row on open', async () => {
      await createPulldown('dataset');
      activeContext.setIntent('', 'm1');
      setRegistry(
        [
          makeDataset('d1', 'Alpha', { media_type: 'image' }), // incompatible, sorts first
          makeDataset('d2', 'Beta', { media_type: 'audio' }), // compatible
        ],
        [makeDetector('m1', 'AudioDet', { media_type: 'audio' })],
      );

      component.openMenu();
      expect(component.open).toBe(true);
      expect(component.rows[component.focusedIndex].id).toBe('d2');
    });

    it('close() resets open state and focus', async () => {
      await createPulldown('dataset');
      setRegistry([makeDataset('d1', 'Alpha')]);
      component.openMenu();
      component.close();
      expect(component.open).toBe(false);
      expect(component.focusedIndex).toBe(-1);
    });

    it('opens on the pulldown-control open signal', async () => {
      await createPulldown('detector');
      setRegistry([makeDataset('d1', 'DS')], [makeDetector('m1', 'Det')]);
      expect(component.open).toBe(false);

      pulldownControl.requestOpen('detector');
      TestBed.tick();

      expect(component.open).toBe(true);
    });
  });

  describe('keyboard', () => {
    function key(k: string): KeyboardEvent {
      return { key: k, preventDefault: () => {} } as KeyboardEvent;
    }

    it('ArrowDown opens the closed menu', async () => {
      await createPulldown('dataset');
      setRegistry([makeDataset('d1', 'Alpha')]);
      component.onKeydown(key('ArrowDown'));
      expect(component.open).toBe(true);
    });

    it('ArrowDown / ArrowUp wrap around the row list when open', async () => {
      await createPulldown('dataset');
      setRegistry([makeDataset('d1', 'Alpha'), makeDataset('d2', 'Beta')]);
      component.openMenu();
      component.focusedIndex = 1; // last row
      component.onKeydown(key('ArrowDown'));
      expect(component.focusedIndex).toBe(0); // wrapped to first
      component.onKeydown(key('ArrowUp'));
      expect(component.focusedIndex).toBe(1); // wrapped back to last
    });

    it('Enter on a focused row picks it', async () => {
      await createPulldown('dataset');
      activeContext.setIntent('d2', '');
      setRegistry([makeDataset('d1', 'Alpha'), makeDataset('d2', 'Beta')]);
      const switchSpy = vi.spyOn(contextSwitch, 'switchTo').mockImplementation(() => {});

      component.openMenu();
      component.focusedIndex = component.rows.findIndex((r) => r.id === 'd1');
      component.onKeydown(key('Enter'));

      expect(switchSpy).toHaveBeenCalledWith('d1', '');
    });

    it('Escape closes the open menu', async () => {
      await createPulldown('dataset');
      setRegistry([makeDataset('d1', 'Alpha')]);
      component.openMenu();
      component.onKeydown(key('Escape'));
      expect(component.open).toBe(false);
    });
  });

  describe('add new', () => {
    it('opens the importer flow for a dataset pulldown', async () => {
      await createPulldown('dataset');
      setRegistry([makeDataset('d1', 'Alpha')]);
      const openSpy = vi.spyOn(newThingFlows, 'openImporter');

      component.addNew();
      expect(openSpy).toHaveBeenCalled();
      expect(component.open).toBe(false);
    });

    it('opens the new-detector flow seeded from the active dataset', async () => {
      await createPulldown('detector');
      activeContext.setIntent('d1', '');
      setRegistry(
        [makeDataset('d1', 'Alpha', { media_type: 'image', embedder: 'siglip' })],
        [],
      );
      const openSpy = vi.spyOn(newThingFlows, 'openNewDetector');

      component.addNew();
      expect(openSpy).toHaveBeenCalledWith({
        defaultMediaType: 'image',
        datasetEmbedder: 'siglip',
      });
    });

    it('auto-switches to a newly created detector', async () => {
      await createPulldown('detector');
      activeContext.setIntent('d1', '');
      setRegistry([makeDataset('d1', 'DS')], []);
      const switchSpy = vi.spyOn(contextSwitch, 'switchTo').mockImplementation(() => {});

      component.addNew(); // arms awaitingNew for this detector pulldown
      newThingFlows.emitDetectorCreated('mNew');
      TestBed.tick();

      expect(switchSpy).toHaveBeenCalledWith('d1', 'mNew');
    });
  });

  describe('busy indicator', () => {
    it('flags a row whose pair has a running job', async () => {
      await createPulldown('dataset');
      activeContext.setIntent('d1', 'm1'); // active detector half = m1
      setRegistry([makeDataset('d1', 'Alpha')], [makeDetector('m1', 'Det')]);

      runningJobs.busyPairsSubject.next(new Map([[pairKey('d1', 'm1'), ['learned-sort']]]));
      TestBed.tick();

      const row = component.rows.find((r) => r.id === 'd1')!;
      expect(row.busy).toBe(true);
      expect(row.busyJobTypes).toEqual(['learned-sort']);
      expect(component.busyTitle(row)).toContain('Learned sort');
    });
  });

  describe('registry error', () => {
    it('surfaces a registry load error', async () => {
      await createPulldown('dataset');
      const ds = datasetState as unknown as { _error: { set: (v: unknown) => void } };
      ds._error.set("Couldn't load datasets and detectors.");
      TestBed.tick();
      expect(component.registryError).toContain("Couldn't load");
    });
  });

  describe('locked on browse / find views', () => {
    it('is unlocked on the dashboard', async () => {
      await createPulldown('dataset');
      await router.navigate(['/dashboard']);
      await settleZoneless(fixture);
      expect(component.locked).toBe(false);
    });

    it('locks on the browse (VTSBrowser) view', async () => {
      await createPulldown('dataset');
      await router.navigate(['/browse', 'd1']);
      await settleZoneless(fixture);
      expect(component.locked).toBe(true);
    });

    it('locks on the find-results view', async () => {
      await createPulldown('detector');
      await router.navigate(['/find', 'd1', 'm1']);
      await settleZoneless(fixture);
      expect(component.locked).toBe(true);
    });

    it('stays unlocked on the label / train view', async () => {
      await createPulldown('detector');
      await router.navigate(['/label', 'd1', 'm1']);
      await settleZoneless(fixture);
      expect(component.locked).toBe(false);
    });

    it('does not open when locked, even via the pulldown-control signal', async () => {
      await createPulldown('dataset');
      setRegistry([makeDataset('d1', 'Alpha')]);
      await router.navigate(['/browse', 'd1']);
      await settleZoneless(fixture);

      // Direct open attempt is a no-op.
      component.openMenu();
      expect(component.open).toBe(false);

      // Programmatic open (keyboard shortcut path) is also swallowed.
      pulldownControl.requestOpen('dataset');
      TestBed.tick();
      expect(component.open).toBe(false);
    });

    it('closes an open menu when navigating into a locked view', async () => {
      await createPulldown('dataset');
      setRegistry([makeDataset('d1', 'Alpha')]);
      component.openMenu();
      expect(component.open).toBe(true);

      await router.navigate(['/browse', 'd1']);
      await settleZoneless(fixture);
      expect(component.open).toBe(false);
      expect(component.locked).toBe(true);
    });

    it('still displays the active label while locked', async () => {
      await createPulldown('detector');
      activeContext.setIntent('d1', 'm1');
      setRegistry([makeDataset('d1', 'DS')], [makeDetector('m1', 'My Detector')]);
      await router.navigate(['/find', 'd1', 'm1']);
      await settleZoneless(fixture);

      expect(component.locked).toBe(true);
      expect(component.activeName).toBe('My Detector');
      expect(component.activeRowExists).toBe(true);
    });
  });
});
