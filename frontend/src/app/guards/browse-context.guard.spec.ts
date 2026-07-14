import { TestBed } from '@angular/core/testing';
import {
  ActivatedRouteSnapshot,
  GuardResult,
  MaybeAsync,
  Router,
  UrlTree,
  convertToParamMap,
  provideRouter,
} from '@angular/router';
import { isObservable, of } from 'rxjs';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { browseContextGuard } from './browse-context.guard';
import { ActiveContextService } from '../services/active-context.service';
import { ContextSwitchService } from '../services/context-switch.service';
import { DatasetStateService } from '../services/dataset-state.service';
import { ToastService } from '../services/toast.service';
import { DatasetRegistryEntry } from '../models/api.models';
import { configureZoneless } from '../testing/zoneless-testbed';

/**
 * Specs for the browse route guard: given `/browse/:datasetId`, decide whether
 * the browse view may activate, loading the dataset into the backend first when
 * needed. Together with the active-context guard and interceptor coverage, this
 * pins the whole context-selection pipeline the backend's header checks lean on.
 */
describe('browseContextGuard', () => {
  let router: Router;
  let datasetState: DatasetStateService;
  let activeContext: ActiveContextService;
  let toast: ToastService;
  let contextSwitch: ContextSwitchService;

  function makeRoute(datasetId: string | null): ActivatedRouteSnapshot {
    return {
      paramMap: convertToParamMap(datasetId === null ? {} : { datasetId }),
    } as unknown as ActivatedRouteSnapshot;
  }

  /** Seed the registry as loaded with the given datasets. */
  function setRegistry(datasets: DatasetRegistryEntry[]): void {
    const ds = datasetState as unknown as {
      _datasets: { set: (v: unknown) => void };
      _loaded: { set: (v: unknown) => void };
    };
    ds._datasets.set(datasets);
    ds._loaded.set(true);
    // `loaded$` is a `toObservable` bridge over the `_loaded` signal; tick so
    // the guard's `loaded$` subscription replays `true` when it subscribes.
    TestBed.tick();
  }

  function runGuard(datasetId: string | null): MaybeAsync<GuardResult> {
    return TestBed.runInInjectionContext(() => browseContextGuard(makeRoute(datasetId), {} as never));
  }

  function resolveGuard(result: MaybeAsync<GuardResult>): Promise<GuardResult> {
    if (isObservable(result)) {
      return new Promise<GuardResult>((resolve) => {
        (result as { subscribe: (cb: (v: GuardResult) => void) => void }).subscribe((v) =>
          resolve(v),
        );
      });
    }
    return Promise.resolve(result as GuardResult);
  }

  beforeEach(() => {
    configureZoneless({
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    });
    router = TestBed.inject(Router);
    datasetState = TestBed.inject(DatasetStateService);
    activeContext = TestBed.inject(ActiveContextService);
    toast = TestBed.inject(ToastService);
    contextSwitch = TestBed.inject(ContextSwitchService);
  });

  it('redirects to /dashboard when the datasetId param is missing', () => {
    const result = runGuard(null);
    // Synchronous UrlTree, not an observable — no registry wait involved.
    expect(result instanceof UrlTree).toBe(true);
    expect((result as UrlTree).toString()).toBe('/dashboard');
  });

  it('short-circuits to true for ephemeral __detpos__ browse contexts', () => {
    const applySpy = vi.spyOn(contextSwitch, 'applyActivePair');
    const result = runGuard('__detpos__abc');
    // Returned synchronously; the registry/load checks are skipped entirely.
    expect(result).toBe(true);
    expect(applySpy).not.toHaveBeenCalled();
  });

  it('redirects + toasts when the dataset id is not in the registry', async () => {
    setRegistry([]);
    const result = await resolveGuard(runGuard('missing'));
    expect(result instanceof UrlTree).toBe(true);
    expect((result as UrlTree).toString()).toBe('/dashboard');
    expect(toast.toasts.length).toBe(1);
    expect(toast.toasts[0].message).toContain("'missing'");
  });

  it('fast-paths to true when the active dataset is already loaded and matches', async () => {
    setRegistry([
      { id: 'd1', name: 'DS', media_type: 'audio', loaded: true } as DatasetRegistryEntry,
    ]);
    activeContext.setActivePair('d1', 'm1');
    const applySpy = vi.spyOn(contextSwitch, 'applyActivePair');

    const result = await resolveGuard(runGuard('d1'));
    expect(result).toBe(true);
    // Genuinely loaded + matching active id => no load needed.
    expect(applySpy).not.toHaveBeenCalled();
  });

  it('loads via applyActivePair when the matching dataset is not loaded', async () => {
    setRegistry([
      { id: 'd1', name: 'DS', media_type: 'audio', loaded: false } as DatasetRegistryEntry,
    ]);
    activeContext.setActivePair('d1', 'm1');
    const applySpy = vi
      .spyOn(contextSwitch, 'applyActivePair')
      .mockImplementation(() => of(undefined));

    const result = await resolveGuard(runGuard('d1'));
    expect(result).toBe(true);
    // A matching active id is NOT sufficient — the dataset may have been
    // evicted since, so the guard loads it before allowing the browse view.
    expect(applySpy).toHaveBeenCalledWith('d1', 'm1');
  });

  it('loads via applyActivePair when a switch is in flight even if ids match', async () => {
    setRegistry([
      { id: 'd1', name: 'DS', media_type: 'audio', loaded: true } as DatasetRegistryEntry,
    ]);
    activeContext.setActivePair('d1', 'm1');
    vi.spyOn(contextSwitch, 'switching', 'get').mockReturnValue(true);
    const applySpy = vi
      .spyOn(contextSwitch, 'applyActivePair')
      .mockImplementation(() => of(undefined));

    const result = await resolveGuard(runGuard('d1'));
    expect(result).toBe(true);
    expect(applySpy).toHaveBeenCalledWith('d1', 'm1');
  });

  it('loads via applyActivePair when the URL dataset differs from the active one', async () => {
    setRegistry([
      { id: 'd1', name: 'One', media_type: 'audio', loaded: true } as DatasetRegistryEntry,
      { id: 'd2', name: 'Two', media_type: 'audio', loaded: true } as DatasetRegistryEntry,
    ]);
    activeContext.setActivePair('d1', 'm1');
    const applySpy = vi
      .spyOn(contextSwitch, 'applyActivePair')
      .mockImplementation(() => of(undefined));

    const result = await resolveGuard(runGuard('d2'));
    expect(result).toBe(true);
    // Carries the active detector half through to the load.
    expect(applySpy).toHaveBeenCalledWith('d2', 'm1');
  });

  it('refreshes the registry when it has not loaded yet, then resolves once loaded', async () => {
    const refreshSpy = vi.spyOn(datasetState, 'refresh');
    // Registry starts unloaded (_loaded is false by default).
    const result = runGuard('d1');
    expect(refreshSpy).toHaveBeenCalled();

    // Registry arrives with d1 loaded; the guard's `filter(loaded)` now passes.
    setRegistry([
      { id: 'd1', name: 'DS', media_type: 'audio', loaded: true } as DatasetRegistryEntry,
    ]);
    activeContext.setActivePair('d1', '');
    expect(await resolveGuard(result)).toBe(true);
  });
});
