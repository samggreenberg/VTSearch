import { TestBed } from '@angular/core/testing';
import {
  ActivatedRouteSnapshot,
  GuardResult,
  MaybeAsync,
  UrlTree,
  convertToParamMap,
  provideRouter,
} from '@angular/router';
import { isObservable, of } from 'rxjs';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { peopleContextGuard } from './people-context.guard';
import { ActiveContextService } from '../services/active-context.service';
import { ContextSwitchService } from '../services/context-switch.service';
import { DatasetStateService } from '../services/dataset-state.service';
import { ToastService } from '../services/toast.service';
import { DatasetRegistryEntry } from '../models/api.models';
import { configureZoneless } from '../testing/zoneless-testbed';

/**
 * `/people/:datasetId` (#5590) is Browse's dataset-only guard for a Face
 * dataset alone: anything else has no people, so it bounces to the Dashboard
 * with a toast before the view asks.
 */
describe('peopleContextGuard', () => {
  let datasetState: DatasetStateService;
  let activeContext: ActiveContextService;
  let toast: ToastService;
  let contextSwitch: ContextSwitchService;

  function setRegistry(datasets: DatasetRegistryEntry[]): void {
    const ds = datasetState as unknown as {
      _datasets: { set: (v: unknown) => void };
      _loaded: { set: (v: unknown) => void };
    };
    ds._datasets.set(datasets);
    ds._loaded.set(true);
    TestBed.tick();
  }

  function run(datasetId: string | null): Promise<GuardResult> {
    const route = {
      paramMap: convertToParamMap(datasetId === null ? {} : { datasetId }),
    } as unknown as ActivatedRouteSnapshot;
    const result: MaybeAsync<GuardResult> = TestBed.runInInjectionContext(() => peopleContextGuard(route, {} as never));
    if (isObservable(result)) {
      return new Promise((resolve) => result.subscribe((v) => resolve(v)));
    }
    return Promise.resolve(result as GuardResult);
  }

  beforeEach(() => {
    configureZoneless({ providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])] });
    datasetState = TestBed.inject(DatasetStateService);
    activeContext = TestBed.inject(ActiveContextService);
    toast = TestBed.inject(ToastService);
    contextSwitch = TestBed.inject(ContextSwitchService);
  });

  it('lets a Face dataset in, loading it and keeping the active detector', async () => {
    setRegistry([{ id: 'f1', name: 'Faces', media_type: 'face', loaded: false } as DatasetRegistryEntry]);
    activeContext.setActivePair('other', 'm1');
    const apply = vi.spyOn(contextSwitch, 'applyActivePair').mockImplementation(() => of(undefined));

    expect(await run('f1')).toBe(true);
    expect(apply).toHaveBeenCalledWith('f1', 'm1');
  });

  it('fast-paths a Face dataset already loaded and active', async () => {
    setRegistry([{ id: 'f1', name: 'Faces', media_type: 'face', loaded: true } as DatasetRegistryEntry]);
    activeContext.setActivePair('f1', '');
    const apply = vi.spyOn(contextSwitch, 'applyActivePair');

    expect(await run('f1')).toBe(true);
    expect(apply).not.toHaveBeenCalled();
  });

  it('bounces any other dataset to the Dashboard, saying why, without loading it', async () => {
    setRegistry([{ id: 'i1', name: 'Photos', media_type: 'image', loaded: true } as DatasetRegistryEntry]);
    const apply = vi.spyOn(contextSwitch, 'applyActivePair');

    const result = await run('i1');
    expect(result instanceof UrlTree).toBe(true);
    expect((result as UrlTree).toString()).toBe('/dashboard');
    expect(apply).not.toHaveBeenCalled();
    expect(toast.toasts.length).toBe(1);
    expect(toast.toasts[0].message).toContain("'Photos' is not one");
  });

  it('bounces an unknown dataset as Browse does', async () => {
    setRegistry([]);
    const result = await run('missing');
    expect((result as UrlTree).toString()).toBe('/dashboard');
    expect(toast.toasts[0].message).toContain("'missing'");
  });

  it('bounces a missing id at once', async () => {
    expect(((await run(null)) as UrlTree).toString()).toBe('/dashboard');
  });
});
