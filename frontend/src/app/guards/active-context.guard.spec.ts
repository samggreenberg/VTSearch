import { TestBed } from '@angular/core/testing';
import {
  ActivatedRouteSnapshot,
  GuardResult,
  MaybeAsync,
  Router,
  RouterStateSnapshot,
  UrlTree,
  convertToParamMap,
  provideRouter,
} from '@angular/router';
import { isObservable, of } from 'rxjs';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { activeContextGuard } from './active-context.guard';
import { ActiveContextService } from '../services/active-context.service';
import { ContextSwitchService } from '../services/context-switch.service';
import { DatasetStateService } from '../services/dataset-state.service';
import { ToastService } from '../services/toast.service';
import { DatasetRegistryEntry, DetectorRegistryEntry } from '../models/api.models';

describe('activeContextGuard', () => {
  let router: Router;
  let datasetState: DatasetStateService;
  let activeContext: ActiveContextService;
  let toast: ToastService;
  let contextSwitch: ContextSwitchService;

  function makeRoute(datasetId: string, detectorId: string): ActivatedRouteSnapshot {
    return {
      paramMap: convertToParamMap({ datasetId, detectorId }),
    } as unknown as ActivatedRouteSnapshot;
  }

  function setRegistry(
    datasets: DatasetRegistryEntry[],
    detectors: DetectorRegistryEntry[],
  ): void {
    const ds = datasetState as unknown as {
      datasetsSubject: { next: (v: unknown) => void };
      detectorsSubject: { next: (v: unknown) => void };
      loadedSubject: { next: (v: unknown) => void };
    };
    ds.datasetsSubject.next(datasets);
    ds.detectorsSubject.next(detectors);
    ds.loadedSubject.next(true);
  }

  function runGuard(datasetId: string, detectorId: string): MaybeAsync<GuardResult> {
    return TestBed.runInInjectionContext(() =>
      activeContextGuard(makeRoute(datasetId, detectorId), {} as RouterStateSnapshot),
    );
  }

  function resolveGuard(result: MaybeAsync<GuardResult>): Promise<GuardResult> {
    if (isObservable(result)) {
      return new Promise<GuardResult>((resolve) => {
        (result as { subscribe: (cb: (v: GuardResult) => void) => void }).subscribe(
          (v) => resolve(v),
        );
      });
    }
    return Promise.resolve(result as GuardResult);
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
      ],
    });
    router = TestBed.inject(Router);
    datasetState = TestBed.inject(DatasetStateService);
    activeContext = TestBed.inject(ActiveContextService);
    toast = TestBed.inject(ToastService);
    contextSwitch = TestBed.inject(ContextSwitchService);
  });

  it('redirects to /dashboard when either id is missing from the URL', async () => {
    setRegistry([], []);
    const result = await resolveGuard(runGuard('', ''));
    expect(result instanceof UrlTree).toBeTrue();
    expect((result as UrlTree).toString()).toBe('/dashboard');
  });

  it('redirects + toasts when the dataset id is not in the registry', async () => {
    setRegistry(
      [],
      [{ id: 'm1', name: 'Det', media_type: 'audio' } as DetectorRegistryEntry],
    );
    const result = await resolveGuard(runGuard('missing', 'm1'));
    expect(result instanceof UrlTree).toBeTrue();
    expect(toast.toasts.length).toBe(1);
    expect(toast.toasts[0].message).toContain("'missing'");
  });

  it('redirects + toasts when the detector id is not in the registry', async () => {
    setRegistry(
      [{ id: 'd1', name: 'DS', media_type: 'audio' } as DatasetRegistryEntry],
      [],
    );
    const result = await resolveGuard(runGuard('d1', 'missing'));
    expect(result instanceof UrlTree).toBeTrue();
    expect(toast.toasts.length).toBe(1);
    expect(toast.toasts[0].message).toContain("'missing'");
  });

  it('fast-paths to true when the active pair already matches the URL', async () => {
    setRegistry(
      [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true } as DatasetRegistryEntry],
      [{ id: 'm1', name: 'Det', media_type: 'audio', detector_loaded: true } as DetectorRegistryEntry],
    );
    activeContext.setActivePair('d1', 'm1');

    const result = await resolveGuard(runGuard('d1', 'm1'));
    expect(result).toBeTrue();
  });

  it('flips the active pair via ContextSwitchService when ids differ', async () => {
    setRegistry(
      [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true } as DatasetRegistryEntry],
      [{ id: 'm1', name: 'Det', media_type: 'audio', detector_loaded: true } as DetectorRegistryEntry],
    );
    const applySpy = spyOn(contextSwitch, 'applyActivePair').and.callFake(() => of(undefined));

    const result = await resolveGuard(runGuard('d1', 'm1'));
    expect(result).toBeTrue();
    expect(applySpy).toHaveBeenCalledWith('d1', 'm1');
  });

  it('passes an incompatible pair through (explainer handles it)', async () => {
    setRegistry(
      [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true } as DatasetRegistryEntry],
      [{ id: 'm1', name: 'Det', media_type: 'image', detector_loaded: true } as DetectorRegistryEntry],
    );
    const applySpy = spyOn(contextSwitch, 'applyActivePair').and.callFake(() => of(undefined));

    const result = await resolveGuard(runGuard('d1', 'm1'));
    expect(result).toBeTrue();
    expect(applySpy).toHaveBeenCalled();
  });
});
