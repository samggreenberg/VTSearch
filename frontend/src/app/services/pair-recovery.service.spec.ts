import { TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { Router } from '@angular/router';
import { Subject } from 'rxjs';

import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { DatasetStateService } from './dataset-state.service';
import { PairRecoveryService } from './pair-recovery.service';
import { ToastService } from './toast.service';

/**
 * Issue #5155: after a server restart an open Train/Tune tab named a pair the
 * new process had never loaded, so every request 409'd under a red toast and
 * nothing loaded the pair again. These pin the service that now does: when a
 * 409 is its to answer, when it is not, and what happens when the reload lands,
 * fails, or is overtaken.
 */
describe('PairRecoveryService', () => {
  let service: PairRecoveryService;
  let activeContext: ActiveContextService;
  let reloads: Subject<void>[];
  let contextSwitch: { switching: boolean; reloadActivePair: ReturnType<typeof vi.fn> };
  let router: { url: string; navigate: ReturnType<typeof vi.fn> };
  let toast: {
    info: ReturnType<typeof vi.fn>;
    success: ReturnType<typeof vi.fn>;
    error: ReturnType<typeof vi.fn>;
    dismiss: ReturnType<typeof vi.fn>;
  };

  function notLoaded(half: 'dataset' | 'detector', id: string): HttpErrorResponse {
    return new HttpErrorResponse({
      status: 409,
      statusText: 'Conflict',
      error: {
        code: 409,
        status: 'Conflict',
        message: half === 'dataset' ? 'Dataset is not loaded' : 'Detector is not loaded',
        error_code: `${half}_not_loaded`,
        [`${half}_id`]: id,
      },
    });
  }

  beforeEach(() => {
    reloads = [];
    contextSwitch = {
      switching: false,
      reloadActivePair: vi.fn(() => {
        const reload = new Subject<void>();
        reloads.push(reload);
        return reload.asObservable();
      }),
    };
    router = { url: '/label/ds1/det1', navigate: vi.fn(() => Promise.resolve(true)) };
    toast = { info: vi.fn(() => 7), success: vi.fn(), error: vi.fn(), dismiss: vi.fn() };
    TestBed.configureTestingModule({
      providers: [
        { provide: ContextSwitchService, useValue: contextSwitch },
        { provide: Router, useValue: router },
        { provide: ToastService, useValue: toast },
        {
          provide: DatasetStateService,
          useValue: {
            datasetById: () => new Map([['ds1', { id: 'ds1', name: 'o365_better' }]]),
            detectorById: () => new Map([['det1', { id: 'det1', name: 'bicycle' }]]),
          },
        },
      ],
    });
    service = TestBed.inject(PairRecoveryService);
    activeContext = TestBed.inject(ActiveContextService);
    activeContext.setActivePair('ds1', 'det1');
  });

  function landReload(): void {
    const reload = reloads[reloads.length - 1];
    reload.next();
    reload.complete();
  }

  it('loads the pair again when a 409 names it and no load is in flight', () => {
    expect(service.absorb(notLoaded('detector', 'det1'), 0)).toBe(true);

    expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);
    expect(toast.info).toHaveBeenCalledTimes(1);
    expect(toast.info.mock.calls[0][0].message).toContain('o365_better · bicycle');
  });

  it('answers a dataset_not_loaded 409 the same way', () => {
    expect(service.absorb(notLoaded('dataset', 'ds1'), 0)).toBe(true);
    expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);
  });

  it('absorbs the 409s that land while its reload is in flight, without starting another', () => {
    service.absorb(notLoaded('detector', 'det1'), 0);

    expect(service.absorb(notLoaded('dataset', 'ds1'), 0)).toBe(true);
    expect(service.absorb(notLoaded('detector', 'det1'), 0)).toBe(true);
    expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);
  });

  it('fires reloaded$ and says so once the reload lands', () => {
    let fired = 0;
    service.reloaded$.subscribe(() => fired++);
    service.absorb(notLoaded('detector', 'det1'), 0);
    expect(fired).toBe(0);

    landReload();

    expect(fired).toBe(1);
    expect(service.epoch).toBe(1);
    expect(toast.success).toHaveBeenCalledTimes(1);
    // It replaces the "loading again" toast rather than stacking under it.
    expect(toast.success.mock.calls[0][0].dedupKey).toBe(toast.info.mock.calls[0][0].dedupKey);
  });

  it('reads a 409 sent before the reload landed as already answered', () => {
    service.absorb(notLoaded('detector', 'det1'), 0);
    landReload();

    // Sent at epoch 0, i.e. while the reload was still loading.
    expect(service.absorb(notLoaded('detector', 'det1'), 0)).toBe(true);
    expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);
    expect(router.navigate).not.toHaveBeenCalled();
  });

  it('absorbs a 409 that lands while a switch is loading the pair (#3644), without reloading', () => {
    contextSwitch.switching = true;

    expect(service.absorb(notLoaded('detector', 'det1'), 0)).toBe(true);
    expect(contextSwitch.reloadActivePair).not.toHaveBeenCalled();
    expect(toast.info).not.toHaveBeenCalled();
  });

  it('leaves a 409 to the ordinary toast when no view is showing the pair', () => {
    router.url = '/dashboard';
    expect(service.absorb(notLoaded('detector', 'det1'), 0)).toBe(false);

    router.url = '/label/ds1/other-det';
    expect(service.absorb(notLoaded('detector', 'det1'), 0)).toBe(false);

    expect(contextSwitch.reloadActivePair).not.toHaveBeenCalled();
  });

  it('answers on the Tune view too, whatever the query string', () => {
    router.url = '/tune/ds1/det1?panel=right';
    expect(service.absorb(notLoaded('detector', 'det1'), 0)).toBe(true);
    expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);
  });

  // #5158: Browse shows a dataset alone, with the active detector riding along,
  // so its URL names only the dataset.
  describe('on the Browse view', () => {
    it('loads the pair again for a 409 naming its dataset, whatever the query string', () => {
      router.url = '/browse/ds1?subset=1';
      expect(service.absorb(notLoaded('dataset', 'ds1'), 0)).toBe(true);
      expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);
    });

    it('loads the pair again for a 409 naming the detector riding along', () => {
      router.url = '/browse/ds1';
      expect(service.absorb(notLoaded('detector', 'det1'), 0)).toBe(true);
      expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);
    });

    it('names the dataset alone, as the view does', () => {
      router.url = '/browse/ds1';
      service.absorb(notLoaded('dataset', 'ds1'), 0);
      landReload();

      expect(toast.info.mock.calls[0][0].message).toBe('The server had lost o365_better; loading it again');
      expect(toast.success.mock.calls[0][0].message).toBe('o365_better is loaded again');
    });

    it('loads a dataset browsed with no detector active', () => {
      router.url = '/browse/ds1';
      activeContext.setActivePair('ds1', '');

      expect(service.absorb(notLoaded('dataset', 'ds1'), 0)).toBe(true);
      expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);
      expect(toast.info.mock.calls[0][0].message).toContain('lost o365_better;');
    });

    it('leaves a 409 to the ordinary toast when the tab browses some other dataset', () => {
      router.url = '/browse/other-ds';
      expect(service.absorb(notLoaded('dataset', 'ds1'), 0)).toBe(false);
      expect(contextSwitch.reloadActivePair).not.toHaveBeenCalled();
    });

    it('sends the tab to the Dashboard when the reload fails', () => {
      router.url = '/browse/ds1';
      service.absorb(notLoaded('dataset', 'ds1'), 0);
      reloads[0].complete();

      expect(toast.error.mock.calls[0][0].message).toBe("Couldn't load o365_better again");
      expect(router.navigate).toHaveBeenCalledWith(['/dashboard']);
    });
  });

  it('ignores a 409 naming an id outside the active pair', () => {
    // E.g. a `DATASET_OVERRIDE` read of a sibling dataset.
    expect(service.absorb(notLoaded('dataset', 'sibling-ds'), 0)).toBe(false);
    expect(contextSwitch.reloadActivePair).not.toHaveBeenCalled();
  });

  it('ignores every other error', () => {
    const otherConflict = new HttpErrorResponse({
      status: 409,
      error: { error_code: 'job_conflict', detector_id: 'det1' },
    });
    const serverError = new HttpErrorResponse({
      status: 500,
      error: { error_code: 'detector_not_loaded', detector_id: 'det1' },
    });
    const textBody = new HttpErrorResponse({ status: 409, error: '<html>409</html>' });

    expect(service.absorb(otherConflict, 0)).toBe(false);
    expect(service.absorb(serverError, 0)).toBe(false);
    expect(service.absorb(textBody, 0)).toBe(false);
    expect(contextSwitch.reloadActivePair).not.toHaveBeenCalled();
  });

  it('sends the tab to the Dashboard, and clears the pair, when the reload fails', async () => {
    service.absorb(notLoaded('detector', 'det1'), 0);

    // A failed load completes the switch without emitting.
    reloads[0].complete();

    expect(toast.error).toHaveBeenCalledTimes(1);
    expect(toast.error.mock.calls[0][0].detail).toContain('Reopen it from the Dashboard');
    expect(router.navigate).toHaveBeenCalledWith(['/dashboard']);
    // Absorbed quietly while the navigation is under way.
    expect(service.absorb(notLoaded('detector', 'det1'), 0)).toBe(true);
    expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);

    await Promise.resolve();
    await Promise.resolve();
    // Cleared so reopening it runs the route guard's load.
    expect(activeContext.datasetId).toBe('');
    expect(activeContext.modelId).toBe('');
  });

  it('steps aside quietly when a switch overtakes the reload', () => {
    service.absorb(notLoaded('detector', 'det1'), 0);

    contextSwitch.switching = true;
    reloads[0].complete();

    expect(toast.dismiss).toHaveBeenCalledWith(7);
    expect(toast.error).not.toHaveBeenCalled();
    expect(router.navigate).not.toHaveBeenCalled();
  });

  it('gives up rather than reloading again when the pair is lost straight after a reload', () => {
    service.absorb(notLoaded('detector', 'det1'), 0);
    landReload();

    // Sent after the reload landed, and the backend still has not got it.
    expect(service.absorb(notLoaded('detector', 'det1'), 1)).toBe(true);

    expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(1);
    expect(toast.error).toHaveBeenCalledTimes(1);
    expect(router.navigate).toHaveBeenCalledWith(['/dashboard']);
  });

  it('loads the pair again for a loss well after the last reload', () => {
    const now = vi.spyOn(Date, 'now').mockReturnValue(1_000_000);
    service.absorb(notLoaded('detector', 'det1'), 0);
    landReload();

    now.mockReturnValue(1_000_000 + 10 * 60_000);
    expect(service.absorb(notLoaded('detector', 'det1'), 1)).toBe(true);

    expect(contextSwitch.reloadActivePair).toHaveBeenCalledTimes(2);
    expect(router.navigate).not.toHaveBeenCalled();
    now.mockRestore();
  });
});
