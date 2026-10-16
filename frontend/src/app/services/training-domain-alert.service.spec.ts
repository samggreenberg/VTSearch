import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';
import { Observable, of, Subject, throwError } from 'rxjs';

import type { FindTrainingDomainResponse } from '../generated/api-client/models/find-training-domain-response';
import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';
import { DetectorsFindApiService } from './detectors-find-api.service';
import { ToastService } from './toast.service';
import { TrainingDomainAlertService } from './training-domain-alert.service';

function report(over: Partial<FindTrainingDomainResponse> & { shifted?: boolean; frac?: number } = {}): FindTrainingDomainResponse {
  const { shifted = true, frac = 0.31, ...rest } = over;
  return {
    reference_dataset_id: 'photos',
    reference_dataset_name: 'Photos',
    labels_total: 52,
    labels_in_reference: 40,
    labels_in_active: 0,
    reason: '',
    domain_shift: {
      reference_dataset_id: 'photos',
      n_items: 200,
      alpha: 0.05,
      frac_atypical: frac,
      expected_atypical: 0.05,
      z_score: 9,
      median_pvalue: 0.2,
      shifted,
    },
    ...rest,
  };
}

const NOTHING_TO_COMPARE = report({
  domain_shift: null as unknown as FindTrainingDomainResponse['domain_shift'],
  reason: 'Reference dataset has no coverage atlas; build it first',
});

describe('TrainingDomainAlertService (#4961)', () => {
  let toast: { warning: ReturnType<typeof vi.fn> };
  let navigate: ReturnType<typeof vi.fn>;
  let url: string;
  let respond: () => Observable<FindTrainingDomainResponse>;
  let calls: number;
  let context: ActiveContextService;

  function start(): TrainingDomainAlertService {
    toast = { warning: vi.fn() };
    navigate = vi.fn(() => Promise.resolve(true));
    url = '/dashboard';
    calls = 0;
    TestBed.configureTestingModule({
      providers: [
        TrainingDomainAlertService,
        ActiveContextService,
        { provide: ToastService, useValue: toast },
        {
          provide: Router,
          useValue: {
            navigate,
            get url() {
              return url;
            },
          },
        },
        {
          provide: DetectorsFindApiService,
          useValue: {
            getTrainingDomain: () => {
              calls++;
              return respond();
            },
          },
        },
        {
          provide: DatasetStateService,
          useValue: {
            detectorById: signal(new Map([['cats', { id: 'cats', name: 'Cats' }]])),
            datasetById: signal(new Map([['screens', { id: 'screens', name: 'Screenshots' }]])),
          },
        },
      ],
    });
    context = TestBed.inject(ActiveContextService);
    const service = TestBed.inject(TrainingDomainAlertService);
    service.start();
    return service;
  }

  afterEach(() => TestBed.resetTestingModule());

  it('warns when the detector meets a dataset unlike the one it learned from', () => {
    respond = () => of(report());
    start();
    context.setActivePair('screens', 'cats');

    expect(toast.warning).toHaveBeenCalledTimes(1);
    const opts = toast.warning.mock.calls[0][0];
    // The Test result's line and verdict, nothing stronger.
    expect(opts.message).toBe('31% of this dataset looks atypical vs Photos: likely domain shift');
    expect(opts.detail).toContain('Cats learned from Photos (40 of its 52 labelled examples are there)');
    expect(opts.detail).toContain('Check its calls on Screenshots by hand');
    expect(opts.dedupKey).toBe('training-domain:screens::cats');

    opts.action.onClick();
    expect(navigate).toHaveBeenCalledWith(['/test', 'screens', 'cats']);
  });

  it('says nothing for a dataset like the training one', () => {
    respond = () => of(report({ shifted: false, frac: 0.03 }));
    start();
    context.setActivePair('screens', 'cats');
    expect(calls).toBe(1);
    expect(toast.warning).not.toHaveBeenCalled();
  });

  it('says nothing when there is nothing to compare, or the check fails', () => {
    respond = () => of(NOTHING_TO_COMPARE);
    start();
    context.setActivePair('screens', 'cats');
    respond = () => throwError(() => new Error('offline'));
    context.setActivePair('other', 'cats');
    expect(calls).toBe(2);
    expect(toast.warning).not.toHaveBeenCalled();
  });

  it('waits for both halves of the pair', () => {
    respond = () => of(report());
    start();
    context.setActivePair('screens', '');
    context.setActivePair('', 'cats');
    expect(calls).toBe(0);
  });

  it('warns once per pair per session', () => {
    respond = () => of(report());
    start();
    context.setActivePair('screens', 'cats');
    respond = () => of(report({ shifted: false }));
    context.setActivePair('more-photos', 'cats');
    respond = () => of(report());
    context.setActivePair('screens', 'cats');
    expect(toast.warning).toHaveBeenCalledTimes(1);
    // An answered pair is not asked again.
    expect(calls).toBe(2);
  });

  it('asks again about a pair whose check could not run', () => {
    respond = () => of(NOTHING_TO_COMPARE);
    start();
    context.setActivePair('screens', 'cats');
    context.setActivePair('photos', 'cats');
    respond = () => of(report());
    context.setActivePair('screens', 'cats');
    expect(calls).toBe(3);
    expect(toast.warning).toHaveBeenCalledTimes(1);
  });

  it('drops the answer for a pair the user has already left', () => {
    const pending = new Subject<FindTrainingDomainResponse>();
    respond = () => pending;
    start();
    context.setActivePair('screens', 'cats');
    respond = () => of(report({ shifted: false }));
    context.setActivePair('photos', 'cats');
    pending.next(report());
    expect(toast.warning).not.toHaveBeenCalled();
  });

  it('offers no Test button when the user is already testing', () => {
    respond = () => of(report());
    start();
    url = '/test/screens/cats';
    context.setActivePair('screens', 'cats');
    expect(toast.warning.mock.calls[0][0].action).toBeUndefined();
  });
});
