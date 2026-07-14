import { TestBed } from '@angular/core/testing';
import { Observable, Subject, of } from 'rxjs';
import { LabelsetStateService } from './labelset-state.service';
import { DetectorsCrudApiService } from './detectors-crud-api.service';
import type { DetectorLabelView } from '../generated/api-client/models/detector-label-view';
import type { DetectorLabelsDetailResponse } from '../generated/api-client/models/detector-labels-detail-response';

describe('LabelsetStateService', () => {
  let service: LabelsetStateService;
  let getLabelsDetail: ReturnType<typeof vi.fn>;
  let voteLabelElement: ReturnType<typeof vi.fn>;
  let response: DetectorLabelsDetailResponse;

  function view(id: string, label: 'good' | 'bad'): DetectorLabelView {
    return { id, label } as DetectorLabelView;
  }

  beforeEach(() => {
    response = { good: [], bad: [], media_type: '' } as DetectorLabelsDetailResponse;
    getLabelsDetail = vi.fn((): Observable<DetectorLabelsDetailResponse> => of(response));
    voteLabelElement = vi.fn(() => of({})); // resolves synchronously by default
    TestBed.configureTestingModule({
      providers: [
        LabelsetStateService,
        { provide: DetectorsCrudApiService, useValue: { getLabelsDetail, voteLabelElement } },
      ],
    });
    service = TestBed.inject(LabelsetStateService);
  });

  it('starts empty', () => {
    expect(service.good).toEqual([]);
    expect(service.bad).toEqual([]);
  });

  it('setModel(name) fetches and populates good / bad / media type', () => {
    response = { good: [view('a', 'good')], bad: [view('b', 'bad')], media_type: 'audio' };
    service.setModel('m1');
    expect(getLabelsDetail).toHaveBeenCalledWith('m1');
    expect(service.good).toEqual([view('a', 'good')]);
    expect(service.bad).toEqual([view('b', 'bad')]);
  });

  it('setModel(null) clears the labelset without fetching', () => {
    response = { good: [view('a', 'good')], bad: [], media_type: 'audio' };
    service.setModel('m1');
    getLabelsDetail.mockClear();

    service.setModel(null);
    expect(service.good).toEqual([]);
    expect(service.bad).toEqual([]);
    expect(getLabelsDetail).not.toHaveBeenCalled();
  });

  it('setting the same model twice does not refetch', () => {
    service.setModel('m1');
    getLabelsDetail.mockClear();
    service.setModel('m1');
    expect(getLabelsDetail).not.toHaveBeenCalled();
  });

  it('vote is a no-op when no model is set', () => {
    service.vote('a', 'good');
    expect(voteLabelElement).not.toHaveBeenCalled();
  });

  it('re-voting an element in its current direction sends target "remove"', () => {
    response = { good: [view('a', 'good')], bad: [], media_type: 'audio' };
    service.setModel('m1');
    // Pending vote so the follow-up refresh does not overwrite optimistic state.
    voteLabelElement.mockReturnValueOnce(new Subject());

    service.vote('a', 'good');
    expect(voteLabelElement).toHaveBeenCalledWith('m1', 'a', 'remove');
    // Optimistically removed from both lists.
    expect(service.good).toEqual([]);
    expect(service.bad).toEqual([]);
  });

  it('voting the opposite direction flips the element and sends the clicked target', () => {
    response = { good: [view('a', 'good')], bad: [], media_type: 'audio' };
    service.setModel('m1');
    voteLabelElement.mockReturnValueOnce(new Subject());

    service.vote('a', 'bad');
    expect(voteLabelElement).toHaveBeenCalledWith('m1', 'a', 'bad');
    expect(service.good).toEqual([]);
    expect(service.bad).toEqual([{ id: 'a', label: 'bad' }]);
  });

  it('a successful vote triggers a refresh', () => {
    service.setModel('m1');
    getLabelsDetail.mockClear();
    response = { good: [view('a', 'good')], bad: [], media_type: 'audio' };

    service.vote('a', 'good');
    // voteLabelElement resolves synchronously (of({})) → refresh runs.
    expect(getLabelsDetail).toHaveBeenCalledWith('m1');
  });

  it('startPolling fetches on an interval; stopPolling halts it', async () => {
    vi.useFakeTimers();
    try {
      service.setModel('m1');
      getLabelsDetail.mockClear();

      service.startPolling(1500);
      await vi.advanceTimersByTimeAsync(0);
      expect(getLabelsDetail).toHaveBeenCalledTimes(1);

      await vi.advanceTimersByTimeAsync(1500);
      expect(getLabelsDetail).toHaveBeenCalledTimes(2);

      service.stopPolling();
      await vi.advanceTimersByTimeAsync(3000);
      expect(getLabelsDetail).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it('startPolling is idempotent while already polling', async () => {
    vi.useFakeTimers();
    try {
      service.setModel('m1');
      getLabelsDetail.mockClear();

      service.startPolling(1500);
      service.startPolling(1500);
      await vi.advanceTimersByTimeAsync(0);
      // A second loop would have doubled the fetches.
      expect(getLabelsDetail).toHaveBeenCalledTimes(1);
      service.stopPolling();
    } finally {
      vi.useRealTimers();
    }
  });

  it('a fetch error leaves the current labelset intact', () => {
    response = { good: [view('a', 'good')], bad: [], media_type: 'audio' };
    service.setModel('m1');
    getLabelsDetail.mockReturnValueOnce(
      new Observable<DetectorLabelsDetailResponse>((o) => o.error(new Error('500'))),
    );

    service.refresh();
    expect(service.good).toEqual([view('a', 'good')]);
  });
});
