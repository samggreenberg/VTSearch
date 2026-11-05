import { Component, inject } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse } from '@angular/common/http';
import { Router, provideRouter } from '@angular/router';
import { Observable, Subject, of, throwError } from 'rxjs';

import type { PeopleList } from '../generated/api-client/models/people-list';
import { LoadingTask } from '../models/api.models';
import { configureZoneless } from '../testing/zoneless-testbed';
import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { PeopleApiService } from './people-api.service';
import { PeoplePrepService } from './people-prep.service';
import { ProgressEventsService } from './progress-events.service';

/** The Dashboard row as it reads the prep: a method call in the template, no bridge. */
@Component({
  selector: 'app-people-prep-row',
  standalone: true,
  template: `<span class="msg">{{ prep.displayTask('faces')?.message }}</span>`,
})
class PeoplePrepRowComponent {
  readonly prep = inject(PeoplePrepService);
}

/**
 * The Dashboard's People item loads the Face dataset and finds its people,
 * with progress on its row, before it opens `/people/:id` (#5590): a dataset
 * that won't load or whose people can't be found says so on the Dashboard.
 */
describe('PeoplePrepService', () => {
  const DS = 'faces';
  let service: PeoplePrepService;
  let router: Router;
  let applyPair$: Subject<void>;
  let loadingTasks: LoadingTask[];
  let listProvider: () => Observable<PeopleList>;
  let listCalls: number;

  beforeEach(() => {
    applyPair$ = new Subject<void>();
    loadingTasks = [];
    listCalls = 0;
    listProvider = () => of({ status: 'ready', partition: 'p1', people: [] });

    configureZoneless({
      providers: [
        provideRouter([]),
        { provide: ContextSwitchService, useValue: { applyActivePair: (): Observable<void> => applyPair$ } },
        { provide: ActiveContextService, useValue: { modelId: '' } },
        { provide: ProgressEventsService, useValue: { loadingTasks: (): LoadingTask[] => loadingTasks } },
        {
          provide: PeopleApiService,
          useValue: {
            list: (): Observable<PeopleList> => {
              listCalls++;
              return listProvider();
            },
          },
        },
      ],
    });
    service = TestBed.inject(PeoplePrepService);
    router = TestBed.inject(Router);
    vi.spyOn(router, 'navigate').mockResolvedValue(true);
  });

  it('opens People once the dataset is loaded and its people are found', () => {
    service.prepareAndOpen(DS);
    expect(service.preparing).toBe(true);
    expect(router.navigate).not.toHaveBeenCalled();

    applyPair$.next();

    expect(router.navigate).toHaveBeenCalledWith(['/people', DS]);
    expect(service.preparing).toBe(false);
  });

  it('polls a fit still running, with its progress on the row, then opens People', async () => {
    vi.useFakeTimers();
    try {
      listProvider = () => of({ status: 'building', message: 'Finding people among 40 faces…', current: 2, total: 10 });
      service.prepareAndOpen(DS);
      applyPair$.next();

      expect(service.taskKind(DS)).toBe('people');
      const row = service.displayTask(DS);
      expect(row?.status).toBe('building');
      expect(row?.current).toBe(2);
      expect(service.ownsTask(row!.task_id)).toBe(true);
      expect(router.navigate).not.toHaveBeenCalled();

      listProvider = () => of({ status: 'ready', partition: 'p1', people: [] });
      await vi.advanceTimersByTimeAsync(1000);

      expect(listCalls).toBe(2);
      expect(router.navigate).toHaveBeenCalledWith(['/people', DS]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('opens People for an uncalibrated face space, which the view explains', () => {
    listProvider = () => of({ status: 'not_calibrated', compat: 'ext:facenet', message: 'not calibrated' });
    service.prepareAndOpen(DS);
    applyPair$.next();
    expect(router.navigate).toHaveBeenCalledWith(['/people', DS]);
  });

  it('shows a failed fit on the row, and does not navigate', () => {
    listProvider = () => of({ status: 'error', error: 'the fit fell over' });
    service.prepareAndOpen(DS);
    applyPair$.next();

    expect(service.displayTask(DS)?.error).toBe('the fit fell over');
    expect(router.navigate).not.toHaveBeenCalled();
    expect(service.preparing).toBe(false);
    service.dismiss();
    expect(service.displayTask(DS)).toBeNull();
  });

  it("shows the server's reason for a dataset with no people", () => {
    listProvider = () =>
      throwError(
        () => new HttpErrorResponse({ status: 409, error: { message: 'Not a Face dataset.', error_code: 'no_people' } }),
      );
    service.prepareAndOpen(DS);
    applyPair$.next();
    expect(service.displayTask(DS)?.error).toBe('Not a Face dataset.');
  });

  it('bails out silently when the load failed (its own row shows the error)', () => {
    loadingTasks = [
      { status: 'idle', message: '', current: 0, total: 0, task_id: 't', name: '', created_at: 0, dataset_id: DS, error: 'boom' },
    ];
    service.prepareAndOpen(DS);
    applyPair$.next();
    expect(listCalls).toBe(0);
    expect(service.displayTask(DS)).toBeNull();
  });

  it('defers the load-phase row to the real SSE task, and is not started twice', () => {
    service.prepareAndOpen(DS);
    expect(service.displayTask(DS)?.status).toBe('loading');
    expect(service.taskKind(DS)).toBe('dataset');
    expect(service.holds(DS)).toBe(true);
    expect(service.holds('other')).toBe(false);

    loadingTasks = [
      { status: 'loading', message: '', current: 0, total: 0, task_id: 't', name: '', created_at: 0, dataset_id: DS, error: '' },
    ];
    expect(service.displayTask(DS)).toBeNull();

    service.prepareAndOpen('other');
    expect(service.holds('other')).toBe(false);
  });

  it('clears on cancel', () => {
    listProvider = () => of({ status: 'building' });
    service.prepareAndOpen(DS);
    applyPair$.next();
    service.cancel();
    expect(service.preparing).toBe(false);
    expect(service.displayTask(DS)).toBeNull();
  });
});

describe('PeoplePrepService: zoneless repaint of the row', () => {
  let listProvider: () => Observable<PeopleList>;
  let applyPair$: Subject<void>;

  async function waitForMessage(fixture: ComponentFixture<PeoplePrepRowComponent>, expected: string): Promise<void> {
    const deadline = Date.now() + 4000;
    while (Date.now() < deadline) {
      await fixture.whenStable();
      const text = fixture.nativeElement.querySelector('.msg')?.textContent?.trim() ?? '';
      if (text === expected) return;
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(`row never showed "${expected}"`);
  }

  it('repaints the row from each poll, with no other signal moving', async () => {
    applyPair$ = new Subject<void>();
    listProvider = () => of({ status: 'building', message: 'Finding people among 4 faces…' });
    configureZoneless({
      imports: [PeoplePrepRowComponent],
      providers: [
        provideRouter([]),
        { provide: ContextSwitchService, useValue: { applyActivePair: (): Observable<void> => applyPair$ } },
        { provide: ActiveContextService, useValue: { modelId: '' } },
        { provide: ProgressEventsService, useValue: { loadingTasks: (): LoadingTask[] => [] } },
        { provide: PeopleApiService, useValue: { list: (): Observable<PeopleList> => listProvider() } },
      ],
    });
    vi.spyOn(TestBed.inject(Router), 'navigate').mockResolvedValue(true);
    const fixture = TestBed.createComponent(PeoplePrepRowComponent);
    const prep = TestBed.inject(PeoplePrepService);

    prep.prepareAndOpen('faces');
    applyPair$.next();
    await waitForMessage(fixture, 'Finding people among 4 faces…');

    listProvider = () => of({ status: 'building', message: 'Finding people among 5 faces…' });
    await waitForMessage(fixture, 'Finding people among 5 faces…');
    prep.cancel();
  });
});
