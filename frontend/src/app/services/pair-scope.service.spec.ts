import { Component, inject } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { Subject } from 'rxjs';

import { MediaStateService } from './media-state.service';
import { PairScopeService } from './pair-scope.service';
import { SortStateService } from './sort-state.service';
import { VoteStateService } from './vote-state.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import { provideHttpTesting } from '../testing/test-providers';

@Component({ selector: 'vt-pair-scope-host', standalone: true, template: '', providers: [PairScopeService] })
class HostComponent {
  readonly pairScope = inject(PairScopeService);
}

/**
 * `PairScopeService` exists to make the pair-change ordering rule *enforced*
 * rather than described in a comment in each view (#3448). These specs pin the
 * three claims its doc comment makes, so a future edit that reorders the reset
 * fails here instead of shipping a silent stale-results bug.
 */
describe('PairScopeService', () => {
  let fixture: ComponentFixture<HostComponent>;
  let service: PairScopeService;
  let httpMock: HttpTestingController;
  let sortState: SortStateService;
  let voteState: VoteStateService;
  let mediaState: MediaStateService;

  beforeEach(() => {
    configureZoneless({ imports: [HostComponent], providers: [...provideHttpTesting()] });
    fixture = TestBed.createComponent(HostComponent);
    service = fixture.componentInstance.pairScope;
    httpMock = TestBed.inject(HttpTestingController);
    sortState = TestBed.inject(SortStateService);
    voteState = TestBed.inject(VoteStateService);
    mediaState = TestBed.inject(MediaStateService);
  });

  afterEach(() => {
    // The reset fires GETs the individual specs do not always drain.
    httpMock.match(() => true).forEach((req) => req.flush({}));
    httpMock.verify();
  });

  it('scoped() completes its stream when the pair changes', () => {
    const src = new Subject<number>();
    const seen: number[] = [];
    let completed = false;
    src.pipe(service.scoped()).subscribe({
      next: (v) => seen.push(v),
      complete: () => { completed = true; },
    });

    src.next(1);
    service.resetForNewPair();
    src.next(2);

    expect(seen).toEqual([1]);
    expect(completed).toBe(true);
  });

  it('supersedes before it installs any of the new pair state', () => {
    // A pair-scoped request in flight for the pair we are about to leave.
    service.loadDatasetName();
    const stale = httpMock.expectOne('/api/dataset/status');
    expect(stale.cancelled).toBe(false);

    // The order under test: if `resetForNewPair` cleared or reloaded *before*
    // firing the scope, this response would still be live and would write the
    // old pair's display name over the new pair's.
    service.resetForNewPair();

    expect(stale.cancelled).toBe(true);
    // The XHR is aborted, so there is no longer any way for it to land — the
    // testing controller refuses to flush a cancelled request at all.
    expect(() => stale.flush({ display_name: 'OLD PAIR' })).toThrow();
    expect(service.datasetName()).toBe('');
  });

  it('runs the quiesce hook after the supersede and before the reloads', () => {
    const order: string[] = [];

    service.loadDatasetName();
    const stale = httpMock.expectOne('/api/dataset/status');

    service.resetForNewPair(() => {
      // Already superseded: the request issued for the old pair is dead.
      order.push(stale.cancelled ? 'after-supersede' : 'before-supersede');
      // Not yet reloaded: the new pair's requests have not gone out.
      order.push(httpMock.match('/api/dataset/status').length === 0 ? 'before-reload' : 'after-reload');
    });

    expect(order).toEqual(['after-supersede', 'before-reload']);
  });

  it('clears the pair-scoped sort and vote state, then reloads for the new pair', () => {
    sortState.setSortResults([{ id: 1, score: 0.9 }], 0.5);
    sortState.setSortStatus('scoring');
    sortState.setSortProgress(3, 10);
    const clearSpy = vi.spyOn(voteState, 'clear');

    service.resetForNewPair();

    expect(sortState.sortOrder).toEqual([]);
    expect(sortState.sortStatus).toBe('');
    expect(sortState.sortProgress).toBe(0);
    expect(clearSpy).toHaveBeenCalledOnce();
    // Reloads for the new pair went out.
    expect(httpMock.match('/api/dataset/status').length).toBe(1);
    expect(httpMock.match('/api/inclusion').length).toBe(1);
  });

  it('clears the selection too, so the centre viewer cannot outlive the pair', () => {
    // Media ids are per-dataset, and `ActiveContextService.mediaUrl` stamps the
    // dataset id into the `<img src>` at build time — so a selection that
    // survives the switch keeps resolving against the pair we *left* and
    // renders happily instead of 404-ing (#3489). Nothing else in the reset
    // touches it: the ranking and the vote cache are cleared, everything
    // derived from the selection goes with them, and the selection itself is
    // the one piece of pair-scoped state left behind.
    mediaState.selectMedia(7);
    expect(mediaState.selectedId()).toBe(7);

    service.resetForNewPair();

    expect(mediaState.selectedId()).toBeNull();
  });

  it('clearPairState alone drops the selection, for find-view\'s entry path', () => {
    // Dashboard -> Find enters against a possibly different dataset while the
    // singleton `MediaStateService` still holds the previous session's pick.
    // That entry path calls `clearPairState()` rather than the full reset, so
    // the selection clear has to live there and not in `resetForNewPair`.
    mediaState.selectMedia(7);

    service.clearPairState();

    expect(mediaState.selectedId()).toBeNull();
  });

  it('seedInclusion pushes the per-detector value into SortStateService', () => {
    service.seedInclusion();
    httpMock.expectOne('/api/inclusion').flush({ inclusion: -4, threshold: 0.3 });
    expect(sortState.inclusion).toBe(-4);
  });

  it('fires the scope when the host component is destroyed, with no manual teardown', () => {
    service.loadDatasetName();
    const inFlight = httpMock.expectOne('/api/dataset/status');

    // No view calls `scope$.next()` in its own `ngOnDestroy` any more; Angular
    // destroys the component-provided service, which is what tears this down.
    fixture.destroy();

    expect(inFlight.cancelled).toBe(true);
  });
});
