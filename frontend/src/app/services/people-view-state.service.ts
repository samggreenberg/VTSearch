import { HttpErrorResponse } from '@angular/common/http';
import { DestroyRef, Injectable, computed, effect, inject, signal, untracked } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import { Observable, catchError, map, of } from 'rxjs';
import { take } from 'rxjs/operators';

import type { CoPerson } from '../generated/api-client/models/co-person';
import type { PeopleList } from '../generated/api-client/models/people-list';
import type { Person } from '../generated/api-client/models/person';
import type { PersonSummary } from '../generated/api-client/models/person-summary';
import type { SharedPhoto } from '../generated/api-client/models/shared-photo';
import { apiErrorMessage } from '../utils/api-error';
import { PeopleApiService } from './people-api.service';
import { pollUntil, type PollHandle } from './poll-until';
import { ToastService } from './toast.service';

/** Where the People view's list stands. */
export type PeopleStatus = 'loading' | 'building' | 'ready' | 'not_calibrated' | 'error';

/** What the people list has picked: a person by rank, or the unplaced faces. */
export type PeoplePick = number | 'unplaced';

/** The faces a minimum-count filter starts at, folding away the one-face tail. */
export const DEFAULT_MIN_FACES = 2;

/** The answer to a per-person read: found, or the partition moved under the id. */
type PersonRead = { kind: 'found'; person: Person } | { kind: 'changed' } | { kind: 'error'; message: string };

/** The answer to a shared-photos read. `not-loaded` names the Image dataset to load. */
export type SharedRead =
  | { kind: 'found'; photos: SharedPhoto[] }
  | { kind: 'not-loaded'; datasetId: string }
  | { kind: 'changed' }
  | { kind: 'error'; message: string };

/**
 * The People view's state (#5590): the people list, which person is picked,
 * the face shown large, and who they appear with. Provided on
 * `vt-people-view`, as Browse's selection and viewport services are on
 * `vt-browse-view`, so it lives exactly as long as the view.
 *
 * **The list is imperative, the reads are resources.** `GET /api/people`
 * answers `building` while the people are found, and the view polls it
 * (`pollUntil`) to `ready`, as Browse polls its map; Rebuild people is a POST
 * that answers the same way. The person and the photos two people share are
 * `rxResource`s keyed by the pick *and the partition*, so a new list re-reads
 * both on its own.
 *
 * **A person id is a rank in one partition.** Every per-person read sends the
 * partition back, and the server answers 409 `people_changed` when the people
 * were found again since (a Rebuild elsewhere, or faces removed). Either read
 * answering that reads the list again and picks the same person anew by their
 * representative face, or no one if that face now stands for nobody: an id is
 * never carried into a partition it was not read from.
 */
@Injectable()
export class PeopleViewStateService {
  private readonly api = inject(PeopleApiService);
  private readonly toast = inject(ToastService);
  private readonly destroyRef = inject(DestroyRef);

  private readonly list = signal<PeopleList | null>(null);
  private readonly _status = signal<PeopleStatus>('loading');
  private readonly _errorMessage = signal('');
  private poll: PollHandle | null = null;
  /** Bumped by every list read, so a reply to an older read is dropped. */
  private listSeq = 0;
  /** The representative of the person picked when the list went stale, to
   *  pick them again in the list read afresh. */
  private repickBy: number | null = null;

  readonly status = this._status.asReadonly();
  /** Why the list could not be read, or the fit failed. */
  readonly errorMessage = this._errorMessage.asReadonly();

  /** The fit's progress while `building`, for the bar. */
  readonly progress = computed(() => (this._status() === 'building' ? this.list() : null));
  /** The uncalibrated face space's wording, when `not_calibrated`. */
  readonly notCalibratedMessage = computed(() =>
    this._status() === 'not_calibrated' ? this.list()?.message || '' : '',
  );

  private readonly ready = computed(() => {
    const l = this.list();
    return l?.status === 'ready' ? l : null;
  });
  /** Every person, the most faces first. */
  readonly people = computed<PersonSummary[]>(() => this.ready()?.people ?? []);
  /** The identity of the partition the people were read from. */
  readonly partition = computed(() => this.ready()?.partition ?? null);
  /** The faces in no person, ascending. */
  readonly unplacedIds = computed<number[]>(() => this.ready()?.unplaced_ids ?? []);

  /** People with fewer faces than this are folded away. */
  readonly minFaces = signal(DEFAULT_MIN_FACES);
  /** The people the filter keeps. */
  readonly shownPeople = computed(() => {
    const min = this.minFaces();
    return this.people().filter((p) => p.face_count >= min);
  });
  /** How many people the filter folds away. */
  readonly foldedCount = computed(() => this.people().length - this.shownPeople().length);

  private readonly _pick = signal<PeoplePick | null>(null);
  /** The person (or the unplaced faces) the centre shows. */
  readonly pick = this._pick.asReadonly();
  /** The picked person's summary row, or null. */
  readonly pickedSummary = computed(() => {
    const pick = this._pick();
    return typeof pick === 'number' ? (this.people().find((p) => p.id === pick) ?? null) : null;
  });

  private readonly personResource = rxResource({
    params: () => {
      const pick = this._pick();
      const partition = this.partition();
      return typeof pick === 'number' && partition ? { id: pick, partition } : undefined;
    },
    stream: ({ params }) => this.readPerson(this.api.person(params.id, params.partition)),
  });

  /** The picked person, once read; null for the unplaced faces or while reading. */
  readonly person = computed(() => {
    const read = this.personResource.value();
    return read?.kind === 'found' ? read.person : null;
  });
  /** Reading the picked person. */
  readonly personLoading = computed(() => this.personResource.isLoading());
  /** Why the picked person could not be read. */
  readonly personError = computed(() => {
    const read = this.personResource.value();
    return read?.kind === 'error' ? read.message : '';
  });

  /** The faces the centre grid shows: the person's (representative first), or the unplaced. */
  readonly faceIds = computed<number[]>(() => {
    if (this._pick() === 'unplaced') return this.unplacedIds();
    return this.person()?.face_ids ?? [];
  });

  private readonly _focusedFace = signal<number | null>(null);
  /** The face shown large: the one clicked, else the first in the grid. */
  readonly focusedFace = computed(() => {
    const ids = this.faceIds();
    const focused = this._focusedFace();
    return focused != null && ids.includes(focused) ? focused : (ids[0] ?? null);
  });

  /** The people who share a photo with the picked person, most shared first. */
  readonly appearsWith = computed<CoPerson[]>(() => this.person()?.appears_with ?? []);

  private readonly _coPick = signal<number | null>(null);
  /** The co-person whose shared photos are shown: the one clicked, else the first. */
  readonly coPick = computed(() => {
    const co = this.appearsWith();
    const picked = this._coPick();
    return picked != null && co.some((c) => c.id === picked) ? picked : (co[0]?.id ?? null);
  });

  private readonly sharedResource = rxResource({
    params: () => {
      const id = this.person()?.id;
      const other = this.coPick();
      const partition = this.person()?.partition;
      return id != null && other != null && partition ? { id, other, partition } : undefined;
    },
    stream: ({ params }) => this.readShared(this.api.shared(params.id, params.other, params.partition)),
  });

  /** The photos the picked person shares with {@link coPick}, or why not. */
  readonly shared = computed<SharedRead | null>(() => this.sharedResource.value() ?? null);
  readonly sharedLoading = computed(() => this.sharedResource.isLoading());

  constructor() {
    this.destroyRef.onDestroy(() => this.stopPoll());

    // A read that met a moved partition reads the list again (see the class
    // note). Untracked, so the list write does not re-run this effect.
    effect(() => {
      const stale = this.personResource.value()?.kind === 'changed' || this.sharedResource.value()?.kind === 'changed';
      if (stale) untracked(() => this.onStale());
    });

    // Pick the first person once there are people and no pick, so the centre
    // opens on someone rather than on a prompt.
    effect(() => {
      const people = this.shownPeople();
      if (this._status() !== 'ready' || untracked(this._pick) !== null) return;
      const first = people[0] ?? this.people()[0];
      if (first) untracked(() => this._pick.set(first.id));
    });
  }

  // --- the list ---

  /** Read the list from the start: the view opening, a reload of the dataset. */
  open(): void {
    this.stopPoll();
    this.list.set(null);
    this._pick.set(null);
    this._focusedFace.set(null);
    this._coPick.set(null);
    this._status.set('loading');
    this.readList(this.api.list());
  }

  /** Rebuild people: fit the people again, and poll to the new list. */
  rebuild(): void {
    this.repickBy = this.pickedSummary()?.representative ?? null;
    this.stopPoll();
    this._status.set('building');
    this.readList(this.api.rebuild());
  }

  /** Read the list again after a 409 `people_changed`, keeping the pick if it survives. */
  private onStale(): void {
    if (this._status() === 'building' || this._status() === 'loading') return;
    this.repickBy = this.pickedSummary()?.representative ?? null;
    this.toast.info({
      message: 'The people were found again since this list was read, so it was read again.',
      dedupKey: 'people-view:changed',
    });
    this.stopPoll();
    this.readList(this.api.list());
  }

  private readList(request: Observable<PeopleList>): void {
    const seq = ++this.listSeq;
    request.pipe(take(1)).subscribe({
      next: (list) => {
        if (seq === this.listSeq && this.applyList(list) === 'continue') this.startPoll();
      },
      error: (err: unknown) => {
        if (seq === this.listSeq) this.fail(errorText(err, "Couldn't read the people."));
      },
    });
  }

  /** Install one list answer; say whether the fit is still running. */
  private applyList(list: PeopleList): 'continue' | 'stop' {
    switch (list.status) {
      case 'building':
        this.list.set(list);
        this._status.set('building');
        return 'continue';
      case 'error':
        this.fail(list.error || 'Finding the people failed.');
        return 'stop';
      case 'not_calibrated':
        this.list.set(list);
        this._pick.set(null);
        this._status.set('not_calibrated');
        return 'stop';
      default:
        this.install(list);
        return 'stop';
    }
  }

  /** A ready list: keep the pick when it still names the same person. */
  private install(list: PeopleList): void {
    const before = this.list();
    const samePartition = before?.status === 'ready' && before.partition === list.partition;
    this.list.set(list);
    this._status.set('ready');
    this._errorMessage.set('');
    const pick = this._pick();
    if (samePartition || pick === 'unplaced') {
      this.repickBy = null;
      return;
    }
    // A new partition: the old rank names no one in it. Pick the person whose
    // representative face is the one the picked person had, or no one.
    const by = this.repickBy ?? null;
    this.repickBy = null;
    this._focusedFace.set(null);
    this._coPick.set(null);
    const again = by == null ? undefined : list.people?.find((p) => p.representative === by);
    this._pick.set(again ? again.id : null);
  }

  private startPoll(): void {
    this.stopPoll();
    const seq = this.listSeq;
    this.poll = pollUntil<PeopleList>({
      fetch: () => this.api.list(),
      apply: (list) => (seq === this.listSeq ? this.applyList(list) : 'stop'),
      onLostContact: () => this.fail('Lost contact with the server while finding the people.'),
    });
  }

  private stopPoll(): void {
    this.poll?.stop();
    this.poll = null;
  }

  private fail(message: string): void {
    this.stopPoll();
    this._errorMessage.set(message);
    this._status.set('error');
  }

  // --- picks ---

  /** Show *pick* in the centre (a person, or the unplaced faces). */
  choose(pick: PeoplePick): void {
    if (pick === this._pick()) return;
    this._pick.set(pick);
    this._focusedFace.set(null);
    this._coPick.set(null);
  }

  /** Show *faceId* large. */
  focusFace(faceId: number): void {
    this._focusedFace.set(faceId);
  }

  /** Show the photos the picked person shares with *otherId*. */
  chooseCoPerson(otherId: number): void {
    this._coPick.set(otherId);
  }

  /** Read the shared photos again: their Image dataset has just been loaded. */
  reloadShared(): void {
    this.sharedResource.reload();
  }

  setMinFaces(value: number): void {
    this.minFaces.set(Math.max(1, Math.floor(Number.isFinite(value) ? value : DEFAULT_MIN_FACES)));
  }

  // --- reads ---

  private readPerson(request: Observable<Person>): Observable<PersonRead> {
    return request.pipe(
      map((person): PersonRead => ({ kind: 'found', person })),
      catchError((err: unknown) =>
        of<PersonRead>(isStale(err) ? { kind: 'changed' } : { kind: 'error', message: errorText(err, "Couldn't read this person.") }),
      ),
    );
  }

  private readShared(request: Observable<{ photos: SharedPhoto[] }>): Observable<SharedRead> {
    return request.pipe(
      map((r): SharedRead => ({ kind: 'found', photos: r.photos })),
      catchError((err: unknown) => {
        if (isStale(err)) return of<SharedRead>({ kind: 'changed' });
        const notLoaded = unloadedSibling(err);
        if (notLoaded) return of<SharedRead>({ kind: 'not-loaded', datasetId: notLoaded });
        return of<SharedRead>({ kind: 'error', message: errorText(err, "Couldn't read the photos they share.") });
      }),
    );
  }
}

function errorBody(err: unknown): Record<string, unknown> {
  if (!(err instanceof HttpErrorResponse) || !err.error || typeof err.error !== 'object') return {};
  return err.error as Record<string, unknown>;
}

/** A 409 saying the partition the id was read under is gone (or not there yet). */
function isStale(err: unknown): boolean {
  if (!(err instanceof HttpErrorResponse) || err.status !== 409) return false;
  const code = errorBody(err)['error_code'];
  return code === 'people_changed' || code === 'people_not_ready';
}

/** The Image dataset a 409 `source_not_loaded` names, or ''. */
function unloadedSibling(err: unknown): string {
  if (!(err instanceof HttpErrorResponse) || err.status !== 409) return '';
  const body = errorBody(err);
  return body['error_code'] === 'source_not_loaded' && typeof body['dataset_id'] === 'string' ? body['dataset_id'] : '';
}

function errorText(err: unknown, fallback: string): string {
  return err instanceof HttpErrorResponse ? apiErrorMessage(err, fallback) : fallback;
}
