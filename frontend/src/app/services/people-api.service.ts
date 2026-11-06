import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import { buildPeople } from '../generated/api-client/fn/people/build-people';
import { getPerson } from '../generated/api-client/fn/people/get-person';
import { listPeople } from '../generated/api-client/fn/people/list-people';
import { sharedPhotos } from '../generated/api-client/fn/people/shared-photos';
import type { PeopleList } from '../generated/api-client/models/people-list';
import type { Person } from '../generated/api-client/models/person';
import type { SharedPhotos } from '../generated/api-client/models/shared-photos';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';

/**
 * The People routes (`vtsearch/routes/people.py`, #5589), through the
 * generated client: a Face dataset's people, one person, and the photos two
 * people share. Every call is about the active dataset (`X-Dataset-Id`).
 *
 * Every call is quiet (`SKIP_ERROR_TOAST`): the People view and its prep
 * answer each failure inline. A 409 here is mostly an answer, not a fault:
 * `people_changed` (the partition moved under a held person id; list again),
 * `people_not_ready`, `no_people` and the shared `source_not_loaded`.
 */
@Injectable({ providedIn: 'root' })
export class PeopleApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  /** The partition's status and its people; `building` while a fit runs (poll it). */
  list(): Observable<PeopleList> {
    return listPeople(this.http, this.config.rootUrl, {}, quietContext()).pipe(map((r) => r.body));
  }

  /** Fit the people again (Rebuild people); answers as {@link list} does. */
  rebuild(): Observable<PeopleList> {
    return buildPeople(this.http, this.config.rootUrl, { body: { force: true } }, quietContext()).pipe(
      map((r) => r.body),
    );
  }

  /** One person's faces, photos and who appears with them, under *partition*. */
  person(personId: number, partition: string): Observable<Person> {
    return getPerson(this.http, this.config.rootUrl, { person_id: personId, partition }, quietContext()).pipe(
      map((r) => r.body),
    );
  }

  /** The photos *personId* and *otherId* share, each with every face in it. */
  shared(personId: number, otherId: number, partition: string): Observable<SharedPhotos> {
    return sharedPhotos(
      this.http,
      this.config.rootUrl,
      { person_id: personId, other_id: otherId, partition },
      quietContext(),
    ).pipe(map((r) => r.body));
  }
}

function quietContext(): HttpContext {
  return new HttpContext().set(SKIP_ERROR_TOAST, true);
}
