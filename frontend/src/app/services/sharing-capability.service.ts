import { Injectable, computed, inject, signal } from '@angular/core';
import { catchError, filter, of, switchMap, take } from 'rxjs';

import { AuthService } from './auth.service';
import { DetectorsSharingApiService, type SharingCapability } from './detectors-sharing-api.service';

/**
 * Whether this deployment shares detectors, and through what: the
 * `GET /api/detectors/sharing` capability, fetched once, after the user is
 * signed in (a sign-in wall answers every other request 401, this one
 * included).
 *
 * Every ambient sharing affordance reads {@link enabled}: the card menu's
 * Share… and a Fork of one's own detector. It is `false` until the answer
 * lands and stays `false` when the request fails, so a deployment without the
 * setting (or a server without the route) sees today's card. What the
 * registry listing already says about an entry (a Received badge, a note) does
 * not wait on it.
 */
@Injectable({ providedIn: 'root' })
export class SharingCapabilityService {
  private readonly api = inject(DetectorsSharingApiService);
  private readonly auth = inject(AuthService);

  private readonly capability = signal<SharingCapability | null>(null);

  /** Sharing is on here. */
  readonly enabled = computed(() => this.capability()?.enabled === true);
  /** The active sharing source's id, or `''`. */
  readonly sourceName = computed(() => (this.enabled() ? (this.capability()?.name ?? '') : ''));
  /** What the sharing source calls itself, for "Share through …". */
  readonly displayName = computed(() =>
    this.enabled() ? (this.capability()?.display_name || this.capability()?.name || '') : '',
  );
  /** The source's icon name, or `''`. */
  readonly icon = computed(() => (this.enabled() ? (this.capability()?.icon ?? '') : ''));
  /** The source can suggest recipients as the user types. */
  readonly supportsRecipientSearch = computed(
    () => this.enabled() && this.capability()?.supports_recipient_search === true,
  );

  constructor() {
    this.auth.status$
      .pipe(
        filter((status) => !!status && (status.authenticated || !status.login_required)),
        take(1),
        switchMap(() => this.api.getCapability().pipe(catchError(() => of(null)))),
      )
      .subscribe((capability) => this.capability.set(capability));
  }
}
