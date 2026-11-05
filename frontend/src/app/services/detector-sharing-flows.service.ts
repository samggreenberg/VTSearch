import { Injectable, inject } from '@angular/core';
import { Subject, firstValueFrom } from 'rxjs';

import type { DetectorRegistryEntry } from '../generated/api-client/models/detector-registry-entry';
import { DatasetStateService } from './dataset-state.service';
import { VtDialogService, type DialogButton } from './dialog.service';
import {
  DetectorsSharingApiService,
  type ForeignPolicy,
  type SharingRefreshResponse,
} from './detectors-sharing-api.service';
import { SharingCapabilityService } from './sharing-capability.service';
import { ToastService } from './toast.service';

/** The detector a flow acts on: its id, and the name its messages use. */
export interface SharingFlowTarget {
  id: string;
  name?: string;
}

/** A fork that landed: the detector it copied, and the copy. */
export interface ForkLanded {
  sourceId: string;
  entry: DetectorRegistryEntry;
}

/** How a Check for updates ended. */
export type RefreshOutcome = 'applied' | 'cancelled' | 'failed';

/** The integrity dialog's answers: re-post with a policy, or leave it. */
type IntegrityChoice = Exclude<ForeignPolicy, 'abort'> | 'cancel';

/**
 * The two sharing actions that more than one surface starts: **Check for
 * updates** on a received detector (the Detectors card) and **Fork** (the
 * card, the Train view's read-only line, and the toast a 409
 * `detector_read_only` raises). Each is a short conversation — a request, a
 * question when the answer needs one, a toast — and lives here once so the
 * surfaces only say what happens next (which row to select, which detector to
 * switch to).
 *
 * Failures toast through `errorInterceptor` as any request's do; these
 * methods resolve to say how the flow ended and never reject.
 */
@Injectable({ providedIn: 'root' })
export class DetectorSharingFlowsService {
  private readonly api = inject(DetectorsSharingApiService);
  private readonly dialog = inject(VtDialogService);
  private readonly toast = inject(ToastService);
  private readonly datasetState = inject(DatasetStateService);
  private readonly sharing = inject(SharingCapabilityService);
  private readonly forkedSubject = new Subject<ForkLanded>();

  /**
   * Every fork that landed, wherever it was started. The Train view follows
   * it: a fork of the detector it is training on switches it to the copy,
   * whether the Fork came from its own read-only line or from a toast.
   */
  readonly forked$ = this.forkedSubject.asObservable();

  /**
   * Fetch the owner's latest labels into a received detector. Asks first
   * (`foreign: abort`), so a container someone other than the owner has
   * written to is never applied unseen: the dialog says how many votes, and
   * offers Import anyway, Skip foreign records, or Cancel. The row's
   * `last_synced` and integrity update from the response.
   */
  async checkForUpdates(target: SharingFlowTarget): Promise<RefreshOutcome> {
    let response = await this.refresh(target.id, 'abort');
    if (!response) return 'failed';
    // An abort that found foreign records changed nothing but the integrity
    // it recorded, which the row's badge shows whatever the user picks next.
    this.landRefresh(target.id, response);
    if (!response.applied) {
      const choice = await this.askAboutForeignRecords(target, response.integrity?.count ?? 0);
      if (choice === 'cancel') return 'cancelled';
      response = await this.refresh(target.id, choice);
      if (!response) return 'failed';
      this.landRefresh(target.id, response);
    }
    // The server's sentence carries the counts: added, changed, removed.
    this.toast.success({ message: response.message });
    return 'applied';
  }

  /**
   * Make a copy of *target* the user owns and can edit, optionally asking for
   * its name first. Resolves to the new entry, with a registry refresh on its
   * way to list it, or `null` when the user cancelled or the request failed. A `remote_sync` note (the copy could not be shared)
   * becomes a toast.
   */
  async fork(target: SharingFlowTarget, options: { askName?: boolean } = {}): Promise<DetectorRegistryEntry | null> {
    let name: string | undefined;
    if (options.askName) {
      // The server's own default, shown so the user can keep it. Kept as is,
      // or left blank, no name is sent: the server suffixes its default
      // until it is free, where a name sent that is taken answers 409.
      const suggested = target.name ? `${target.name} (fork)` : '';
      const answer = await this.dialog.prompt(`Name your copy of "${target.name ?? 'this detector'}":`, suggested);
      if (answer === null) return null;
      const typed = answer.trim();
      name = typed && typed !== suggested ? typed : undefined;
    }
    let response;
    try {
      response = await firstValueFrom(this.api.fork(target.id, name));
    } catch {
      return null;
    }
    const entry = response.detector;
    this.datasetState.refresh();
    if (!response.remote_sync) {
      this.toast.success({ message: response.message });
    } else if (this.sharing.enabled()) {
      // Sharing is on and the copy could not be shared: not what was asked.
      this.toast.warning({ message: `${response.message} ${response.remote_sync}` });
    } else {
      this.toast.info({ message: `${response.message} ${response.remote_sync}` });
    }
    this.forkedSubject.next({ sourceId: target.id, entry });
    return entry;
  }

  private async refresh(id: string, foreign: ForeignPolicy): Promise<SharingRefreshResponse | null> {
    try {
      return await firstValueFrom(this.api.refresh(id, foreign));
    } catch {
      return null;
    }
  }

  private askAboutForeignRecords(target: SharingFlowTarget, count: number): Promise<IntegrityChoice> {
    const votes = `${count} vote${count === 1 ? ' was' : 's were'}`;
    const buttons: DialogButton<IntegrityChoice>[] = [
      { label: 'Cancel', primary: false, value: 'cancel' },
      { label: 'Import anyway', primary: false, value: 'import' },
      { label: 'Skip foreign records', primary: true, value: 'skip' },
    ];
    return this.dialog.choose(
      `${votes} modified by someone other than the owner of "${target.name ?? 'this detector'}".`,
      buttons,
      'warning',
      'Skip them to take only what the owner saved, or import them as they are. ' +
        'Nothing changes until you choose.',
    );
  }

  /** Write what the refresh says about the row into the store, without a reload. */
  private landRefresh(id: string, response: SharingRefreshResponse): void {
    if (response.detector) this.datasetState.patchDetector(id, response.detector);
  }
}
