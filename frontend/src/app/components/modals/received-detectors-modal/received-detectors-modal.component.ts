import { ChangeDetectionStrategy, Component, inject, OnInit, output, signal } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';

import { ModalComponent } from '../../modal/modal.component';
import { DetectorSharingFlowsService } from '../../../services/detector-sharing-flows.service';
import { DetectorsSharingApiService, type ForeignPolicy } from '../../../services/detectors-sharing-api.service';
import { SharingCapabilityService } from '../../../services/sharing-capability.service';
import { ToastService } from '../../../services/toast.service';
import { apiErrorMessage } from '../../../utils/api-error';
import { formatTimestamp } from '../../../utils/format-date';
import type { DetectorRegistryEntry } from '../../../generated/api-client/models/detector-registry-entry';
import type { DetectorSharingReceiveResponse } from '../../../generated/api-client/models/detector-sharing-receive-response';
import type { DetectorSharingSummary } from '../../../generated/api-client/models/detector-sharing-summary';

/** The words for how a container reached the user (`shared_via`). */
const SHARED_VIA: Record<string, string> = {
  direct: 'Directly',
  group: 'By group',
  public: 'Public',
  owner: 'Yours',
};

/**
 * **Received…** (#5579): the detectors other people share with the user,
 * opened from the Detectors card's header while sharing is on.
 *
 * Each row is a container from `GET /api/detectors/sharing/received`. **Add**
 * receives it as a read-only detector of the user's (`POST .../receive`); a
 * container someone other than its owner has written to asks first, through
 * the same integrity dialog Check for updates uses, and is sent again with the
 * policy the user picks. A row the user already holds says **Added**, with a
 * link to their copy. The list stays open after an Add, so several can be
 * received in a row; the new detector lands in Drafts, selected, behind it.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-received-detectors-modal',
  standalone: true,
  imports: [ModalComponent],
  templateUrl: './received-detectors-modal.component.html',
  styleUrl: './received-detectors-modal.component.scss',
})
export class ReceivedDetectorsModalComponent implements OnInit {
  private readonly api = inject(DetectorsSharingApiService);
  private readonly flows = inject(DetectorSharingFlowsService);
  private readonly toast = inject(ToastService);
  readonly sharing = inject(SharingCapabilityService);

  readonly closed = output<void>();
  /** A container was received: the new detector, for the card to select. */
  readonly received = output<DetectorRegistryEntry>();
  /** The user asked to see their copy of a container: its detector id. */
  readonly show = output<string>();

  readonly rows = signal<DetectorSharingSummary[]>([]);
  readonly loading = signal(true);
  readonly loadError = signal('');
  /** The `remote_ref` being received now, or `''`. */
  readonly adding = signal('');

  readonly formatTimestamp = formatTimestamp;

  ngOnInit(): void {
    this.api.listReceived().subscribe({
      next: (rows) => {
        this.rows.set(rows);
        this.loading.set(false);
      },
      error: (err) => {
        this.loadError.set(apiErrorMessage(err, 'Could not list the detectors shared with you'));
        this.loading.set(false);
      },
    });
  }

  sharedVia(row: DetectorSharingSummary): string {
    return SHARED_VIA[row.shared_via] ?? row.shared_via;
  }

  /** Why a row's Add is off, or `''`. */
  addBlocker(row: DetectorSharingSummary): string {
    const busy = this.adding();
    if (!busy) return '';
    return busy === row.remote_ref ? 'Adding…' : 'Wait for the other detector to be added';
  }

  /**
   * Receive *row*'s container. Asks first when someone other than its owner
   * has written to it (`foreign: abort` answers `applied: false`), then sends
   * the user's choice; Cancel there receives nothing.
   */
  async add(row: DetectorSharingSummary): Promise<void> {
    if (this.adding()) return;
    this.adding.set(row.remote_ref);
    try {
      let response = await this.receive(row.remote_ref, 'abort');
      if (response && !response.applied) {
        const choice = await this.flows.askAboutForeignRecords(
          { id: row.remote_ref, name: response.name || row.name },
          response.integrity?.count ?? 0,
        );
        response = choice === 'cancel' ? null : await this.receive(row.remote_ref, choice);
      }
      if (response?.applied && response.detector) {
        this.markAdded(row.remote_ref, response.detector.id);
        this.toast.success({ message: response.message });
        this.received.emit(response.detector);
      }
    } finally {
      this.adding.set('');
    }
  }

  showCopy(row: DetectorSharingSummary): void {
    if (row.detector_id) this.show.emit(row.detector_id);
  }

  close(): void {
    this.closed.emit();
  }

  /**
   * One receive request, resolving `null` on a failure (which the error
   * interceptor has toasted). A container the user turns out to hold already
   * (`already_received`, a receive from another tab) is marked Added.
   */
  private async receive(remoteRef: string, foreign: ForeignPolicy): Promise<DetectorSharingReceiveResponse | null> {
    try {
      return await firstValueFrom(this.api.receive(remoteRef, foreign));
    } catch (err) {
      const body = err instanceof HttpErrorResponse ? (err.error as { error_code?: unknown; detector_id?: unknown }) : null;
      if (body?.error_code === 'already_received' && typeof body.detector_id === 'string') {
        this.markAdded(remoteRef, body.detector_id);
      }
      return null;
    }
  }

  private markAdded(remoteRef: string, detectorId: string): void {
    this.rows.update((rows) => rows.map((r) => (r.remote_ref === remoteRef ? { ...r, detector_id: detectorId } : r)));
  }
}
