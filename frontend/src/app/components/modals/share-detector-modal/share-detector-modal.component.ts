import {
  ChangeDetectionStrategy,
  Component,
  computed,
  inject,
  input,
  OnInit,
  output,
  signal,
  viewChild,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, firstValueFrom, of } from 'rxjs';
import { catchError, debounceTime, distinctUntilChanged, map, switchMap } from 'rxjs/operators';

import { ModalComponent } from '../../modal/modal.component';
import {
  ComboboxOption,
  comboboxValueFor,
  PluginComboboxComponent,
} from '../../plugin-combobox/plugin-combobox.component';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { DetectorsSharingApiService } from '../../../services/detectors-sharing-api.service';
import { SharingCapabilityService } from '../../../services/sharing-capability.service';
import { VtDialogService } from '../../../services/dialog.service';
import { ToastService } from '../../../services/toast.service';
import { apiErrorMessage } from '../../../utils/api-error';
import { sharingNote } from '../../../utils/detector-sharing';
import type { DetectorShareRequest } from '../../../generated/api-client/models/detector-share-request';
import type { DetectorSharingRecipient } from '../../../generated/api-client/models/detector-sharing-recipient';
import type { DetectorSharingView } from '../../../generated/api-client/models/detector-sharing-view';

/** How long the recipient search waits after the last keystroke. */
export const RECIPIENT_SEARCH_DEBOUNCE_MS = 250;

/** What the server's own `share` body is: one change per call. */
type ShareChange = Pick<DetectorShareRequest, 'add' | 'remove' | 'public'>;

/**
 * **Share…** for a detector of the user's own (#5579): who its container is
 * shared with, opened from the Detectors card's menu and from its "Remote sync
 * failed" link.
 *
 * Every change is one `POST .../share` and lands at once, so there is nothing
 * to save: adding a person (picked from the source's suggestions, or typed as
 * an identity when the source cannot search), removing one, and the Public
 * box. The first of them creates the container, which is why a detector that
 * has none says what sharing will do, and that it replaces a plain auto-export
 * binding. Stop Sharing archives the container. A detector whose container
 * belongs to a source that is not the active one (`sharing_note`) opens
 * read-only, with the note.
 *
 * The requests are serialised: two first changes in flight at once would make
 * two containers.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-share-detector-modal',
  standalone: true,
  imports: [ModalComponent, PluginComboboxComponent],
  templateUrl: './share-detector-modal.component.html',
  styleUrl: './share-detector-modal.component.scss',
})
export class ShareDetectorModalComponent implements OnInit {
  private readonly api = inject(DetectorsSharingApiService);
  private readonly datasetState = inject(DatasetStateService);
  private readonly dialog = inject(VtDialogService);
  private readonly toast = inject(ToastService);
  readonly sharing = inject(SharingCapabilityService);

  readonly detectorId = input('');
  readonly detectorName = input('');
  readonly closed = output<void>();

  /** The detector's registry entry, which says whether it has a container. */
  private readonly entry = computed(() => this.datasetState.detectorById().get(this.detectorId()));
  /** Why the container is not in use here, or `''`: the read-only state. */
  readonly note = computed(() => sharingNote(this.entry()));

  /** The container as last read or changed, or `null` before either. */
  readonly view = signal<DetectorSharingView | null>(null);
  readonly loading = signal(false);
  readonly loadError = signal('');
  /** A change (or Stop Sharing) is in flight. */
  readonly busy = signal(false);

  /** The detector has a container: listed by the registry, or made just now. */
  readonly hasContainer = computed(() => !!this.view() || !!this.entry()?.sharing?.remote_ref);
  /** The plain auto-export the first share replaces, by display name, or `''`. */
  readonly replacedBinding = computed(() => {
    if (this.hasContainer()) return '';
    const status = this.entry()?.labelset_source_status;
    return status ? status.display_name || status.source_name : '';
  });
  readonly sourceName = computed(() => this.sharing.displayName() || 'the sharing service');
  readonly recipients = computed(() => this.view()?.recipients ?? []);
  readonly isPublic = computed(() => this.view()?.public === true);

  /** The recipient box's text, as typed. */
  readonly query = signal('');
  /** The source's suggestions for {@link query}, those already shared with left out. */
  private readonly suggestions = signal<DetectorSharingRecipient[]>([]);
  readonly suggestionOptions = computed<ComboboxOption[]>(() => {
    const already = new Set(this.recipients());
    return this.suggestions()
      .filter((s) => !already.has(s.id))
      .map((s) => ({ value: s.id, label: s.display_name || s.id, tag: s.kind === 'group' ? 'group' : 'person' }));
  });
  private readonly query$ = new Subject<string>();
  private readonly combobox = viewChild(PluginComboboxComponent);

  /** Why the box's text cannot be added now, or `''`. */
  readonly addBlocker = computed(() => {
    if (this.busy()) return 'Wait for the last change to finish';
    if (this.loadError()) return 'Close and reopen to try again';
    if (!this.query().trim()) return 'Type who to share with first';
    return '';
  });
  /** Why the remove buttons and the Public box are off, or `''`. */
  readonly changeBlocker = computed(() => {
    if (this.busy()) return 'Wait for the last change to finish';
    if (this.loadError()) return 'Close and reopen to try again';
    return '';
  });

  constructor() {
    this.query$
      .pipe(
        map((q) => q.trim()),
        debounceTime(RECIPIENT_SEARCH_DEBOUNCE_MS),
        distinctUntilChanged(),
        switchMap((q) => (q ? this.api.searchRecipients(q).pipe(catchError(() => of([]))) : of([]))),
        takeUntilDestroyed(),
      )
      .subscribe((found) => this.suggestions.set(found));
  }

  ngOnInit(): void {
    // A detector with no container has nobody to list; one whose container
    // is not in use here is not read (the server would answer the note).
    if (this.note() || !this.entry()?.sharing?.remote_ref) return;
    this.loading.set(true);
    this.api.getSharingView(this.detectorId()).subscribe({
      next: (view) => {
        this.view.set(view);
        this.loading.set(false);
      },
      error: (err) => {
        this.loadError.set(apiErrorMessage(err, 'Could not read who this detector is shared with'));
        this.loading.set(false);
      },
    });
  }

  onQuery(text: string): void {
    this.query.set(text);
    this.query$.next(text);
  }

  onPick(option: ComboboxOption): void {
    void this.add(option.value);
  }

  /** Enter in the box adds what is typed, unless it just picked a row. */
  onEnter(event: Event): void {
    if (event.defaultPrevented) return;
    event.preventDefault();
    this.addTyped();
  }

  /** The typed text, or the suggestion whose name it is exactly. */
  addTyped(): void {
    const text = this.query().trim();
    if (!text) return;
    void this.add(comboboxValueFor(text, this.suggestionOptions()));
  }

  async add(identity: string): Promise<void> {
    const who = identity.trim();
    if (!who || this.busy()) return;
    if (this.recipients().includes(who)) {
      this.clearEntry();
      return;
    }
    if (await this.change({ add: [who] })) this.clearEntry();
  }

  remove(recipient: string): void {
    void this.change({ remove: [recipient] });
  }

  /** The Public box. A refused change puts the box back as it was. */
  async setPublic(event: Event): Promise<void> {
    const box = event.target as HTMLInputElement;
    const wanted = box.checked;
    if (!(await this.change({ public: wanted }))) box.checked = this.isPublic();
  }

  async stopSharing(): Promise<void> {
    if (this.busy()) return;
    const ok = await this.dialog.confirmDestructive(
      `Stop sharing "${this.detectorName()}"?`,
      'The people it is shared with keep their copies, which stop getting your changes. ' +
        'Your detector and its labels stay as they are.',
      'Stop Sharing',
    );
    if (!ok) return;
    this.busy.set(true);
    try {
      const response = await firstValueFrom(this.api.unshare(this.detectorId()));
      this.datasetState.patchDetector(this.detectorId(), response.detector);
      this.toast.success({ message: response.message });
      this.closed.emit();
    } catch {
      // The error interceptor has toasted it; the dialog stays as it was.
    } finally {
      this.busy.set(false);
    }
  }

  close(): void {
    this.closed.emit();
  }

  /**
   * Send one change and take the container it answers with. Resolves `true`
   * when it landed. The first change creates the container: its message
   * (which says what auto-export it replaced) is toasted, and the registry is
   * re-read so the row gains its Shared badge.
   */
  private async change(body: ShareChange): Promise<boolean> {
    if (this.busy()) return false;
    this.busy.set(true);
    try {
      const response = await firstValueFrom(this.api.share(this.detectorId(), body));
      this.view.set(response);
      if (response.created) {
        this.toast.success({ message: response.message });
        this.datasetState.refresh();
      }
      return true;
    } catch {
      // The error interceptor has toasted it.
      return false;
    } finally {
      this.busy.set(false);
    }
  }

  private clearEntry(): void {
    this.query.set('');
    this.query$.next('');
    this.suggestions.set([]);
    this.combobox()?.reset();
  }
}
