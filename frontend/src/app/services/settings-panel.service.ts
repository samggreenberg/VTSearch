import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';
import type { SettingsTabId } from '../components/modals/settings-modal/settings-modal.component';

/** Where a caller outside the app shell asks the Settings modal to open. */
export interface SettingsOpenRequest {
  tab: SettingsTabId;
  /** Open the settings importer over the tab straight away. */
  importer?: boolean;
}

/** How a Settings session opened through {@link SettingsPanelService.open} ended. */
export interface SettingsSessionResult {
  /** Whether a settings import was applied while it was open. */
  imported: boolean;
}

/**
 * Opens the Settings modal from outside the app shell, which owns it (the
 * gear opens it there directly). The Dashboard's AutoFind prompt (#5344)
 * opens it on the AutoFind tab, or with the settings importer up.
 *
 * `open()` resolves when the modal closes, saying whether a settings import
 * was applied meanwhile, so the caller can redo whatever an import may have
 * overwritten. The shell reports the import and the close through
 * {@link noteImported} and {@link closed}.
 */
@Injectable({ providedIn: 'root' })
export class SettingsPanelService {
  private readonly requestSubject = new Subject<SettingsOpenRequest>();
  /** Open requests, for the app shell. */
  readonly requests$: Observable<SettingsOpenRequest> = this.requestSubject.asObservable();

  private pending: { resolve: (result: SettingsSessionResult) => void; imported: boolean } | null = null;

  open(request: SettingsOpenRequest): Promise<SettingsSessionResult> {
    // A second open before the first closed settles the first, so its
    // caller is never left waiting.
    this.closed();
    return new Promise((resolve) => {
      this.pending = { resolve, imported: false };
      this.requestSubject.next(request);
    });
  }

  /** The shell: a settings import was applied in the open modal. */
  noteImported(): void {
    if (this.pending) this.pending.imported = true;
  }

  /** The shell: the Settings modal closed. */
  closed(): void {
    const pending = this.pending;
    this.pending = null;
    pending?.resolve({ imported: pending.imported });
  }
}
