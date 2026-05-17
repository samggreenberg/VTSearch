import { Injectable } from '@angular/core';
import { Router } from '@angular/router';
import { combineLatest } from 'rxjs';
import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';
import { ToastService } from './toast.service';

/**
 * Watches the registry vs. the active pair and reacts when the active
 * dataset or detector disappears from the registry (deleted /
 * unregistered from another tab, the CLI, or a different session):
 *
 *  1. Clears the affected half on `ActiveContextService` so the
 *     interceptor stops sending a header pointing at a dead id.
 *  2. Emits a toast naming the removed item so the user knows their
 *     view state changed underneath them.
 *
 * Only fires after we've seen the active id appear in the registry at
 * least once — a transient "id set, registry still loading" state on
 * page load shouldn't trigger a phantom deletion toast.
 *
 * See `docs/plans/active-context-switcher.md` § "Edge cases" #3 and
 * "Open follow-ups (Phase 1)".
 */
@Injectable({ providedIn: 'root' })
export class ActiveContextWatcherService {
  private lastSeenDatasetName: { id: string; name: string } | null = null;
  private lastSeenDetectorName: { id: string; name: string } | null = null;
  private started = false;

  constructor(
    private activeContext: ActiveContextService,
    private datasetState: DatasetStateService,
    private toast: ToastService,
    private router: Router,
  ) {}

  /** Navigate to /dashboard when an active half is cleared from under
   *  us. Phase 2 makes the URL the source of truth, so a half-set pair
   *  is not a representable URL state — bouncing to /dashboard is the
   *  only sensible recovery. No-op when the user is already on
   *  `/dashboard`. */
  private leaveBrokenPairView(): void {
    const url = this.router.url.split('?')[0];
    if (url.startsWith('/label') || url.startsWith('/find')) {
      this.router.navigate(['/dashboard']);
    }
  }

  /** Idempotent — safe to call from multiple bootstrappers. */
  start(): void {
    if (this.started) return;
    this.started = true;
    combineLatest([
      this.activeContext.pair$,
      this.datasetState.datasets$,
      this.datasetState.detectors$,
    ]).subscribe(([pair, datasets, detectors]) => {
      const activeDataset = pair.datasetId
        ? datasets.find((d) => d.id === pair.datasetId)
        : null;
      const activeDetector = pair.modelId
        ? detectors.find((d) => d.id === pair.modelId)
        : null;

      if (activeDataset) {
        this.lastSeenDatasetName = { id: activeDataset.id, name: activeDataset.name };
      } else if (
        pair.datasetId &&
        this.lastSeenDatasetName?.id === pair.datasetId &&
        datasets.length > 0
      ) {
        const removedName = this.lastSeenDatasetName.name;
        this.lastSeenDatasetName = null;
        this.activeContext.setActivePair('', pair.modelId);
        this.toast.error({
          message: `Dataset '${removedName}' was removed.`,
          detail: 'Pick another from the top-bar pulldown.',
          dedupKey: `active-removed:dataset:${pair.datasetId}`,
        });
        this.leaveBrokenPairView();
      }

      if (activeDetector) {
        this.lastSeenDetectorName = { id: activeDetector.id, name: activeDetector.name };
      } else if (
        pair.modelId &&
        this.lastSeenDetectorName?.id === pair.modelId &&
        detectors.length > 0
      ) {
        const removedName = this.lastSeenDetectorName.name;
        this.lastSeenDetectorName = null;
        this.activeContext.setActivePair(pair.datasetId, '');
        this.toast.error({
          message: `Detector '${removedName}' was removed.`,
          detail: 'Pick another from the top-bar pulldown.',
          dedupKey: `active-removed:detector:${pair.modelId}`,
        });
        this.leaveBrokenPairView();
      }
    });
  }
}
