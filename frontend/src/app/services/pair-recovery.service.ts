import { HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Router } from '@angular/router';
import { Subject } from 'rxjs';
import { ActiveContextService } from './active-context.service';
import { ContextSwitchService } from './context-switch.service';
import { DatasetStateService } from './dataset-state.service';
import { ToastService } from './toast.service';

/**
 * How soon after a reload lands a fresh loss of the same pair is read as the
 * reload not holding, rather than as a second restart. Stops a backend that
 * keeps answering 409 from cycling the view through reload after reload.
 */
const RELOAD_COOLDOWN_MS = 60_000;

/** Dataset id prefix of a detector-positives map (`vtscore.detectors.positives_browse`). */
const POSITIVES_MAP_PREFIX = '__detpos__';

/**
 * Loads the active pair again when the backend turns out to have lost it (#5155).
 *
 * A restart (a redeploy, the job's time limit, a crash) empties the backend of
 * loaded pairs, but an open Train or Tune tab goes on naming its pair on every
 * request, so each one answers 409 `dataset_not_loaded` / `detector_not_loaded`.
 * Nothing else loads it again: the route guard let the view in long ago, and
 * the polls just skip a failed tick. The view sat dead under a red toast until
 * the reviewer went back to the Dashboard and reopened the queue. An open
 * Browse tab went the same way, only quietly: its tiles opt out of the toast,
 * so the map just stopped filling in (#5158).
 *
 * {@link errorInterceptor} hands this service every failed response. A 409 that
 * names the active pair, while no switch is loading it and the tab is on that
 * pair's view (Train or Tune, or Browse on its dataset), starts a reload
 * ({@link ContextSwitchService.reloadActivePair}); once it lands,
 * {@link reloaded$} fires and the view starts over as on a pair change, since
 * whatever the backend held for the session went with it. A 409
 * that lands while a switch or a reload is in flight is expected and absorbed
 * (the #3644 case: a poll landing mid-load). A reload that fails sends the tab
 * to the Dashboard with a message saying to reopen the pair from there.
 *
 * A detector-positives map (`/browse/__detpos__<detectorId>`) cannot be loaded
 * again at all: its context lived only in the memory of the process that built
 * it, so after a restart the backend answers 404 `dataset_not_found` instead,
 * and a tab on it goes straight to the Dashboard to browse them again (#5159).
 */
@Injectable({ providedIn: 'root' })
export class PairRecoveryService {
  private activeContext = inject(ActiveContextService);
  private contextSwitch = inject(ContextSwitchService);
  private datasetState = inject(DatasetStateService);
  private router = inject(Router);
  private toast = inject(ToastService);

  private epochValue = 0;
  private reloading = false;
  /** Set while {@link giveUp} is taking the tab off the pair's view. */
  private leaving = false;
  private lastReloadAt = Number.NEGATIVE_INFINITY;

  private readonly reloadedSubject = new Subject<void>();
  /** Fires when a reload lands. Train, Tune and Browse reset on it as on a pair change. */
  readonly reloaded$ = this.reloadedSubject.asObservable();

  /**
   * Counts the reloads that have landed. {@link errorInterceptor} reads it as
   * each request goes out and passes it back with the response, so a 409 for a
   * request sent before the latest reload landed is known to be one that reload
   * has already answered, and does not start another.
   */
  get epoch(): number {
    return this.epochValue;
  }

  /**
   * Called by {@link errorInterceptor} for every failed response; *sentEpoch* is
   * {@link epoch} when the request went out. Returns `true` when the failure is
   * the active pair missing from the backend and this service is answering it
   * (by a reload, one already in flight, or a switch that is loading it), so the
   * interceptor raises no toast for it.
   */
  absorb(err: HttpErrorResponse, sentEpoch: number): boolean {
    const lostMap = lostPositivesMap(err);
    if (lostMap) return this.leaveLostPositivesMap(lostMap);
    const { datasetId, modelId } = this.activeContext;
    if (!namesUnloaded(err, datasetId, modelId)) return false;
    // Mid-load: the switch or reload in flight is what will load it. Or the
    // tab is already on its way off the view.
    if (this.reloading || this.leaving || this.contextSwitch.switching) return true;
    // Sent before the latest reload landed, so that reload has answered it.
    if (sentEpoch < this.epochValue) return true;
    // Only a pair a view is showing is worth loading again; anywhere else the
    // 409 stays an ordinary error.
    if (!this.onPairView(datasetId, modelId)) return false;
    if (Date.now() - this.lastReloadAt < RELOAD_COOLDOWN_MS) {
      this.giveUp(datasetId, modelId);
    } else {
      this.reload(datasetId, modelId);
    }
    return true;
  }

  private reload(datasetId: string, detectorId: string): void {
    this.reloading = true;
    const label = this.pairLabel(datasetId, detectorId);
    const dedupKey = reloadToastKey(datasetId, detectorId);
    const toastId = this.toast.info({
      message: `The server had lost ${label}; loading it again`,
      detail: 'It probably restarted. The view starts over once the load finishes.',
      autoDismissMs: 0,
      dedupKey,
    });
    let reloaded = false;
    this.contextSwitch.reloadActivePair().subscribe({
      next: () => (reloaded = true),
      complete: () => {
        this.reloading = false;
        if (reloaded) {
          this.epochValue += 1;
          this.lastReloadAt = Date.now();
          this.toast.success({ message: `${label} is loaded again`, dedupKey });
          this.reloadedSubject.next();
          return;
        }
        // Completed without loading: a switch the user made meanwhile now owns
        // the pair, or they left the view. Otherwise a load failed.
        if (!this.contextSwitch.switching && this.onPairView(datasetId, detectorId)) {
          this.giveUp(datasetId, detectorId);
        } else {
          this.toast.dismiss(toastId);
        }
      },
    });
  }

  /** The pair cannot be loaded again from here: say so and leave its dead view. */
  private giveUp(datasetId: string, detectorId: string): void {
    this.toast.error({
      message: `Couldn't load ${this.pairLabel(datasetId, detectorId)} again`,
      detail: 'The server no longer had it loaded, probably after a restart. Reopen it from the Dashboard.',
      dedupKey: reloadToastKey(datasetId, detectorId),
    });
    this.leaveForDashboard();
  }

  /**
   * The server no longer has the positives map *mapId*, so the tab browsing it
   * goes back to the Dashboard, where the detector's Browse action builds a new
   * one. Absorbed even off that view, where it can only be a request the view
   * sent before it was left: the backend's "deleted; reload the page" would be
   * wrong twice over, since nothing was deleted and a reload cannot bring it back.
   */
  private leaveLostPositivesMap(mapId: string): true {
    const [view, ds] = this.viewSegments();
    if (this.leaving || view !== 'browse' || ds !== mapId) return true;
    const detectorId = mapId.slice(POSITIVES_MAP_PREFIX.length);
    const detector = this.datasetState.detectorById().get(detectorId)?.name || 'this detector';
    this.toast.error({
      message: `The map of ${detector}'s positives is gone`,
      detail:
        "The server no longer has it, probably after a restart. Browse the detector's positives again from the Dashboard.",
      dedupKey: `positives-map-lost:${mapId}`,
    });
    this.leaveForDashboard();
    return true;
  }

  /** Take the tab off its dead view. The pair is cleared once off it, so
   *  reopening the pair runs the route guard's load instead of its
   *  already-active fast path. */
  private leaveForDashboard(): void {
    this.leaving = true;
    this.router
      .navigate(['/dashboard'])
      .then((navigated) => {
        if (navigated) this.activeContext.setActivePair('', '');
      })
      .finally(() => (this.leaving = false));
  }

  /**
   * Whether the tab is on a view of this pair: Train or Tune of both halves, or
   * Browse of the dataset. Browse shows a dataset alone and carries the active
   * detector along (`browseContextGuard`), so its URL names no detector, but
   * the reload brings that half back too, as the guard had it loaded (#5158).
   */
  private onPairView(datasetId: string, detectorId: string): boolean {
    const [view, ds, det] = this.viewSegments();
    if (view === 'browse') return ds === datasetId;
    return (view === 'label' || view === 'tune') && ds === datasetId && det === detectorId;
  }

  /** The route's view and its id segments, query string dropped. */
  private viewSegments(): string[] {
    return this.router.url.split('?')[0].split('/').slice(1);
  }

  /** The pair as the view names it: Browse by its dataset alone. */
  private pairLabel(datasetId: string, detectorId: string): string {
    const dataset = this.datasetState.datasetById().get(datasetId)?.name || datasetId;
    if (!detectorId || this.viewSegments()[0] === 'browse') return dataset;
    const detector = this.datasetState.detectorById().get(detectorId)?.name || detectorId;
    return `${dataset} · ${detector}`;
  }
}

function reloadToastKey(datasetId: string, detectorId: string): string {
  return `pair-reload:${datasetId}:${detectorId}`;
}

/** Whether *err* is the backend's 409 for a half of the (*datasetId*,
 *  *detectorId*) pair it has not loaded. The id comes from the error body, so a
 *  request aimed at some other dataset (`DATASET_OVERRIDE`) never matches. */
function namesUnloaded(err: HttpErrorResponse, datasetId: string, detectorId: string): boolean {
  if (err.status !== 409) return false;
  const body = err.error as { error_code?: unknown; dataset_id?: unknown; detector_id?: unknown } | null;
  if (!body || typeof body !== 'object') return false;
  if (body.error_code === 'dataset_not_loaded') return !!datasetId && body.dataset_id === datasetId;
  if (body.error_code === 'detector_not_loaded') return !!detectorId && body.detector_id === detectorId;
  return false;
}

/** The positives map *err* says the backend no longer has (its 404
 *  `dataset_not_found`), or `''` when *err* is anything else. */
function lostPositivesMap(err: HttpErrorResponse): string {
  if (err.status !== 404) return '';
  const body = err.error as { error_code?: unknown; dataset_id?: unknown } | null;
  if (!body || typeof body !== 'object' || body.error_code !== 'dataset_not_found') return '';
  const id = body.dataset_id;
  return typeof id === 'string' && id.startsWith(POSITIVES_MAP_PREFIX) ? id : '';
}
