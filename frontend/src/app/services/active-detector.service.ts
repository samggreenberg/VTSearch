import { Injectable, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { DetectorRegistryEntry } from '../generated/api-client/models/detector-registry-entry';
import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';

/**
 * Signal view of *which detector the user is working with*, resolved from the
 * active-context id to the registry entry (and therefore to a display name).
 *
 * `ActiveContextService` carries ids only, on an RxJS layer that a `computed`
 * can't track, so every consumer that wanted a name re-did the same
 * `detectors.find(...)` lookup imperatively. That reads whatever happens to be
 * loaded at call time and never updates when the registry lands a moment
 * later: the lifecycle gap that leaves an export filename detector-less when
 * the modal opens before the detector registry resolves (issue #2819).
 *
 * Exposing the name as a signal closes both halves: it is one source of truth,
 * and it repopulates on its own once the registry (or an in-flight context
 * switch) settles, so `computed`s, `effect`s and templates downstream follow
 * along instead of latching a stale first read.
 *
 * Deliberately a separate service rather than fields on `ActiveContextService`:
 * that one is injected by the HTTP interceptor, and pulling the registry state
 * (which itself depends on `HttpClient`) into it would make the interceptor's
 * dependency graph cyclic.
 */
@Injectable({ providedIn: 'root' })
export class ActiveDetectorService {
  private readonly activeContext = inject(ActiveContextService);
  private readonly datasetState = inject(DatasetStateService);

  /** Detector id currently loaded into the backend (`''` when none). */
  readonly activeId = toSignal(this.activeContext.modelId$, { initialValue: '' });

  /** Detector id the user picked, which leads {@link activeId} for as long as
   *  a context switch is still loading. */
  readonly intentId = toSignal(this.activeContext.intentModelId$, { initialValue: '' });

  /**
   * The id to describe: the active detector, or — when nothing is active yet —
   * the one being switched to, so the UI can name the user's pick immediately
   * instead of blanking for the duration of the load.
   */
  readonly detectorId = computed(() => this.activeId() || this.intentId());

  /** Registry entry for {@link detectorId}: null when nothing is selected, and
   *  also while the registry fetch is still in flight. */
  readonly detector = computed<DetectorRegistryEntry | null>(() => {
    const id = this.detectorId();
    if (!id) return null;
    return this.datasetState.detectors.find((d) => d.id === id) ?? null;
  });

  /** Display name of the selected detector; `''` when unknown (nothing
   *  selected, or the registry hasn't resolved the id yet). */
  readonly detectorName = computed(() => this.detector()?.name ?? '');
}
