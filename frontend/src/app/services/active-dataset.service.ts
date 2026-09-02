import { Injectable, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { DatasetRegistryEntry } from '../models/api.models';
import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';

/**
 * Signal view of *which dataset the user is working with*, resolved from the
 * active-context id to its registry entry.
 *
 * The dataset-half twin of {@link ActiveDetectorService}, and it exists for
 * the same reason: `ActiveContextService` carries ids only, on an RxJS layer
 * a `computed` can't track, so a consumer that wanted the entry (for its
 * name, its media type, its embedder) re-did the `datasets.find(...)` lookup
 * imperatively — reading whatever happened to be loaded at call time, and
 * never updating when the registry landed a moment later.
 *
 * The two services are deliberately separate objects rather than one
 * `ActiveContext` facade with four fields: consumers overwhelmingly want one
 * half, and keeping them apart means a component that only names the detector
 * doesn't take a dependency on the dataset registry.
 *
 * Deliberately *not* fields on `ActiveContextService`: that one is injected by
 * the HTTP interceptor, and pulling in registry state (which itself depends on
 * `HttpClient`) would make the interceptor's dependency graph cyclic.
 */
@Injectable({ providedIn: 'root' })
export class ActiveDatasetService {
  private readonly activeContext = inject(ActiveContextService);
  private readonly datasetState = inject(DatasetStateService);

  /** Dataset id currently loaded into the backend (`''` when none). */
  readonly activeId = toSignal(this.activeContext.datasetId$, { initialValue: '' });

  /** Dataset id the user picked, which leads {@link activeId} for as long as
   *  a context switch is still loading. */
  readonly intentId = toSignal(this.activeContext.intentDatasetId$, { initialValue: '' });

  /**
   * The id to describe: the active dataset, or — when nothing is active yet —
   * the one being switched to, so the UI can name the user's pick immediately
   * instead of blanking for the duration of the load.
   */
  readonly datasetId = computed(() => this.activeId() || this.intentId());

  /** Registry entry for {@link datasetId}: null when nothing is selected, and
   *  also while the registry fetch is still in flight. */
  readonly dataset = computed<DatasetRegistryEntry | null>(() => {
    const id = this.datasetId();
    if (!id) return null;
    return this.datasetState.datasetById().get(id) ?? null;
  });

  /** Display name of the selected dataset; `''` when unknown (nothing
   *  selected, or the registry hasn't resolved the id yet). */
  readonly datasetName = computed(() => this.dataset()?.name ?? '');
}
