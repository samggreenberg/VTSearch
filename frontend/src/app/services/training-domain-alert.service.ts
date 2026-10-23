import { DestroyRef, Injectable, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Router } from '@angular/router';
import { EMPTY } from 'rxjs';
import { catchError, filter, map, switchMap } from 'rxjs/operators';

import type { FindTrainingDomainResponse } from '../generated/api-client/models/find-training-domain-response';
import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';
import { DetectorsFindApiService } from './detectors-find-api.service';
import { ToastService } from './toast.service';

/** The active pair a check was run for. */
interface Pair {
  datasetId: string;
  modelId: string;
}

/**
 * Warn when a detector meets a dataset unlike the one it was trained on, before
 * the user runs it there (#4961).
 *
 * The Tune result's **Training-domain overlap** check used to be the only word
 * on this, and it shows only once a test is done. The check itself needs no
 * scored run, so this service runs it the moment a (dataset, detector) pair
 * becomes active: `GET /api/find/training-domain` works out the training
 * dataset (the loaded one holding most of the detector's labelled examples)
 * and reports how much of the active dataset looks unlike it. A
 * *likely domain shift* verdict raises a warning toast; anything else, and any
 * failure, says nothing.
 *
 * One toast per pair per session. A pair whose check could not run (the
 * training dataset not loaded yet, say) is asked again the next time
 * it becomes active.
 */
@Injectable({ providedIn: 'root' })
export class TrainingDomainAlertService {
  private readonly activeContext = inject(ActiveContextService);
  private readonly datasetState = inject(DatasetStateService);
  private readonly findApi = inject(DetectorsFindApiService);
  private readonly toast = inject(ToastService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  /** Pairs (`<datasetId>::<detectorId>`) whose check has given its answer this session. */
  private readonly answered = new Set<string>();
  private started = false;

  /** Start watching the active pair. Safe to call more than once. */
  start(): void {
    if (this.started) return;
    this.started = true;
    this.activeContext.pair$
      .pipe(
        filter((pair) => !!pair.datasetId && !!pair.modelId && !this.answered.has(pairKey(pair))),
        // A newer pair supersedes the check still running for the last one.
        switchMap((pair) =>
          this.findApi.getTrainingDomain().pipe(
            map((report) => ({ pair, report })),
            catchError(() => EMPTY),
          ),
        ),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe(({ pair, report }) => this.onReport(pair, report));
  }

  private onReport(pair: Pair, report: FindTrainingDomainResponse): void {
    const shift = report.domain_shift;
    if (!shift) return;
    const key = pairKey(pair);
    this.answered.add(key);
    if (!shift.shifted) return;

    const reference = report.reference_dataset_name || report.reference_dataset_id;
    const detector = this.datasetState.detectorById().get(pair.modelId)?.name ?? 'This detector';
    const dataset = this.datasetState.datasetById().get(pair.datasetId)?.name ?? 'this dataset';
    const pct = Math.round(shift.frac_atypical * 100);
    const onTune = this.router.url.split('?')[0].startsWith('/tune/');
    this.toast.warning({
      // The Tune result's own line, word for word: the verdict and nothing stronger.
      message: `${pct}% of this dataset looks atypical vs ${reference}: likely domain shift`,
      detail:
        `${detector} learned from ${reference} (${report.labels_in_reference} of its ` +
        `${report.labels_total} labelled examples are there). Check its calls on ${dataset} ` +
        `by hand before relying on them.`,
      action: onTune
        ? undefined
        : {
            label: 'Tune it',
            title: `Tune ${detector} on ${dataset}`,
            onClick: () => void this.router.navigate(['/tune', pair.datasetId, pair.modelId]),
          },
      dedupKey: `training-domain:${key}`,
    });
  }
}

function pairKey(pair: Pair): string {
  return `${pair.datasetId}::${pair.modelId}`;
}
