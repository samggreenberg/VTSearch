import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { Subject, combineLatest } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { DatasetStateService } from '../../services/dataset-state.service';
import { ActiveContextService } from '../../services/active-context.service';
import { PulldownControlService } from '../../services/pulldown-control.service';
import { DatasetRegistryEntry, DetectorRegistryEntry } from '../../models/api.models';

/**
 * Renders in place of `/label` and `/find` content when the active
 * dataset/detector pair is incompatible (different media types, or one
 * half unset). Tells the user what's wrong and offers two ways out:
 * fix the other half via its pulldown, or go to the Dashboard.
 *
 * Visibility is governed by the parent (`AppComponent`); this component
 * just renders the explanation given the current active pair.
 */
@Component({
  selector: 'vt-incompatible-pair-explainer',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './incompatible-pair-explainer.component.html',
  styleUrl: './incompatible-pair-explainer.component.scss',
})
export class IncompatiblePairExplainerComponent implements OnInit, OnDestroy {
  dataset: DatasetRegistryEntry | null = null;
  detector: DetectorRegistryEntry | null = null;

  private destroy$ = new Subject<void>();

  constructor(
    private router: Router,
    private activeContext: ActiveContextService,
    private datasetState: DatasetStateService,
    private pulldownControl: PulldownControlService,
  ) {}

  ngOnInit(): void {
    combineLatest([
      this.activeContext.pair$,
      this.datasetState.datasets$,
      this.datasetState.detectors$,
    ])
      .pipe(takeUntil(this.destroy$))
      .subscribe(([pair, datasets, detectors]) => {
        this.dataset = pair.datasetId
          ? datasets.find((d) => d.id === pair.datasetId) ?? null
          : null;
        this.detector = pair.modelId
          ? detectors.find((d) => d.id === pair.modelId) ?? null
          : null;
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get bothMissing(): boolean {
    return !this.dataset && !this.detector;
  }

  get datasetMissing(): boolean {
    return !this.dataset && !!this.detector;
  }

  get detectorMissing(): boolean {
    return !!this.dataset && !this.detector;
  }

  get mediaTypeMismatch(): boolean {
    return (
      !!this.dataset &&
      !!this.detector &&
      this.dataset.media_type !== this.detector.media_type
    );
  }

  /** Which half the primary "Pick…" button should focus.
   *
   *  - `bothMissing` / `datasetMissing` → pick a dataset.
   *  - `detectorMissing` → pick a detector.
   *  - `mediaTypeMismatch` → default to swapping the detector (matches
   *    the design wording "Pick a compatible detector"); the user can
   *    click the dataset pulldown directly if they'd rather swap the
   *    other half.
   */
  get pickHalf(): 'dataset' | 'detector' {
    if (this.detectorMissing || this.mediaTypeMismatch) return 'detector';
    return 'dataset';
  }

  get pickButtonLabel(): string {
    return this.pickHalf === 'detector' ? 'Pick a compatible detector' : 'Pick a dataset';
  }

  openOtherPulldown(): void {
    this.pulldownControl.requestOpen(this.pickHalf);
  }

  goToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }
}
