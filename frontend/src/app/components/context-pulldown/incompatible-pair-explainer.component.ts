import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { Router } from '@angular/router';
import { ActiveDatasetService } from '../../services/active-dataset.service';
import { ActiveDetectorService } from '../../services/active-detector.service';
import { PulldownControlService } from '../../services/pulldown-control.service';
import { DatasetRegistryEntry } from '../../models/api.models';
import { DetectorRegistryEntry } from '../../generated/api-client/models/detector-registry-entry';

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
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-incompatible-pair-explainer',
  standalone: true,
  imports: [],
  templateUrl: './incompatible-pair-explainer.component.html',
  styleUrl: './incompatible-pair-explainer.component.scss',
})
export class IncompatiblePairExplainerComponent {
  private router = inject(Router);
  private activeDataset = inject(ActiveDatasetService);
  private activeDetector = inject(ActiveDetectorService);
  private pulldownControl = inject(PulldownControlService);

  // Signal reads behind getters, not fields written from a subscribe.
  //
  // The old subscribe assigned plain fields on an OnPush component without
  // `markForCheck()`, which under zoneless CD schedules nothing. It repainted
  // anyway, and still would: `AppComponent` only mounts this component once
  // it has decided the pair is incompatible, so the fields are always read
  // fresh at first paint, and a later change unmounts it rather than
  // updating it in place. The pattern was one refactor away from being a
  // real staleness bug, not a bug today — reading the signal inside a
  // template-evaluated getter is tracked, so it cannot become one.
  get dataset(): DatasetRegistryEntry | null {
    return this.activeDataset.dataset();
  }

  get detector(): DetectorRegistryEntry | null {
    return this.activeDetector.detector();
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
