import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

import { ModalComponent } from '../../modal/modal.component';

/**
 * The hand-off shown once, when Autopilot reaches a terminal phase.
 *
 * This replaces the old auto-return toast. Training finishing is news, but
 * whether the user is finished is not the app's call: plenty of people keep
 * labeling well past the point where the quality indicators go green, and the
 * toast's countdown made staying the outcome they had to fight for. Here both
 * ways out are plain buttons, nothing happens on its own, and the dialog opens
 * only for the run that actually did the training (see
 * ``AutopilotStateService.shouldAnnounceCompletion``).
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-autopilot-complete-modal',
  standalone: true,
  imports: [ModalComponent],
  templateUrl: './autopilot-complete-modal.component.html',
  styleUrl: './autopilot-complete-modal.component.scss',
})
export class AutopilotCompleteModalComponent {
  /** Dialog title: what just finished. */
  readonly heading = input('Detector Trained');
  /** Why Autopilot stopped. */
  readonly detail = input('');
  /** What the user can do now, whichever button they pick. */
  readonly nextSteps = input('');
  /** Label for the "don't go anywhere" button. */
  readonly stayLabel = input('Continue Training');

  /** Stay in the Train window. */
  readonly stay = output<void>();
  /** Leave for the Dashboard. */
  readonly goToDashboard = output<void>();
}
