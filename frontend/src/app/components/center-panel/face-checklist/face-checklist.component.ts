import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

/** One thing to compare between the two faces: a part, and what to look at in it. */
export interface FaceCheck {
  readonly part: string;
  readonly look: string;
}

/**
 * What to compare, most telling first (#5643; `docs/plans/face-vote-helper.md`).
 *
 * The order is the research's, not the face's: Towler et al. found ears and
 * facial marks the features whose similarity best predicted an examiner's
 * right answer, and face shape and mouth the worst; the ENFSI manual adds the
 * nose. The words are the FISWG feature list's (v2.0), cut from its 19
 * components to five a vote can afford: its skin marks are "freckles, moles
 * ... birth marks", its ear lobule is "attached or detached", its ear helix is
 * the rim, and its nasal root is the bridge.
 */
export const FACE_CHECKS: readonly FaceCheck[] = [
  { part: 'Marks', look: 'Moles, scars, freckles, birthmarks. In the same place on both?' },
  { part: 'Ears', look: 'Lobe attached or detached, rim shape, how far they stand out.' },
  { part: 'Nose', look: 'Tip, nostrils, bridge.' },
  { part: 'Eyes and brows', look: 'Eyelid folds, brow shape (not how the brows are groomed).' },
  { part: 'Mouth, chin, face outline', look: 'The weakest evidence: expression moves them.' },
];

/** The kind of face the fitness notice describes, for its wording. */
const HARD_VIEW = 'turned, covered or small';

/**
 * The checklist strip of the Face compare layout (#5643): five things to
 * compare between the reference and the candidate, most telling first, and
 * what not to weigh.
 *
 * It is a prompt, not a form. Nothing is ticked off and nothing is recorded:
 * the research supports pointing attention at the telling parts, and a form
 * would slow every vote.
 *
 * Under the rows sits a fitness notice when either face is a hard view
 * (`HARD_VIEW_QUALITY`, #5372), so the user expects a weaker comparison. It
 * offers no way past the vote: a vote is Good or Bad, and the app has no
 * other answer to give.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-face-checklist',
  standalone: true,
  templateUrl: './face-checklist.component.html',
  styleUrl: './face-checklist.component.scss',
})
export class FaceChecklistComponent {
  /** The face being voted on is a hard view; null when there is no verdict. */
  readonly candidateHardView = input<boolean | null>(null);
  /** The reference on screen is a hard view; null when there is no verdict, or no reference. */
  readonly referenceHardView = input<boolean | null>(null);

  readonly checks = FACE_CHECKS;

  /** The fitness notice's one line, or null when neither face is a hard view. */
  readonly fitness = computed<string | null>(() => {
    const candidate = this.candidateHardView() === true;
    const reference = this.referenceHardView() === true;
    if (candidate && reference) return `Both faces are hard views (${HARD_VIEW}): expect a weaker comparison.`;
    if (candidate) return `This face is a hard view (${HARD_VIEW}): expect a weaker comparison.`;
    if (reference) return `The reference is a hard view (${HARD_VIEW}): expect a weaker comparison.`;
    return null;
  });
}
