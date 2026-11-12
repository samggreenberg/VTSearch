import { NgTemplateOutlet } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, input, signal } from '@angular/core';

import { FaceRect, FaceView, ViewLink, sameFaceView } from '../image-viewer/face-link';
import { FaceRegion, sharedViews } from './face-regions';

/** One thing to compare between the two faces: a part, what to look at in it,
 *  and where on a face its row zooms to. */
export interface FaceCheck {
  readonly part: string;
  readonly look: string;
  readonly region: FaceRegion;
}

/** A row as drawn: `views` is null while the row is plain text (the panes are
 *  not linked), and empty when either image lacks the part, which greys it. */
export interface FaceCheckRow {
  readonly check: FaceCheck;
  readonly views: readonly FaceRect[] | null;
  readonly title: string;
}

/** The view a row last put on screen, to tell whether it is still there. */
interface ShownView {
  readonly part: string;
  readonly index: number;
  readonly view: FaceView;
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
  { part: 'Marks', look: 'Moles, scars, freckles, birthmarks. In the same place on both?', region: 'marks' },
  { part: 'Ears', look: 'Lobe attached or detached, rim shape, how far they stand out.', region: 'ears' },
  { part: 'Nose', look: 'Tip, nostrils, bridge.', region: 'nose' },
  { part: 'Eyes and brows', look: 'Eyelid folds, brow shape (not how the brows are groomed).', region: 'eyes' },
  { part: 'Mouth, chin, face outline', look: 'The weakest evidence: expression moves them.', region: 'lower' },
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
 * While the two panes are linked ({@link link}, #5647) a row is a button that
 * zooms both to its part, located from the faces' landmarks
 * (`face-regions.ts`). Pressing it again steps to the part's next view (the
 * other ear) and, after the last, back out to the whole face. A row whose part
 * is off either image is greyed. With the panes not linked (a face without
 * landmarks, or the Wider view) the rows are the plain text they were.
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
  imports: [NgTemplateOutlet],
  templateUrl: './face-checklist.component.html',
  styleUrl: './face-checklist.component.scss',
})
export class FaceChecklistComponent {
  /** The face being voted on is a hard view; null when there is no verdict. */
  readonly candidateHardView = input<boolean | null>(null);
  /** The reference on screen is a hard view; null when there is no verdict, or no reference. */
  readonly referenceHardView = input<boolean | null>(null);

  /** The compare layout's link between the two panes, which the rows zoom through. */
  readonly link = input<ViewLink | null>(null);

  /** The rows, each with the views its part has on both faces right now. */
  readonly rows = computed<readonly FaceCheckRow[]>(() => {
    const faces = this.link()?.faces() ?? null;
    return FACE_CHECKS.map((check) => {
      const views = faces ? sharedViews(faces.lead, faces.follower, check.region) : null;
      return { check, views, title: rowTitle(views) };
    });
  });

  private readonly shown = signal<ShownView | null>(null);

  /**
   * The view a row put on screen, while it is still what the panes show: a
   * zoom, a pan or the next face moves the lead's view off it, and the row is
   * no longer the one in view.
   */
  private readonly inView = computed<ShownView | null>(() => {
    const shown = this.shown();
    const state = this.link()?.lead()?.leadState();
    return shown && state?.kind === 'on' && sameFaceView(state.view, shown.view) ? shown : null;
  });

  /** The part whose row is in view, for its pressed state. */
  readonly partInView = computed(() => this.inView()?.part ?? null);

  /** Zoom both panes to *row*'s part: its first view, the next on a repeat,
   *  and the whole face again after the last. */
  zoomTo(row: FaceCheckRow): void {
    const lead = this.link()?.lead();
    if (!lead || !row.views?.length) return;
    const inView = this.inView();
    const index = inView?.part === row.check.part ? inView.index + 1 : 0;
    if (index >= row.views.length) {
      lead.homeView();
      this.shown.set(null);
      return;
    }
    lead.showFaceRect(row.views[index]);
    // What the lead settled on, which its zoom range and its image's edges
    // may have kept short of the view asked for.
    const state = lead.leadState();
    this.shown.set(state.kind === 'on' ? { part: row.check.part, index, view: state.view } : null);
  }

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

function rowTitle(views: readonly FaceRect[] | null): string {
  if (!views) return '';
  if (!views.length) return 'Not in view on both faces';
  return views.length > 1
    ? 'Zoom both faces here. Click again for the other side, then to zoom back out.'
    : 'Zoom both faces here. Click again to zoom back out.';
}
