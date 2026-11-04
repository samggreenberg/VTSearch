import type { FindTrainingDomainResponse } from '../generated/api-client/models/find-training-domain-response';

/** Tune's training-domain warning, as a toast or a note shows it: its line, and what to do about it. */
export interface DomainShiftWarning {
  message: string;
  detail: string;
}

/**
 * Where the warning is shown. It picks the detail's next step, which names a
 * control the user can see from there (#5545), so the advice is a click, not
 * a chore to work out.
 *
 * - `toast`: the toast off Tune, with its **Tune it** button.
 * - `tune`: the toast on Tune itself, where the calls get checked.
 * - `results`: Find Results, with its **Browse** button under the list.
 */
export type DomainShiftSurface = 'toast' | 'tune' | 'results';

/**
 * The warning's one line, read off the check's own figure: the verdict and
 * nothing stronger. Every surface words it alike (#4961, #5501), so a user
 * who has met it in Tune recognises it in Find Results and in a run's notice.
 */
export function domainShiftLine(fracAtypical: number, reference: string): string {
  return `${Math.round(fracAtypical * 100)}% of this dataset looks atypical vs ${reference}: likely domain shift`;
}

/**
 * The warning for one detector's training-domain check on *dataset*, or `null`
 * when the check found no likely domain shift (or could not run). *detector*
 * and *dataset* are the names the detail speaks of; with no name it says
 * "This detector" / "this dataset". *surface* picks the next step.
 *
 * The detail quotes every name (#5545): a dataset called `Caltech-256 (S)` or a
 * detector called `Mammal detector` runs into the words around it otherwise.
 */
export function domainShiftWarning(
  report: FindTrainingDomainResponse | null | undefined,
  detector: string | null | undefined,
  dataset: string | null | undefined,
  surface: DomainShiftSurface,
): DomainShiftWarning | null {
  const shift = report?.domain_shift;
  if (!report || !shift?.shifted) return null;
  const reference = report.reference_dataset_name || report.reference_dataset_id;
  const learned =
    `${named(detector, 'This detector')} learned ` +
    `${share(report.labels_in_reference, report.labels_total)}from ${named(reference, 'another dataset')}.`;
  return {
    message: domainShiftLine(shift.frac_atypical, reference),
    detail: `${learned} ${nextStep(surface, named(dataset, 'this dataset'))}`,
  };
}

/** *name* in quotes, or *fallback* (which is words, not a name) when there is none. */
function named(name: string | null | undefined, fallback: string): string {
  return name ? `"${name}"` : fallback;
}

/**
 * How much of what the detector learned came from the reference, as a word
 * before "from": nothing when it all did, else `mostly` or `partly`. The
 * reference is the dataset holding the most of its examples, not always a
 * majority of them.
 */
function share(inReference: number, total: number): string {
  if (total <= 0 || inReference >= total) return '';
  return inReference * 2 > total ? 'mostly ' : 'partly ';
}

/** What to do about it, naming the control *surface* offers for it. */
function nextStep(surface: DomainShiftSurface, dataset: string): string {
  switch (surface) {
    case 'toast':
      return `Tune it to check its calls on ${dataset}.`;
    case 'tune':
      return `Check its calls on ${dataset} here, on the Autopilot tab.`;
    case 'results':
      return 'Check its matches with Browse, or test it in Tune.';
  }
}
