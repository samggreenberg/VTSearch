import type { FindTrainingDomainResponse } from '../generated/api-client/models/find-training-domain-response';

/** Tune's training-domain warning, as a toast or a note shows it: its line, and what to do about it. */
export interface DomainShiftWarning {
  message: string;
  detail: string;
}

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
 * and *dataset* are the names the detail speaks of.
 */
export function domainShiftWarning(
  report: FindTrainingDomainResponse | null | undefined,
  detector: string,
  dataset: string,
): DomainShiftWarning | null {
  const shift = report?.domain_shift;
  if (!report || !shift?.shifted) return null;
  const reference = report.reference_dataset_name || report.reference_dataset_id;
  return {
    message: domainShiftLine(shift.frac_atypical, reference),
    detail:
      `${detector} learned from ${reference} (${report.labels_in_reference} of its ` +
      `${report.labels_total} labelled examples are there). Check its calls on ${dataset} ` +
      `by hand before relying on them.`,
  };
}
