import type { LabelQuota } from '../generated/api-client/models/label-quota';

/**
 * The label quota (#4643; `vtscore/detectors/label_quota.py`): under it a
 * detector's labels give no trained detector. A detector made from a typed
 * query gives that query's own set, the text sort at its line (tier `query`,
 * #4758), from before its first label; one with no typed query gives the
 * Goods' centroid, cut by the count line at the balance (tier `centroid`,
 * #4732), and nothing before its first Good. These are the words the Tune view
 * and AutoFind's results put on it.
 */

function plural(n: number, word: string): string {
  return `${n} more ${word}${n === 1 ? '' : 's'}`;
}

/** "1 more Good and 3 more Bads": the labels a detector under the quota still owes before a trained one. */
export function labelsOwed(quota: Pick<LabelQuota, 'goods_owed' | 'bads_owed'>): string {
  const owed: string[] = [];
  if (quota.goods_owed > 0) owed.push(plural(quota.goods_owed, 'Good'));
  if (quota.bads_owed > 0) owed.push(plural(quota.bads_owed, 'Bad'));
  return owed.join(' and ');
}

/** The note under the quota: why there is no trained detector, what this one is instead, and what is owed. */
function underQuotaNote(quota: LabelQuota, servedAs: string): string {
  const owed = labelsOwed(quota);
  // The second quota (#4731): with a Good in hand, Bads alone get there too. Nothing is owed by
  // either route when the counts already meet the first (an unresolved label kept the head away).
  const dryOwed = owed && quota.n_good >= 1 ? Math.max(0, quota.dry_bad_quota - quota.n_bad) : 0;
  const routes = [owed, dryOwed > 0 ? plural(dryOwed, 'Bad') : ''].filter(Boolean).join(', or ');
  return (
    `Too few labels for a trained detector (it takes ${quota.good_quota} Goods and ${quota.bad_quota} Bads, ` +
    `or a Good and ${quota.dry_bad_quota} Bads), ` +
    `so this is ${servedAs}; the Threshold moves it.` +
    (routes ? ` ${routes} in Train give a trained one.` : '')
  );
}

/**
 * The note a view shows when the detector it was given is the Goods'
 * centroid, or `''` for any other (and before any response).
 */
export function centroidNote(quota: LabelQuota | null | undefined): string {
  if (!quota || quota.tier !== 'centroid') return '';
  return underQuotaNote(quota, "the Goods' centroid, keeping the images that stand out from the rest");
}

/**
 * The note a view shows when the detector it was given is its typed query
 * (#4758), or `''` for any other (and before any response). Past the quota it
 * is the typed query's hold (#4736) that kept the trained one away, and the
 * note says so rather than naming counts the labels already meet (#4889).
 */
export function queryNote(quota: LabelQuota | null | undefined): string {
  if (!quota || quota.tier !== 'query') return '';
  const servedAs = "the detector's typed query, keeping the images its text sort keeps in Train";
  return quota.hold_owed > 0 ? heldNote(quota, servedAs) : underQuotaNote(quota, servedAs);
}

/** The note on a typed-query detector past the quota, held on its query until the hold's labels (#4736). */
function heldNote(quota: LabelQuota, servedAs: string): string {
  const n = quota.n_good + quota.n_bad;
  const more = `${quota.hold_owed} more label${quota.hold_owed === 1 ? '' : 's'}`;
  return (
    `A detector made from a typed query keeps its query's set until it has ${quota.hold_labels} labels ` +
    `(this one has ${n}), so this is ${servedAs}; the Threshold moves it. ` +
    `${more}, Good or Bad, in Train give a trained one.`
  );
}

/** The note for whichever detector the quota gave (the typed query or the Goods' centroid), or `''` for a trained one. */
export function quotaNote(quota: LabelQuota | null | undefined): string {
  return queryNote(quota) || centroidNote(quota);
}

/**
 * The typed query a detector was made from: its `text_query`, else its first
 * text example; `''` for none. As `detector_text_query` reads it
 * (`vtscore/detectors/query_head.py`), so a detector with one can be tested and
 * run before its first label (#4758).
 */
export function typedQuery(detector: {
  text_query?: string | null;
  examples?: ReadonlyArray<{ [key: string]: unknown }> | null;
}): string {
  const text = (detector.text_query ?? '').trim();
  if (text) return text;
  for (const ex of detector.examples ?? []) {
    if (ex?.['type'] === 'text' && typeof ex['value'] === 'string' && ex['value'].trim()) return ex['value'].trim();
  }
  return '';
}
