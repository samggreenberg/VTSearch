import type { LabelQuota } from '../generated/api-client/models/label-quota';

/**
 * The label quota (#4643; `vtscore/detectors/label_quota.py`): under it a
 * detector's labels give the Goods' centroid, cut by the count line at the balance (#4732),
 * rather than a trained detector. These are the words the Test view and
 * AutoFind's results put on it.
 */

function plural(n: number, word: string): string {
  return `${n} more ${word}${n === 1 ? '' : 's'}`;
}

/** "1 more Good and 3 more Bads": the labels a centroid still owes before a trained detector. */
export function labelsOwed(quota: Pick<LabelQuota, 'goods_owed' | 'bads_owed'>): string {
  const owed: string[] = [];
  if (quota.goods_owed > 0) owed.push(plural(quota.goods_owed, 'Good'));
  if (quota.bads_owed > 0) owed.push(plural(quota.bads_owed, 'Bad'));
  return owed.join(' and ');
}

/**
 * The note a view shows when the detector it was given is the Goods'
 * centroid, or `''` for a trained detector (and before any response).
 */
export function centroidNote(quota: LabelQuota | null | undefined): string {
  if (!quota || quota.tier !== 'centroid') return '';
  const owed = labelsOwed(quota);
  // The second quota (#4731): with a Good in hand, Bads alone get there too. Nothing is owed by
  // either route when the counts already meet the first (an unresolved label kept the head away).
  const dryOwed = owed && quota.n_good >= 1 ? Math.max(0, quota.dry_bad_quota - quota.n_bad) : 0;
  const routes = [owed, dryOwed > 0 ? plural(dryOwed, 'Bad') : ''].filter(Boolean).join(', or ');
  return (
    `Too few labels for a trained detector (it takes ${quota.good_quota} Goods and ${quota.bad_quota} Bads, ` +
    `or a Good and ${quota.dry_bad_quota} Bads), ` +
    `so this is the Goods' centroid, keeping the images that stand out from the rest; the Threshold moves it.` +
    (routes ? ` ${routes} in Train give a trained one.` : '')
  );
}
