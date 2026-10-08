import type { LabelQuota } from '../generated/api-client/models/label-quota';

/**
 * The label quota (#4643; `vtscore/detectors/label_quota.py`): under it a
 * detector's labels give the Goods' centroid, cut where its scores split,
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
  return (
    `Too few labels for a trained detector (it takes ${quota.good_quota} Goods and ${quota.bad_quota} Bads), ` +
    `so this is the Goods' centroid, cut where its scores split; the Threshold doesn't move it.` +
    (owed ? ` ${owed} in Train give a trained one.` : '')
  );
}
