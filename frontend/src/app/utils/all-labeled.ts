/**
 * Does every item in `items` carry a good or a bad label?
 *
 * The "nothing left here" question, asked over a list of media rather than over
 * a sort window — which is what the centre pane's dataset-exhausted message and
 * the left panel's "all labeled" chip both need (#4028). Both read the same two
 * vote sets, so the rule lives in one place rather than being spelled twice.
 *
 * `false` for an empty list: nothing loaded is not the same fact as nothing
 * left, and the two want different messages.
 *
 * The size comparison is a cheap reject, not the answer. Votes can name items
 * outside the list — a labelset carries elements from wherever they were
 * labeled — so a full count never proves coverage, but a short one disproves
 * it, which keeps the common case O(1) instead of a scan per vote.
 */
export function allItemsLabeled(
  items: readonly { id: number }[],
  good: ReadonlySet<number>,
  bad: ReadonlySet<number>,
): boolean {
  if (items.length === 0) return false;
  if (good.size + bad.size < items.length) return false;
  return items.every((m) => good.has(m.id) || bad.has(m.id));
}
