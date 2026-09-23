import type { SortedItem } from '../services/sort-state.service';

/** Which face of the Find cutoff a boundary walk serves. */
export type FindSide = 'above' | 'below';

export interface BoundaryPick {
  id: number;
  /** The side the pick came from — the walk serves the other one next. */
  took: FindSide;
}

/**
 * Find's "just sit and vote" advance: the unverified item nearest the cutoff
 * on `side`, falling back to the other side when that one is exhausted. `null`
 * when no unverified item remains on either side — the done state.
 *
 * `order` is descending by score, so the unverified item closest above the
 * line is the *lowest* one still ≥ threshold (keep overwriting as we descend);
 * the closest below is the *highest* one < threshold (the first sub-threshold
 * item we hit). One pass finds both.
 *
 * Pure so the image prefetch (#3896) can ask what the next advances will be
 * without taking them, through the same rule `FindViewComponent` applies.
 */
export function pickBoundaryNext(
  order: readonly SortedItem[],
  threshold: number,
  isVerified: (id: number) => boolean,
  side: FindSide,
): BoundaryPick | null {
  let closestAbove: number | null = null;
  let closestBelow: number | null = null;
  for (const item of order) {
    if (isVerified(item.id)) continue;
    if (item.score >= threshold) {
      closestAbove = item.id;
    } else if (closestBelow == null) {
      closestBelow = item.id;
    }
  }
  const [first, second]: [FindSide, FindSide] = side === 'above' ? ['above', 'below'] : ['below', 'above'];
  const byside = { above: closestAbove, below: closestBelow };
  if (byside[first] != null) return { id: byside[first]!, took: first };
  if (byside[second] != null) return { id: byside[second]!, took: second };
  return null;
}

/**
 * The next `depth` items the boundary walk would show if the reviewer verified
 * `currentId` and then each pick in turn, with the ranking unchanged. Mirrors
 * the walk's side alternation: each step serves the face opposite the one the
 * previous step took.
 */
export function peekBoundaryQueue(
  order: readonly SortedItem[] | null,
  threshold: number | null,
  verified: ReadonlySet<number>,
  side: FindSide,
  currentId: number | null,
  depth: number,
): number[] {
  if (!order || threshold == null || currentId === null) return [];
  const passed = new Set<number>([currentId]);
  const isVerified = (id: number): boolean => passed.has(id) || verified.has(id);
  const upcoming: number[] = [];
  let next = side;
  while (upcoming.length < depth) {
    const pick = pickBoundaryNext(order, threshold, isVerified, next);
    if (!pick) break;
    upcoming.push(pick.id);
    passed.add(pick.id);
    next = pick.took === 'above' ? 'below' : 'above';
  }
  return upcoming;
}
