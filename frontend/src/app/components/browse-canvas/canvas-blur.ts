/**
 * Blurred cells on the Browse canvas, and the hover peek (#5689).
 *
 * The canvas is a 2D canvas, so the shared `.vt-blurred` CSS cannot reach a
 * cell. It draws a pre-blurred copy of the thumbnail instead (built and cached
 * by `ThumbStore.blurred`), and this module holds the rest: how much to blur
 * a cell, what a cell is painted with, and the peek's clock. It is free of
 * Angular and of the canvas, so each rule is testable on its own.
 */

import { blurRadiusFor } from '../../utils/blur-radius';

/**
 * How long the pointer rests on a blurred cell before its blur fades. The
 * same second as the DOM peek (`.vt-blurred:hover` in `_components.scss`).
 */
export const PEEK_DELAY_MS = 1000;

/** How long the fade from blurred to sharp takes, as in the DOM peek. */
export const PEEK_FADE_MS = 1500;

/**
 * The blur for the canvas's cells, as a share of a thumbnail's shorter side.
 *
 * A DOM image's blur is sized from its rendered box (`blurRadiusFor`). A
 * cell's size moves with every zoom step, and rebuilding every blurred copy
 * at each step would make zooming a blurred dataset stall, so a copy is
 * blurred once, by a share of the image, and drawn at any size. The share is
 * the one `blurRadiusFor` gives the *smallest* cell the map draws for this
 * target radius: level selection keeps a cell's radius within a factor of √2
 * of the target, so every cell is blurred at least as hard as a DOM image of
 * its size would be, and the minimum radius that hides a small chip holds.
 */
export function cellBlurFraction(targetRadius: number): number {
  const side = (2 * targetRadius) / Math.SQRT2;
  if (!(side > 0)) return 0;
  return blurRadiusFor(side, side) / side;
}

/**
 * What a cell's thumbnail is painted with: `base`, then `over` on top of it at
 * `overAlpha`. `over` is only set mid-peek, as the sharp image fading in over
 * the blurred copy.
 */
export interface StillPaint<T> {
  readonly base: T;
  readonly over: T | null;
  readonly overAlpha: number;
}

/**
 * Pick what a cell's thumbnail is painted with.
 *
 * - A sharp item, or a blurred one whose peek has finished: the sharp image.
 * - A blurred item: its blurred copy, with the sharp image over it at the
 *   peek's progress while the peek fades.
 * - A blurred item whose copy is not built yet: `null`, and the cell draws as
 *   though its thumbnail had not landed. Never the sharp image.
 *
 * `blurredCopy` is only called when the copy is needed.
 */
export function stillPaint<T>(
  sharp: T,
  blurred: boolean,
  peek: number,
  blurredCopy: () => T | null,
): StillPaint<T> | null {
  if (!blurred || peek >= 1) return { base: sharp, over: null, overAlpha: 0 };
  const copy = blurredCopy();
  if (copy === null) return null;
  if (peek > 0) return { base: copy, over: sharp, overAlpha: peek };
  return { base: copy, over: null, overAlpha: 0 };
}

/** What a {@link BlurPeek} needs from its host. */
export interface BlurPeekDeps {
  /** Clock in ms, the one the fade is measured against. */
  now(): number;
  /** Paint a frame: the delay is up and the fade begins. */
  repaint(): void;
}

/** The peek's ease: quick at first, settling into sharp, like `ease-out`. */
function easeOut(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * The hover peek's clock.
 *
 * The host tells it, every frame, which blurred cell the pointer rests on
 * ({@link rest}): the hovered cell, or the pinned one while its popup is open.
 * After {@link PEEK_DELAY_MS} on the same cell the fade begins, and
 * {@link progress} climbs from 0 (blurred) to 1 (sharp) over
 * {@link PEEK_FADE_MS}. Any other answer, another cell or none, puts the cell
 * back to blurred at once and starts the clock over.
 */
export class BlurPeek {
  /** The cell the pointer rests on, or `null` when it rests on no blurred cell. */
  private key: string | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  /** When the fade began, or `null` before the delay is up. */
  private fadeFrom: number | null = null;

  constructor(private readonly deps: BlurPeekDeps) {}

  /**
   * The blurred cell the pointer rests on now, by a key naming it, or `null`
   * for none. The same key as last time carries on; anything else ends the
   * peek and, for a cell, starts the delay again.
   */
  rest(key: string | null): void {
    if (key === this.key) return;
    this.stop();
    if (key === null) return;
    this.key = key;
    this.timer = setTimeout(() => {
      this.timer = null;
      this.fadeFrom = this.deps.now();
      this.deps.repaint();
    }, PEEK_DELAY_MS);
  }

  /**
   * How far cell *key* has faded, from 0 (blurred) to 1 (sharp). Every cell
   * but the one resting is 0. `instant` (low-effects mode, or reduced motion)
   * skips the fade: the cell turns sharp when the delay is up.
   */
  progress(key: string, instant = false): number {
    if (key !== this.key || this.fadeFrom === null) return 0;
    if (instant) return 1;
    const t = (this.deps.now() - this.fadeFrom) / PEEK_FADE_MS;
    return t >= 1 ? 1 : easeOut(Math.max(0, t));
  }

  /** Whether the fade is under way, so the host owes another frame. */
  fading(instant = false): boolean {
    return !instant && this.fadeFrom !== null && this.deps.now() - this.fadeFrom < PEEK_FADE_MS;
  }

  /** End any peek: the pointer left, or the host is going away. */
  stop(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.fadeFrom = null;
    this.key = null;
  }
}
