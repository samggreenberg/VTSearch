import { Signal, computed, effect, inject, signal, untracked } from '@angular/core';

import { BlurStateService } from '../services/blur-state.service';

/**
 * How long the pointer rests on a blurred focused item before the peek
 * starts: the delay of the shared `.vt-blurred:hover` rule
 * (`scss/_components.scss`), which a focused viewer times itself (see
 * {@link FocusedBlur.peeking}).
 */
export const PEEK_DELAY_MS = 1000;

/** What {@link focusedBlur} reads from the viewer that shows the item. */
export interface FocusedBlurSource {
  /** The dataset item shown, or null when the viewer shows none (nothing blurs it, no button can). */
  readonly itemId: () => number | null;
  /** The dataset the item is in: `''` when none is loaded. */
  readonly datasetId: () => string;
  /** The item's picture is on screen. The peek's wait starts only then. */
  readonly ready: () => boolean;
  /** Changes with every new picture (its URL): the peek's wait starts again for it. */
  readonly shown: () => unknown;
}

/** The blur of the item a focused viewer shows (#5689), as {@link focusedBlur} builds it. */
export interface FocusedBlur {
  /** Whether the Blur / Unblur button has an item to act on. */
  readonly blurrable: Signal<boolean>;
  /**
   * Whether the item is drawn blurred: its Blur / Unblur override, or else
   * the blur rules, as `BlurStateService` has them (and blurred while that
   * does not know yet, on an instance with rules).
   */
  readonly blurred: Signal<boolean>;
  /**
   * The pointer has rested on the blurred item for {@link PEEK_DELAY_MS}, so
   * the viewer fades its blur out; leaving drops it at once.
   *
   * Every other site peeks through the shared `:hover` rule, but a focused
   * viewer times it here, because its item changes under a pointer that
   * stays put: the user votes with the arrow keys with the pointer left on
   * the pane, and under a hover rule each next item would arrive already
   * peeked, sharp. So the wait starts again for every new picture, and
   * whenever the item is blurred again.
   */
  readonly peeking: Signal<boolean>;
  /** The pointer came onto the pane the viewer peeks from. */
  pointerEnter(): void;
  /** The pointer left that pane: the peek ends at once. */
  pointerLeave(): void;
  /**
   * The Blur / Unblur button and its `b` key: flip the item's state for
   * good, in its own dataset, so every view of it follows and the choice
   * outlasts whatever the blur rules say.
   */
  toggle(): void;
}

/**
 * The blur state, peek and button action of the item a focused viewer shows:
 * the image viewer's picture, the video player's video. Call it in an
 * injection context (a field initializer or constructor): it keeps the
 * item's dataset's blurred set loaded and polled while the viewer lives, as
 * the `vtBlur` directive does for every other site, and runs the peek's
 * timer. How the blur is drawn, and its radius, stay the viewer's.
 */
export function focusedBlur(source: FocusedBlurSource): FocusedBlur {
  const blurState = inject(BlurStateService);

  const blurrable = computed(() => source.itemId() !== null && !!source.datasetId());
  const blurred = computed(() => {
    const id = source.itemId();
    return id !== null && blurState.isBlurred(source.datasetId(), id);
  });
  const pointerInside = signal(false);
  const peeking = signal(false);

  effect((onCleanup) => {
    if (source.itemId() === null) return;
    const datasetId = source.datasetId();
    onCleanup(untracked(() => blurState.retain(datasetId)));
  });

  // The peek's wait: it runs while the pointer is on a blurred item that is
  // on screen, and starts again for each new picture and each re-blur.
  // Anything else ends the peek at once.
  effect((onCleanup) => {
    source.shown();
    const waiting = blurred() && pointerInside() && source.ready();
    peeking.set(false);
    if (!waiting) return;
    const timer = setTimeout(() => peeking.set(true), PEEK_DELAY_MS);
    onCleanup(() => clearTimeout(timer));
  });

  return {
    blurrable,
    blurred,
    peeking: peeking.asReadonly(),
    pointerEnter: () => pointerInside.set(true),
    pointerLeave: () => pointerInside.set(false),
    toggle: () => {
      const id = source.itemId();
      const datasetId = source.datasetId();
      if (id === null || !datasetId) return;
      blurState.setBlurred(datasetId, id, !blurred());
    },
  };
}
