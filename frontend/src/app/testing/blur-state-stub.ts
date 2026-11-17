import { Provider, WritableSignal, signal } from '@angular/core';

import { BlurStateService } from '../services/blur-state.service';

/** The key {@link provideBlurStateStub} files a blurred item under. */
export function blurKey(datasetId: string, id: number): string {
  return `${datasetId}:${id}`;
}

/**
 * A `BlurStateService` for a display site's spec (#5696): it answers
 * `isBlurred` from a signal of `"<datasetId>:<id>"` keys (see {@link blurKey})
 * and loads nothing, so a site's spec asserts what its `vtBlur` hosts show
 * without flushing the blur routes.
 */
export function provideBlurStateStub(initial: Iterable<string> = []): {
  provider: Provider;
  blurred: WritableSignal<ReadonlySet<string>>;
} {
  const blurred = signal<ReadonlySet<string>>(new Set(initial));
  const stub: Pick<BlurStateService, 'isBlurred' | 'retain' | 'setBlurred'> = {
    isBlurred: (datasetId, id) => blurred().has(blurKey(datasetId, id)),
    retain: () => () => {},
    setBlurred: () => {},
  };
  return { provider: { provide: BlurStateService, useValue: stub }, blurred };
}
