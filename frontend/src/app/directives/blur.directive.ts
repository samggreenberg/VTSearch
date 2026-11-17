import { Directive, ElementRef, Injectable, OnDestroy, computed, effect, inject, input, signal, untracked } from '@angular/core';

import { ActiveDatasetService } from '../services/active-dataset.service';
import { BlurStateService } from '../services/blur-state.service';
import { blurRadiusFor } from '../utils/blur-radius';

/**
 * One `ResizeObserver` for every blurred image, rather than one each: a grid
 * can hold hundreds, and before a dataset's blurred set arrives on an instance
 * with blur rules, all of them are blurred at once.
 */
@Injectable({ providedIn: 'root' })
export class SharedResizeObserver implements OnDestroy {
  private readonly listeners = new Map<Element, (width: number, height: number) => void>();
  private observer: ResizeObserver | null = null;

  /** Call *listener* with *element*'s content-box size now and on every resize, until the returned stop. */
  observe(element: Element, listener: (width: number, height: number) => void): () => void {
    if (typeof ResizeObserver === 'undefined') return () => {};
    this.observer ??= new ResizeObserver((entries) => {
      for (const entry of entries) {
        this.listeners.get(entry.target)?.(entry.contentRect.width, entry.contentRect.height);
      }
    });
    this.listeners.set(element, listener);
    this.observer.observe(element);
    return () => {
      this.listeners.delete(element);
      this.observer?.unobserve(element);
    };
  }

  ngOnDestroy(): void {
    this.observer?.disconnect();
    this.listeners.clear();
  }
}

/**
 * Blurs an image whose blur state says so (#5689): `[vtBlur]="mediaId"`.
 *
 * It puts `.vt-blurred` on the host while `BlurStateService` reports the item
 * blurred, and sets `--vt-blur-radius` from the host's rendered size, so a
 * chip and a full-size image are blurred to the same degree. Everything else,
 * the hover-to-peek included, is the shared CSS: resting the pointer on the
 * image for a second fades the blur out, and leaving puts it straight back.
 *
 * The item is read from the active dataset unless `vtBlurDataset` names
 * another, as a site showing a sibling dataset's item (a face's source photo)
 * must. While mounted, the directive keeps that dataset's blurred set loaded.
 * A host that draws no item of a loaded dataset (`vtBlur` null) is sharp,
 * unless `vtBlurFallback` says otherwise: a detector's stored label outside
 * the loaded dataset carries the server's answer for it.
 *
 * The blur is clipped to the host's own box by the shared CSS, so its
 * container need not clip. The host must take the pointer (no
 * `pointer-events: none`), or it can never be peeked at; where something is
 * drawn over it, `vt-blur-peek` on the ancestor that takes the pointer peeks
 * it instead.
 */
@Directive({
  selector: '[vtBlur]',
  standalone: true,
  host: {
    // The class's rules, the peek among them, are in `scss/_components.scss`.
    '[class.vt-blurred]': 'blurred()',
    '[style.--vt-blur-radius]': 'radiusCss()',
  },
})
export class BlurDirective {
  /** The id of the item the host draws. `null` draws no dataset item, which is never blurred. */
  readonly vtBlur = input.required<number | null | undefined>();
  /** The dataset the item belongs to; the active dataset when left out. */
  readonly vtBlurDataset = input<string | null | undefined>(undefined);
  /**
   * Whether the host shows blurred when `vtBlur` is null, decided elsewhere
   * and fixed. Ignored while `vtBlur` names an item, whose state is the
   * dataset's.
   */
  readonly vtBlurFallback = input<boolean | null | undefined>(false);

  private readonly blurState = inject(BlurStateService);
  private readonly activeDataset = inject(ActiveDatasetService);
  private readonly sizes = inject(SharedResizeObserver);
  private readonly host: HTMLElement = inject(ElementRef).nativeElement;

  private readonly datasetId = computed(() => this.vtBlurDataset() || this.activeDataset.activeId());

  /** Whether the host is blurred now. */
  readonly blurred = computed(() => {
    const id = this.vtBlur();
    if (id == null) return !!this.vtBlurFallback();
    return this.blurState.isBlurred(this.datasetId(), id);
  });

  /** The radius for the host's last measured size, or `null` before it was measured. */
  private readonly radius = signal<number | null>(null);

  /** `--vt-blur-radius` while blurred and measured; unset otherwise, where the CSS falls back. */
  readonly radiusCss = computed(() => {
    const radius = this.radius();
    return this.blurred() && radius !== null ? `${radius}px` : null;
  });

  constructor() {
    effect((onCleanup) => {
      const datasetId = this.datasetId();
      onCleanup(untracked(() => this.blurState.retain(datasetId)));
    });
    // Measure only while blurred: a sharp image has no radius to keep current.
    effect((onCleanup) => {
      if (!this.blurred()) return;
      onCleanup(
        untracked(() =>
          this.sizes.observe(this.host, (width, height) => this.radius.set(blurRadiusFor(width, height))),
        ),
      );
    });
  }
}
