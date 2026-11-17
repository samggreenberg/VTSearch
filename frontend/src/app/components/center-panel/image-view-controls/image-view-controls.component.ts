import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  afterNextRender,
  inject,
  input,
  signal,
} from '@angular/core';
import { ImageViewerComponent } from '../image-viewer/image-viewer.component';
import { IconComponent } from '../../icon/icon.component';
import { BlurToggleComponent } from '../blur-toggle/blur-toggle.component';

/**
 * The image viewer's toolbar: rotate, zoom (buttons, slider and readout) and
 * reset, led by the Marquee and Highlight toggles where the host offers them,
 * and ended by Blur / Unblur (#5689) whenever the viewer shows an item of a
 * dataset. The centre panel, the Face compare layout's reference pane and the
 * Show in photo overlay all use it (#5227), so a photo opened from a face crop
 * gets the controls the ranking's image has. Every control acts through the
 * viewer, which knows which dataset its item is in.
 *
 * In a panel narrower than the toolbar, its zoom slider gives up width first
 * and the controls then wrap onto a second row, so none is cut off (#5713).
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-image-view-controls',
  standalone: true,
  imports: [IconComponent, BlurToggleComponent],
  host: { class: 'image-view-controls' },
  templateUrl: './image-view-controls.component.html',
  styleUrl: './image-view-controls.component.scss',
})
export class ImageViewControlsComponent {
  private static nextId = 0;

  /** The viewer the controls drive. */
  readonly viewer = input.required<ImageViewerComponent>();
  /** Offer the Marquee toggle: a drag on the viewer draws a region box. */
  readonly marquee = input(false);
  /** Offer the Highlight toggle: a bound embedder emits a best-match region. */
  readonly highlight = input(false);
  readonly marqueeTitle = input('');
  readonly marqueeAriaLabel = input('');

  /** Per instance, as the overlay's toolbar shares the page with the panel's. */
  readonly zoomSliderId = `ivc-zoom-${ImageViewControlsComponent.nextId++}`;

  /**
   * The separators a wrap has left at the start or end of a row, which are
   * hidden: a divider separates the controls either side of it, so one with
   * nothing beside it on its row is a stray mark. Measured whenever the
   * toolbar changes size, which is when its rows can change.
   */
  readonly edgeSeps = signal<ReadonlySet<Element>>(new Set(), { equal: sameMembers });

  constructor() {
    const host = inject<ElementRef<HTMLElement>>(ElementRef).nativeElement;
    const destroyRef = inject(DestroyRef);
    afterNextRender(() => {
      if (typeof ResizeObserver === 'undefined') return;
      const observer = new ResizeObserver(() => this.edgeSeps.set(separatorsAtRowEdge(host)));
      observer.observe(host);
      destroyRef.onDestroy(() => observer.disconnect());
    });
  }
}

/** The toolbar's separators that do not have a control beside them on both sides in their row. */
function separatorsAtRowEdge(host: HTMLElement): Set<Element> {
  const edges = new Set<Element>();
  for (const sep of host.querySelectorAll(':scope > .ivc-sep')) {
    if (!sameRow(sep, sep.previousElementSibling) || !sameRow(sep, sep.nextElementSibling)) edges.add(sep);
  }
  return edges;
}

/**
 * Whether *b* is on *a*'s row. The toolbar centres every control on its row,
 * so two on one row overlap vertically, and two on different rows, with the
 * row gap between them, do not.
 */
function sameRow(a: Element, b: Element | null): boolean {
  if (!b) return false;
  const ra = a.getBoundingClientRect();
  const rb = b.getBoundingClientRect();
  return ra.top < rb.bottom && rb.top < ra.bottom;
}

function sameMembers(a: ReadonlySet<Element>, b: ReadonlySet<Element>): boolean {
  return a.size === b.size && [...a].every((el) => b.has(el));
}
