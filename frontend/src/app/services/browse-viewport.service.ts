import { Injectable } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';

/**
 * The current visible region of the browse canvas, in projection space, as
 * ``[xmin, ymin, xmax, ymax]``. ``null`` before the canvas has drawn once.
 */
export type ViewportBounds = [number, number, number, number] | null;

/**
 * Coordinates the browse canvas and its minimap. The canvas publishes the
 * region it is currently showing (so the minimap can draw the viewport
 * rectangle); the minimap requests a recenter (so clicking/dragging the
 * minimap pans the main view). Scoped to a single ``vt-browse-view`` via its
 * ``providers`` so multiple browse views never cross-talk.
 */
@Injectable()
export class BrowseViewportService {
  /** Latest visible region published by the canvas. */
  readonly viewport$ = new BehaviorSubject<ViewportBounds>(null);

  /** Recenter requests from the minimap → consumed by the canvas. */
  readonly recenter$ = new Subject<{ x: number; y: number }>();

  setViewport(bounds: ViewportBounds): void {
    this.viewport$.next(bounds);
  }

  requestRecenter(x: number, y: number): void {
    this.recenter$.next({ x, y });
  }

  /**
   * Arm a one-shot zoom-to-fit for the next meta the canvas receives. Used by
   * the Remove-from-Good cull: the same projection id comes back (so the
   * canvas would normally hold the user's pan/zoom), but the survivors'
   * bounds have shrunk and the old framing leaves dead space where the culled
   * cluster was — re-frame to what's left instead.
   */
  requestFitOnNextMeta(): void {
    this.pendingFit = true;
  }

  /** Consume the one-shot fit request, returning whether it was set. */
  consumeFitOnNextMeta(): boolean {
    const fit = this.pendingFit;
    this.pendingFit = false;
    return fit;
  }

  private pendingFit = false;
}
