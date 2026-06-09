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
}
