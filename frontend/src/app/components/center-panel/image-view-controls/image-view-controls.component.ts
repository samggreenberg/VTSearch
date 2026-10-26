import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { ImageViewerComponent } from '../image-viewer/image-viewer.component';
import { IconComponent } from '../../icon/icon.component';

/**
 * The image viewer's toolbar: rotate, zoom (buttons, slider and readout) and
 * reset, led by the Marquee and Highlight toggles where the host offers them.
 * The centre panel and the Show in photo overlay both use it (#5227), so a
 * photo opened from a face crop gets the controls the ranking's image has.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-image-view-controls',
  standalone: true,
  imports: [IconComponent],
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
}
