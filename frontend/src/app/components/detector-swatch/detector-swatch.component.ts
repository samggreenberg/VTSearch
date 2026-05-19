import { Component, HostBinding, Input } from '@angular/core';
import { detectorHue } from '../../utils/detector-color';

/**
 * Small colored dot keyed to a detector's name hue. Rendered next to detector
 * names across the app to give each detector a recognisable visual handle.
 */
@Component({
  selector: 'vt-detector-swatch',
  standalone: true,
  template: '',
  styleUrl: './detector-swatch.component.scss',
})
export class DetectorSwatchComponent {
  @Input() name = '';

  @HostBinding('attr.aria-hidden') readonly ariaHidden = 'true';

  @HostBinding('style.--detector-hue')
  get hueStyle(): string {
    return String(detectorHue(this.name));
  }
}
