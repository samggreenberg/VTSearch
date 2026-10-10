import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { CollapsePhotosService } from '../../../services/collapse-photos.service';
import { IconComponent } from '../../icon/icon.component';

/**
 * Faces | Photos: Collapse to photos' switch (#4751), in the ranking's header.
 * Shown only where it can collapse (`CollapsePhotosService.offered`).
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-collapse-photos-toggle',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './collapse-photos-toggle.component.html',
})
export class CollapsePhotosToggleComponent {
  readonly collapse = inject(CollapsePhotosService);
}
