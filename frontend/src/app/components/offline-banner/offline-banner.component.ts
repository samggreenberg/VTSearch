import { ChangeDetectionStrategy, Component, inject } from '@angular/core';
import { ConnectionStateService } from '../../services/connection-state.service';
import { ToastyHintComponent } from '../toasty-hint/toasty-hint.component';
import { TOASTY_FACES } from '../../utils/toasty-faces';

/**
 * Bottom-centre banner shown while the frontend's connection circuit
 * breaker ({@link ConnectionStateService}) is tripped. Mounted once in
 * `AppComponent`, outside the login gate so it surfaces even on the login
 * screen. Replaces the per-request "could not reach the server" toasts with
 * a single persistent surface plus a manual Retry — background polling and
 * the SSE stream stay paused until the user reconnects. Toasty stands over
 * Retry while it is up (#5149), the one hint shown while the server is gone.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-offline-banner',
  standalone: true,
  imports: [ToastyHintComponent],
  templateUrl: './offline-banner.component.html',
  styleUrl: './offline-banner.component.scss',
})
export class OfflineBannerComponent {
  connection = inject(ConnectionStateService);

  /** Toasty's sad face, fetched now while the server is up: the hint needs it
   *  only once the server is gone, and the browser shows an image this page
   *  has already loaded without asking the server for it again. */
  private readonly sadFace = Object.assign(new Image(), { src: TOASTY_FACES.sad });
}
