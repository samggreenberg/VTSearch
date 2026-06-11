import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { ConnectionStateService } from '../../services/connection-state.service';

/**
 * Bottom-centre banner shown while the frontend's connection circuit
 * breaker ({@link ConnectionStateService}) is tripped. Mounted once in
 * `AppComponent`, outside the login gate so it surfaces even on the login
 * screen. Replaces the per-request "could not reach the server" toasts with
 * a single persistent surface plus a manual Retry — background polling and
 * the SSE stream stay paused until the user reconnects.
 */
@Component({
  selector: 'vt-offline-banner',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './offline-banner.component.html',
  styleUrl: './offline-banner.component.scss',
})
export class OfflineBannerComponent {
  constructor(public connection: ConnectionStateService) {}
}
