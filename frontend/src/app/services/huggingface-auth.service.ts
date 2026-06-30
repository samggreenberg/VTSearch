import { Injectable, inject, signal } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';

import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';

/** Sign-in state for VTSearch's outbound HuggingFace Hub requests. */
export interface HfAuthStatus {
  /** Whether the operator has configured OAuth client credentials. */
  configured: boolean;
  /** Whether a (non-expired) HuggingFace token is currently held. */
  authenticated: boolean;
  /** Display name of the signed-in account (best-effort; may be empty). */
  username: string;
  /** Space-separated OAuth scopes granted. */
  scopes: string;
}

const UNKNOWN: HfAuthStatus = { configured: false, authenticated: false, username: '', scopes: '' };

/**
 * Drives the "Sign in with HuggingFace" flow that authenticates the server's
 * downloads of gated demo datasets (and gated model weights).  Talks to the
 * plain ``/api/auth/huggingface/*`` blueprint directly (those routes are not
 * part of the generated OpenAPI client).
 */
@Injectable({ providedIn: 'root' })
export class HuggingFaceAuthService {
  private http = inject(HttpClient);

  /** Current sign-in state, or null until the first fetch resolves. */
  readonly status = signal<HfAuthStatus | null>(null);

  /** Re-fetch sign-in state from the server. */
  refresh(): void {
    const ctx = new HttpContext().set(SKIP_ERROR_TOAST, true);
    this.http.get<HfAuthStatus>('/api/auth/huggingface/status', { context: ctx }).subscribe({
      next: (s) => this.status.set(s),
      error: () => this.status.set(UNKNOWN),
    });
  }

  /**
   * Begin the OAuth handshake.  When configured, navigates the browser to
   * HuggingFace's consent screen; otherwise records the not-configured state so
   * the UI can show setup guidance instead of a dead button.
   */
  login(): void {
    const ctx = new HttpContext().set(SKIP_ERROR_TOAST, true);
    this.http
      .get<{ configured: boolean; authorize_url?: string }>('/api/auth/huggingface/login', { context: ctx })
      .subscribe({
        next: (r) => {
          if (r.configured && r.authorize_url) {
            this.navigate(r.authorize_url);
          } else {
            this.status.update((s) => ({ ...(s ?? UNKNOWN), configured: false }));
          }
        },
      });
  }

  /** Sign out: drop the server-held token, then refresh state. */
  logout(): void {
    this.http.post('/api/auth/huggingface/logout', {}).subscribe({
      next: () => this.refresh(),
      error: () => this.refresh(),
    });
  }

  /** Browser navigation, isolated so unit tests can stub it. */
  protected navigate(url: string): void {
    window.location.href = url;
  }
}
