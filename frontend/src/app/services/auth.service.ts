import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';
import { map, tap } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { AuthStatus } from '../generated/api-client/models/auth-status';
import { apiAuthLoginPost } from '../generated/api-client/fn/auth/api-auth-login-post';
import { apiAuthLogoutPost } from '../generated/api-client/fn/auth/api-auth-logout-post';
import { apiAuthStatusGet } from '../generated/api-client/fn/auth/api-auth-status-get';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  private statusSubject = new BehaviorSubject<AuthStatus | null>(null);
  status$ = this.statusSubject.asObservable();

  /** True once the initial /api/auth/status call has completed. */
  private readySubject = new BehaviorSubject<boolean>(false);
  ready$ = this.readySubject.asObservable();

  /** Fetch auth status from the server.  Called once at app startup. */
  checkStatus(): void {
    // Skip the global error banner: auth-status failures are handled inline
    // (we fall back to "no login required") and would otherwise pop up a
    // banner every time the user opens the app while offline.
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    apiAuthStatusGet(this.http, this.config.rootUrl, undefined, context)
      .pipe(map((r) => r.body))
      .subscribe({
        next: (status) => {
          this.statusSubject.next(status);
          this.readySubject.next(true);
        },
        error: () => {
          // If the endpoint fails, assume no login required.
          this.statusSubject.next({
            provider: 'default',
            user: 'default',
            authenticated: true,
            login_required: false,
          });
          this.readySubject.next(true);
        },
      });
  }

  /** Whether the user needs to log in before using the app. */
  get needsLogin(): boolean {
    const s = this.statusSubject.value;
    return s != null && s.login_required && !s.authenticated;
  }

  login(username: string): Observable<AuthStatus> {
    // Skip the global error banner: the login form renders its own
    // inline error message, and a duplicate banner would be redundant.
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    return apiAuthLoginPost(this.http, this.config.rootUrl, { body: { username } }, context).pipe(
      map((r) => r.body),
      tap((status) => this.statusSubject.next(status)),
    );
  }

  logout(): Observable<AuthStatus> {
    return apiAuthLogoutPost(this.http, this.config.rootUrl).pipe(
      map((r) => r.body),
      tap((status) => this.statusSubject.next(status)),
    );
  }
}
