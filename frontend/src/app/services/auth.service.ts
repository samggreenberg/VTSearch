import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';
import { tap } from 'rxjs/operators';

export interface AuthStatus {
  provider: string;
  user: string;
  authenticated: boolean;
  login_required: boolean;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  private statusSubject = new BehaviorSubject<AuthStatus | null>(null);
  status$ = this.statusSubject.asObservable();

  /** True once the initial /api/auth/status call has completed. */
  private readySubject = new BehaviorSubject<boolean>(false);
  ready$ = this.readySubject.asObservable();

  constructor(private http: HttpClient) {}

  /** Fetch auth status from the server.  Called once at app startup. */
  checkStatus(): void {
    this.http.get<AuthStatus>('/api/auth/status').subscribe({
      next: (status) => {
        this.statusSubject.next(status);
        this.readySubject.next(true);
      },
      error: () => {
        // If the endpoint fails, assume no login required (backwards compat).
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
    return this.http
      .post<AuthStatus>('/api/auth/login', { username })
      .pipe(tap((status) => this.statusSubject.next(status)));
  }

  logout(): Observable<AuthStatus> {
    return this.http
      .post<AuthStatus>('/api/auth/logout', {})
      .pipe(tap((status) => this.statusSubject.next(status)));
  }
}
