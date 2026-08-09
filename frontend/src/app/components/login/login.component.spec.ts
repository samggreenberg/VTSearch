import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';

import { LoginComponent } from './login.component';
import { AuthService } from '../../services/auth.service';
import type { AuthStatus } from '../../generated/api-client/models/auth-status';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('LoginComponent', () => {
  let component: LoginComponent;
  let fixture: ComponentFixture<LoginComponent>;
  /** Controllable stand-in for the login request. */
  let login$: Subject<AuthStatus>;
  let requestedNames: string[];

  beforeEach(async () => {
    login$ = new Subject<AuthStatus>();
    requestedNames = [];

    await configureZoneless({
      imports: [LoginComponent],
      providers: [
        {
          provide: AuthService,
          useValue: {
            login: (name: string) => {
              requestedNames.push(name);
              return login$.asObservable();
            },
          },
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(LoginComponent);
    component = fixture.componentInstance;
  });

  const input = () => fixture.nativeElement.querySelector('#login-username') as HTMLInputElement;
  const submitBtn = () =>
    fixture.nativeElement.querySelector('button[type="submit"]') as HTMLButtonElement;
  const errorEl = () => fixture.nativeElement.querySelector('.login-error') as HTMLElement | null;

  /** Type a username through ngModel and submit the form. */
  async function submitAs(name: string): Promise<void> {
    const el = input();
    el.value = name;
    el.dispatchEvent(new Event('input'));
    await settleZoneless(fixture);
    fixture.nativeElement.querySelector('form')!.dispatchEvent(new Event('submit'));
    await settleZoneless(fixture);
  }

  it('rejects an empty username without calling the service', async () => {
    await fixture.whenStable();
    await submitAs('   ');

    expect(requestedNames).toEqual([]);
    expect(errorEl()?.textContent).toContain('Please enter a username.');
  });

  it('disables the form while the login request is in flight', async () => {
    await fixture.whenStable();
    await submitAs('alice');

    expect(requestedNames).toEqual(['alice']);
    expect(submitBtn().textContent).toContain('Logging in…');
    expect(submitBtn().disabled).toBe(true);
    expect(input().disabled).toBe(true);
  });

  // Zoneless staleness canary: the failure state is written from the login
  // subscribe's error callback, and the success-only `loggedIn` output is the
  // component's sole other CD trigger. With plain fields the form stayed
  // rendered as "Logging in…" with the input AND button disabled — leaving no
  // element that could fire an event to un-stick it, so the login screen was
  // bricked until a page reload.
  it('re-enables the form and shows the message when login fails (zoneless canary)', async () => {
    await fixture.whenStable();
    await submitAs('alice');

    login$.error({ status: 401, error: { message: 'Unknown user.' } });
    await settleZoneless(fixture);

    expect(errorEl()?.textContent).toContain('Unknown user.');
    expect(submitBtn().textContent).toContain('Log in');
    expect(submitBtn().disabled).toBe(false);
    expect(input().disabled).toBe(false);
  });

  it('falls back to a generic message when the server sends no body', async () => {
    await fixture.whenStable();
    await submitAs('alice');

    login$.error({ status: 0 });
    await settleZoneless(fixture);

    expect(errorEl()?.textContent).toContain('Login failed.');
  });

  it('emits loggedIn on success', async () => {
    await fixture.whenStable();
    let emitted = false;
    component.loggedIn.subscribe(() => (emitted = true));
    await submitAs('alice');

    login$.next({
      provider: 'default',
      user: 'alice',
      authenticated: true,
      login_required: true,
    });
    await settleZoneless(fixture);

    expect(emitted).toBe(true);
    expect(component.busy()).toBe(false);
  });
});
