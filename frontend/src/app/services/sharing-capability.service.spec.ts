import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { BehaviorSubject } from 'rxjs';

import type { AuthStatus } from '../generated/api-client/models/auth-status';
import { provideHttpTesting } from '../testing/test-providers';
import { AuthService } from './auth.service';
import { SharingCapabilityService } from './sharing-capability.service';

const CAPABILITY_URL = '/api/detectors/sharing';

const signedIn: AuthStatus = { provider: 'default', user: 'me', authenticated: true, login_required: false };

describe('SharingCapabilityService', () => {
  let http: HttpTestingController;
  let status: BehaviorSubject<AuthStatus | null>;

  beforeEach(() => {
    status = new BehaviorSubject<AuthStatus | null>(null);
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting(), { provide: AuthService, useValue: { status$: status.asObservable() } }],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('waits for auth, then fetches the capability once', () => {
    const caps = TestBed.inject(SharingCapabilityService);
    http.expectNone(CAPABILITY_URL);
    // A sign-in wall answers 401 to everything: still not.
    status.next({ provider: 'trivial', user: '', authenticated: false, login_required: true });
    http.expectNone(CAPABILITY_URL);

    status.next(signedIn);
    http.expectOne(CAPABILITY_URL).flush({
      enabled: true,
      name: 'shared_folder',
      display_name: 'Shared folder',
      icon: 'folder',
      supports_recipient_search: true,
      missing_fields: [],
    });
    expect(caps.enabled()).toBe(true);
    expect(caps.displayName()).toBe('Shared folder');
    expect(caps.icon()).toBe('folder');
    expect(caps.supportsRecipientSearch()).toBe(true);

    // A later status change (sign-out, sign-in) fetches nothing more.
    status.next({ ...signedIn, user: 'other' });
    http.expectNone(CAPABILITY_URL);
  });

  it('reports everything off when sharing is disabled', () => {
    const caps = TestBed.inject(SharingCapabilityService);
    status.next(signedIn);
    http.expectOne(CAPABILITY_URL).flush({
      enabled: false,
      name: null,
      display_name: null,
      icon: null,
      supports_recipient_search: false,
      missing_fields: [],
    });
    expect(caps.enabled()).toBe(false);
    expect(caps.displayName()).toBe('');
    expect(caps.supportsRecipientSearch()).toBe(false);
  });

  it('stays off when the server has no such route', () => {
    const caps = TestBed.inject(SharingCapabilityService);
    status.next(signedIn);
    http.expectOne(CAPABILITY_URL).flush({ message: 'Not found' }, { status: 404, statusText: 'Not Found' });
    expect(caps.enabled()).toBe(false);
  });
});
