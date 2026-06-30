import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { HuggingFaceAuthService, HfAuthStatus } from './huggingface-auth.service';

describe('HuggingFaceAuthService', () => {
  let service: HuggingFaceAuthService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(HuggingFaceAuthService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('refresh() GETs status and stores it', () => {
    service.refresh();
    const req = httpMock.expectOne('/api/auth/huggingface/status');
    expect(req.request.method).toBe('GET');
    const status: HfAuthStatus = { configured: true, authenticated: true, username: 'alice', scopes: 'read-repos' };
    req.flush(status);
    expect(service.status()).toEqual(status);
  });

  it('refresh() falls back to an unknown/anonymous state on error', () => {
    service.refresh();
    httpMock.expectOne('/api/auth/huggingface/status').error(new ProgressEvent('fail'));
    expect(service.status()).toEqual({ configured: false, authenticated: false, username: '', scopes: '' });
  });

  it('login() navigates the browser to the authorize URL when configured', () => {
    const navSpy = vi.fn();
    (service as unknown as { navigate: (u: string) => void }).navigate = navSpy;

    service.login();
    const req = httpMock.expectOne('/api/auth/huggingface/login');
    expect(req.request.method).toBe('GET');
    req.flush({ configured: true, authorize_url: 'https://huggingface.co/oauth/authorize?x=1' });

    expect(navSpy).toHaveBeenCalledWith('https://huggingface.co/oauth/authorize?x=1');
  });

  it('login() records not-configured without navigating', () => {
    const navSpy = vi.fn();
    (service as unknown as { navigate: (u: string) => void }).navigate = navSpy;

    service.login();
    httpMock.expectOne('/api/auth/huggingface/login').flush({ configured: false });

    expect(navSpy).not.toHaveBeenCalled();
    expect(service.status()?.configured).toBe(false);
  });

  it('logout() POSTs then refreshes status', () => {
    service.logout();
    const logoutReq = httpMock.expectOne('/api/auth/huggingface/logout');
    expect(logoutReq.request.method).toBe('POST');
    logoutReq.flush({ ok: true });

    // logout() chains a refresh().
    const statusReq = httpMock.expectOne('/api/auth/huggingface/status');
    expect(statusReq.request.method).toBe('GET');
    statusReq.flush({ configured: true, authenticated: false, username: '', scopes: '' });
    expect(service.status()?.authenticated).toBe(false);
  });
});
