import { TestBed } from '@angular/core/testing';
import { Observable, of, throwError } from 'rxjs';
import { BuildSkewService } from './build-skew.service';
import { SettingsApiService } from './settings-api.service';
import { ToastService } from './toast.service';

/**
 * The bundle stamp is baked in at build time, so a spec can't choose it. Every
 * case here is expressed relative to the stamp the suite was built with:
 * `sameAsBundle` for agreement, `OTHER` for a stamp that cannot equal it.
 */
describe('BuildSkewService', () => {
  const OTHER = '1999-01-01T00:00:00Z';
  const UNKNOWN = '0.0.0-unknown';

  let toast: {
    warning: ReturnType<typeof vi.fn>;
    error: ReturnType<typeof vi.fn>;
    success: ReturnType<typeof vi.fn>;
  };
  let versionResponse: () => Observable<{ version: string }>;

  function makeService(): BuildSkewService {
    toast = { warning: vi.fn(), error: vi.fn(), success: vi.fn() };
    TestBed.configureTestingModule({
      providers: [
        BuildSkewService,
        { provide: ToastService, useValue: toast },
        { provide: SettingsApiService, useValue: { getVersion: () => versionResponse() } },
      ],
    });
    return TestBed.inject(BuildSkewService);
  }

  afterEach(() => TestBed.resetTestingModule());

  it('stays quiet when the server matches the bundle', () => {
    const probe = makeService();
    const sameAsBundle = probe.bundleVersion;
    versionResponse = () => of({ version: sameAsBundle });
    probe.check();
    expect(probe.skewed()).toBe(false);
    expect(toast.warning).not.toHaveBeenCalled();
  });

  it('toasts when the server disagrees with the bundle', () => {
    const service = makeService();
    versionResponse = () => of({ version: OTHER });
    service.check();
    expect(service.skewed()).toBe(true);
    expect(toast.warning).toHaveBeenCalledTimes(1);
    const opts = toast.warning.mock.calls[0][0];
    expect(opts.message).toContain('out-of-date build');
    // Both stamps must be in the detail: the whole point is telling the user
    // which two things disagree, and the rebuild command that fixes it.
    expect(opts.detail).toContain(OTHER);
    expect(opts.detail).toContain(service.bundleVersion);
    expect(opts.detail).toContain('npm run build:prod');
    // Deduped so a reload storm can't stack five identical warnings.
    expect(opts.dedupKey).toBe('build-skew');
  });

  it('never auto-dismisses the warning', () => {
    const service = makeService();
    versionResponse = () => of({ version: OTHER });
    service.check();
    // It must route through warning(), which stays up until dismissed — not
    // through success()/info(), whose 5s timer would take the explanation away
    // mid-read. The notice stays true until the user rebuilds.
    expect(toast.warning).toHaveBeenCalledTimes(1);
    expect(toast.success).not.toHaveBeenCalled();
    // And it must not hand back an override that would re-introduce a timer.
    expect(toast.warning.mock.calls[0][0].autoDismissMs).toBeUndefined();
  });

  it('says nothing when either side cannot identify its own commit', () => {
    const service = makeService();
    versionResponse = () => of({ version: UNKNOWN });
    service.check();
    // A source checkout served by a Docker-built package is a normal setup;
    // "unknown vs <stamp>" is noise, not a finding.
    expect(service.skewed()).toBe(false);
    expect(toast.warning).not.toHaveBeenCalled();
  });

  it('says nothing when the server reports a blank version', () => {
    const service = makeService();
    versionResponse = () => of({ version: '   ' });
    service.check();
    expect(service.skewed()).toBe(false);
    expect(toast.warning).not.toHaveBeenCalled();
  });

  it('swallows a failed version request', () => {
    const service = makeService();
    versionResponse = () => throwError(() => new Error('offline'));
    expect(() => service.check()).not.toThrow();
    expect(toast.warning).not.toHaveBeenCalled();
    expect(service.skewed()).toBe(false);
  });
});
