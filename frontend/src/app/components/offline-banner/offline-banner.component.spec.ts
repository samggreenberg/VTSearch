
import { HttpTestingController } from '@angular/common/http/testing';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { OfflineBannerComponent } from './offline-banner.component';
import { ConnectionStateService } from '../../services/connection-state.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * Zoneless staleness canary for the offline banner. The banner reads
 * `ConnectionStateService.status()` / `retrying()` — both signals — directly
 * in its template. This spec
 * runs under a zoneless `TestBed`, drives connectivity through the *production
 * channel* (the breaker's `recordNetworkFailure`/`recordSuccess` and the Retry
 * probe), and asserts on the rendered DOM after `settleZoneless()` with NO manual
 * `detectChanges()`. If a status flip failed to schedule CD the DOM would stay
 * stale and these assertions would fail.
 */
describe('OfflineBannerComponent (zoneless)', () => {
  let fixture: ComponentFixture<OfflineBannerComponent>;
  let connection: ConnectionStateService;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    configureZoneless({
      imports: [OfflineBannerComponent],
      providers: [...provideHttpTesting()],
    });
    fixture = TestBed.createComponent(OfflineBannerComponent);
    connection = TestBed.inject(ConnectionStateService);
    httpMock = TestBed.inject(HttpTestingController);
    await settleZoneless(fixture);
  });

  afterEach(() => httpMock.verify());

  function banner(): HTMLElement | null {
    return fixture.nativeElement.querySelector('.offline-banner');
  }

  function retryButton(): HTMLButtonElement | null {
    return fixture.nativeElement.querySelector('.offline-banner__retry');
  }

  /** Trip the breaker the way the app does: consecutive network failures. */
  async function tripOffline(): Promise<void> {
    connection.recordNetworkFailure();
    connection.recordNetworkFailure();
    connection.recordNetworkFailure();
    await settleZoneless(fixture);
  }

  it('hides the banner while online', () => {
    expect(banner()).toBeNull();
  });

  it('shows the banner when the breaker trips, with no manual detectChanges', async () => {
    await tripOffline();
    expect(banner()).not.toBeNull();
  });

  it('hides the banner again once connectivity is recorded', async () => {
    await tripOffline();
    expect(banner()).not.toBeNull();

    connection.recordSuccess();
    await settleZoneless(fixture);
    expect(banner()).toBeNull();
  });

  it('reflects the in-flight Retry probe in the button, then clears on recovery', async () => {
    await tripOffline();

    connection.retry();
    await settleZoneless(fixture);
    const button = retryButton()!;
    expect(button.textContent?.trim()).toBe('Retrying…');
    expect(button.disabled).toBe(true);

    // Any response proves the backend is reachable → back online, banner gone.
    httpMock.expectOne('/healthz').flush('ok');
    await settleZoneless(fixture);
    expect(banner()).toBeNull();
  });
});
