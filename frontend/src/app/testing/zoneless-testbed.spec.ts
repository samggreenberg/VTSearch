import { Component, signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureZoneless, provideZoneless } from './zoneless-testbed';
import { settleZoneless } from './settle-resource';

/**
 * Reference spec for the zoneless test harness (docs/plans/zoneless-migration.md,
 * Phase 0.3 / 0.4). It is NOT testing app code — it proves the harness itself
 * works, and serves as the copy-me pattern for per-component canary specs in
 * Phases 1–2:
 *
 *   - run the TestBed under `provideZonelessChangeDetection()` (via
 *     `configureZoneless` / `provideZoneless`),
 *   - drive state through the production channel (here, a signal write) with NO
 *     manual `fixture.detectChanges()`,
 *   - `await settleZoneless(fixture)`,
 *   - assert on the rendered DOM.
 *
 * Because the fixture refreshes only what change detection actually *schedules*,
 * a forgotten notification surfaces as a stale-DOM assertion failure rather than
 * being masked by a forced render.
 */
@Component({
  selector: 'app-zoneless-canary',
  standalone: true,
  template: `<span class="value">{{ count() }}</span>`,
})
class ZonelessCanaryComponent {
  readonly count = signal(0);
}

describe('zoneless test harness', () => {
  function valueText(fixture: ComponentFixture<ZonelessCanaryComponent>): string | null {
    return fixture.nativeElement.querySelector('.value')?.textContent?.trim() ?? null;
  }

  it('renders the initial signal value under a zoneless TestBed', async () => {
    configureZoneless({ imports: [ZonelessCanaryComponent] });
    const fixture = TestBed.createComponent(ZonelessCanaryComponent);

    await settleZoneless(fixture);

    expect(valueText(fixture)).toBe('0');
  });

  it('repaints when a template-read signal changes, with no manual detectChanges', async () => {
    configureZoneless({ imports: [ZonelessCanaryComponent] });
    const fixture = TestBed.createComponent(ZonelessCanaryComponent);
    await settleZoneless(fixture);

    // Drive state the way the app would: write the signal the template reads.
    // A signal write read in a template schedules CD with no zone involved.
    fixture.componentInstance.count.set(42);
    await settleZoneless(fixture);

    // Assert on the DOM, not the field. If the write failed to schedule CD this
    // would still read "0" and fail — which is exactly the staleness we want to
    // catch.
    expect(valueText(fixture)).toBe('42');
  });

  it('exposes the same provider via provideZoneless() for manual module config', async () => {
    TestBed.configureTestingModule({
      imports: [ZonelessCanaryComponent],
      providers: [...provideZoneless()],
    });
    const fixture = TestBed.createComponent(ZonelessCanaryComponent);
    await settleZoneless(fixture);

    fixture.componentInstance.count.set(7);
    await settleZoneless(fixture);

    expect(valueText(fixture)).toBe('7');
  });
});
