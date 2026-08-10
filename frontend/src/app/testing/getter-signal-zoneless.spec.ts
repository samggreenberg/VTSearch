import { Component, Injectable, signal, inject } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { configureZoneless } from './zoneless-testbed';
import { settleZoneless } from './settle-resource';

/**
 * Harness canary for the service-signalization pattern: SortStateService /
 * VoteStateService expose their state through *value-returning getters* over
 * private signals, so existing
 * `sortState.sortBusy` template bindings stay byte-for-byte the same yet must
 * become reactive under zoneless. This pins the load-bearing framework
 * guarantee: a signal read THROUGH A GETTER, during template evaluation, is
 * tracked as a dependency of that view, so a setter write repaints it with no
 * manual change detection. If a future Angular bump regressed that, this fails.
 */
@Injectable()
class GetterSignalService {
  private readonly _value = signal('initial');
  get value(): string {
    return this._value();
  }
  setValue(v: string): void {
    this._value.set(v);
  }
}

@Component({
  selector: 'app-getter-signal-canary',
  standalone: true,
  template: `<span class="value">{{ svc.value }}</span>`,
})
class GetterSignalCanaryComponent {
  readonly svc = inject(GetterSignalService);
}

describe('zoneless: signal read through a getter in a template', () => {
  function valueText(fixture: ComponentFixture<GetterSignalCanaryComponent>): string | null {
    return fixture.nativeElement.querySelector('.value')?.textContent?.trim() ?? null;
  }

  it('repaints when the backing signal changes via the service setter, no manual CD', async () => {
    configureZoneless({
      imports: [GetterSignalCanaryComponent],
      providers: [GetterSignalService],
    });
    const fixture = TestBed.createComponent(GetterSignalCanaryComponent);
    await settleZoneless(fixture);
    expect(valueText(fixture)).toBe('initial');

    // Mutate the signal through the value-returning getter's setter — exactly the
    // SortStateService.setSortStatus(...) shape. If a getter-wrapped signal read
    // were NOT tracked, the DOM would stay "initial" and this would fail.
    fixture.componentInstance.svc.setValue('updated');
    await settleZoneless(fixture);
    expect(valueText(fixture)).toBe('updated');
  });
});
