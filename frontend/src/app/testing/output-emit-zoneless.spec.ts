import { Component, output, signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { configureZoneless } from './zoneless-testbed';
import { settleZoneless } from './settle-resource';

/**
 * Framework canary for the zoneless migration (Phase 2.2, Recipe D).
 *
 * Several modals close themselves from a `setTimeout(() => this.close())` where
 * `close()` emits a bound `output()` (`settings-importer`, `label-importer`,
 * `settings-exporter`, …). Under zoneless, the `setTimeout` callback is unpatched
 * and does not itself schedule change detection — so the question is whether the
 * *parent's* `(closed)="..."` handler (a **bound template listener**) still runs
 * and schedules CD when the child emits from that timer.
 *
 * The official trigger list names "bound host or template listener callbacks" as
 * a notification source, so emitting through a template `(event)` binding — even
 * from an unpatched callback — should schedule the parent's CD and let its `@if`
 * gate drop the child. This spec proves it: a bare `output()` emit from a real
 * `setTimeout` removes the child from the parent DOM with no manual
 * `detectChanges`. That is why the four `setTimeout(close)` paths need no
 * Recipe-D rework — only the components' own template-bound state was signalized.
 */
@Component({
  selector: 'vt-emit-child',
  standalone: true,
  template: '<span class="child">child</span>',
})
class EmitChildComponent {
  readonly closed = output<void>();

  /** Emit `closed` from an unpatched macrotask, mimicking the modals'
   *  `setTimeout(() => this.close())` auto-close. */
  closeLater(): void {
    setTimeout(() => this.closed.emit());
  }
}

@Component({
  selector: 'vt-emit-host',
  standalone: true,
  imports: [EmitChildComponent],
  template: `
    @if (open()) {
      <vt-emit-child (closed)="open.set(false)" />
    }
  `,
})
class EmitHostComponent {
  readonly open = signal(true);
}

describe('output() emit from setTimeout (zoneless framework canary)', () => {
  let fixture: ComponentFixture<EmitHostComponent>;

  beforeEach(async () => {
    await configureZoneless({ imports: [EmitHostComponent] }).compileComponents();
    fixture = TestBed.createComponent(EmitHostComponent);
    await settleZoneless(fixture);
  });

  it('removes the child via the parent @if when the child emits from a timer', async () => {
    const child = fixture.debugElement.children[0].componentInstance as EmitChildComponent;
    expect(fixture.nativeElement.querySelector('.child')).toBeTruthy();

    child.closeLater();
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.child')).toBeFalsy();
  });
});
