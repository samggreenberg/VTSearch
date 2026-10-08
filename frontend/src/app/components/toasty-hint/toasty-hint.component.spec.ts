import { ChangeDetectionStrategy, Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { RectLike, ToastyHintComponent, toastyHintPoint } from './toasty-hint.component';
import { SettingsStateService } from '../../services/settings-state.service';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../testing/test-providers';
import { settleResource, settleZoneless } from '../../testing/settle-resource';

function rect(left: number, top: number, width: number, height: number): RectLike {
  return { left, top, width, height, right: left + width, bottom: top + height };
}

describe('toastyHintPoint', () => {
  const origin = { left: 0, top: 0 };

  it('stands Toasty just below the middle of the control', () => {
    // A panel's + button, top right.
    expect(toastyHintPoint(rect(900, 20, 30, 30), origin, 'below')).toEqual({ x: 915, y: 54 });
  });

  it('stands Toasty just above the control', () => {
    // The Train button, under the panels.
    expect(toastyHintPoint(rect(380, 600, 120, 40), origin, 'above')).toEqual({ x: 440, y: 596 });
  });

  it('translates into the containing block', () => {
    expect(toastyHintPoint(rect(900, 20, 30, 30), { left: 100, top: 10 }, 'below')).toEqual({ x: 815, y: 44 });
  });

  it('places nothing for an unrendered control', () => {
    expect(toastyHintPoint(rect(0, 0, 0, 0), origin, 'below')).toBeNull();
  });
});

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  imports: [ToastyHintComponent],
  template: `
    <div class="box">
      <button #target>+</button>
      <vt-toasty-hint hintId="add-dataset" [anchor]="target">Click + to add a dataset.</vt-toasty-hint>
    </div>
  `,
})
class HostComponent {}

describe('ToastyHintComponent', () => {
  let httpMock: HttpTestingController;

  afterEach(() => {
    httpMock.verify();
    // The prototype stub below would otherwise leak into later spec files
    // (the suite runs with `isolate: false`).
    vi.restoreAllMocks();
  });

  /** Render one hint under a positioned box, with `settings` loaded first
   *  (or never, when `settings` is null). */
  async function render(settings: object | null): Promise<HTMLElement> {
    TestBed.configureTestingModule({
      imports: [HostComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    });
    httpMock = TestBed.inject(HttpTestingController);
    // jsdom does no layout, so hand the button and its box what a browser would.
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
      if (this.tagName === 'BUTTON') return rect(900, 20, 30, 30) as DOMRect;
      if (this.classList.contains('box')) return rect(100, 10, 1000, 700) as DOMRect;
      return rect(0, 0, 0, 0) as DOMRect;
    });
    if (settings) {
      TestBed.inject(SettingsStateService).load();
      TestBed.tick();
      httpMock.expectOne('/api/settings').flush(settings);
      await settleResource();
    }
    const fixture = TestBed.createComponent(HostComponent);
    await settleZoneless(fixture);
    return (fixture.nativeElement as HTMLElement).querySelector('vt-toasty-hint') as HTMLElement;
  }

  function checkbox(host: HTMLElement, label: string): HTMLInputElement {
    const row = [...host.querySelectorAll('label')].find((l) => l.textContent?.trim() === label);
    return row?.querySelector('input') as HTMLInputElement;
  }

  it('stands Toasty under the control and says what to do', async () => {
    const host = await render({});
    expect(host.classList).not.toContain('toasty-hint--off');
    expect(host.querySelector('.toasty-hint__text')?.textContent?.trim()).toBe('Click + to add a dataset.');
    expect(host.querySelector('img')?.getAttribute('src')).toBe('logo.png');
    // Under the button's middle, in the box's coordinates.
    expect(host.style.left).toBe('815px');
    expect(host.style.top).toBe('44px');
    expect(host.style.visibility).toBe('');
  });

  it('stays hidden until the settings say whether the user hid it', async () => {
    const host = await render(null);
    expect(host.classList).toContain('toasty-hint--off');
  });

  it('renders nothing for a hint the user hid, or when every hint is hidden', async () => {
    expect((await render({ hidden_hints: ['add-dataset'] })).classList).toContain('toasty-hint--off');
    httpMock.verify();
    TestBed.resetTestingModule();
    expect((await render({ hide_all_hints: true })).classList).toContain('toasty-hint--off');
  });

  it('shows a hint when only some other hint is hidden', async () => {
    const host = await render({ hidden_hints: ['train'] });
    expect(host.classList).not.toContain('toasty-hint--off');
  });

  it('"Hide this hint" hides it at once and saves it to the user\'s settings', async () => {
    const host = await render({ hidden_hints: ['train'] });
    checkbox(host, 'Hide this hint').click();
    TestBed.tick();
    // Off on the click, before the save comes back.
    expect(host.classList).toContain('toasty-hint--off');

    const req = httpMock.expectOne('/api/settings');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ hidden_hints: ['train', 'add-dataset'] });
    req.flush({ hidden_hints: ['train', 'add-dataset'] });
    TestBed.tick();
    expect(host.classList).toContain('toasty-hint--off');
  });

  it('"Hide all hints" turns every hint off', async () => {
    const host = await render({});
    checkbox(host, 'Hide all hints').click();
    const req = httpMock.expectOne('/api/settings');
    expect(req.request.body).toEqual({ hide_all_hints: true });
    TestBed.tick();
    expect(host.classList).toContain('toasty-hint--off');
    req.flush({ hide_all_hints: true });
    TestBed.tick();
    expect(host.classList).toContain('toasty-hint--off');
  });

  it('brings the hint back when hiding it fails to save', async () => {
    const host = await render({});
    checkbox(host, 'Hide this hint').click();
    httpMock.expectOne('/api/settings').flush('nope', { status: 500, statusText: 'Server Error' });
    TestBed.tick();
    expect(host.classList).not.toContain('toasty-hint--off');
  });
});
