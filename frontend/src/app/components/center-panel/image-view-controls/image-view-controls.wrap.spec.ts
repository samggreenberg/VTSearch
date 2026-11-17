import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { HttpTestingController } from '@angular/common/http/testing';

import { ImageViewControlsComponent } from './image-view-controls.component';
import { ImageViewerComponent } from '../image-viewer/image-viewer.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { Media } from '../../../models/api.models';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';
import { settleZoneless } from '../../../testing/settle-resource';
import { makeSettingsStateStub } from '../../../testing/mocks';

/** The viewer and its toolbar, with Marquee offered so both separators show. */
@Component({
  standalone: true,
  imports: [ImageViewerComponent, ImageViewControlsComponent],
  template: `
    <vt-image-viewer #v [media]="media()" />
    <vt-image-view-controls [viewer]="v" [marquee]="true" />
  `,
})
class HostComponent {
  readonly media = signal<Media>({ id: 4, media_type: 'image', filename: '4.jpg', md5: 'm4', custom_metadata: {} });
}

/** A ResizeObserver that records what it observes and lets the spec say it fired. */
class RecordingResizeObserver {
  static last: RecordingResizeObserver | null = null;
  readonly observed = new Set<Element>();
  constructor(private readonly callback: ResizeObserverCallback) {
    RecordingResizeObserver.last = this;
  }
  observe(element: Element): void {
    this.observed.add(element);
  }
  unobserve(element: Element): void {
    this.observed.delete(element);
  }
  disconnect(): void {
    this.observed.clear();
  }
  fire(): void {
    this.callback([], this as unknown as ResizeObserver);
  }
}

/**
 * The toolbar in a panel narrower than it (#5713): it wraps, and a separator
 * the wrap leaves at the start or end of a row is hidden. Whether it wraps,
 * and where, is the browser's (checked there; jsdom lays nothing out), so
 * these place the controls in rows by hand and check what the toolbar makes
 * of them.
 */
describe('ImageViewControlsComponent wrapping', () => {
  let fixture: ComponentFixture<HostComponent>;
  const realResizeObserver = globalThis.ResizeObserver;

  beforeEach(async () => {
    RecordingResizeObserver.last = null;
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = RecordingResizeObserver;
    const settings = makeSettingsStateStub();
    configureZoneless({
      imports: [HostComponent],
      providers: [...provideHttpTesting(), { provide: SettingsStateService, useValue: settings.stub }],
    });
    TestBed.inject(ActiveContextService).setActivePair('active-ds', 'det');
    fixture = TestBed.createComponent(HostComponent);
    await settleZoneless(fixture);
    // The dataset's blurred set, which the Blur button waits on.
    const httpMock = TestBed.inject(HttpTestingController);
    httpMock.expectOne((r) => r.url === '/api/medias/blurred').flush({ ids: [] });
    await settleZoneless(fixture);
  });

  afterEach(() => {
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = realResizeObserver;
    fixture.destroy();
  });

  const toolbar = (): HTMLElement => fixture.nativeElement.querySelector('vt-image-view-controls');
  const seps = (): HTMLElement[] => Array.from(toolbar().querySelectorAll<HTMLElement>(':scope > .ivc-sep'));
  /** The separator after Marquee, and the one before Blur. */
  const togglesSep = (): HTMLElement => seps()[0];
  const blurSep = (): HTMLElement => seps()[1];
  const blur = (): Element => toolbar().querySelector(':scope > vt-blur-toggle')!;
  const hidden = (): boolean[] => seps().map((s) => s.classList.contains('ivc-sep--edge'));

  /**
   * Put each of the toolbar's controls on the row *rowOf* says, as the
   * browser would: every control centred on its row, the rows 10px apart.
   */
  function layOut(rowOf: (el: Element) => number): void {
    for (const el of Array.from(toolbar().children)) {
      const isSep = el.classList.contains('ivc-sep');
      const top = rowOf(el) * 40 + (isSep ? 6 : 0);
      const height = isSep ? 18 : 30;
      (el as HTMLElement).getBoundingClientRect = () =>
        ({ top, bottom: top + height, left: 0, right: 0, width: 0, height, x: 0, y: top, toJSON: () => ({}) }) as DOMRect;
    }
  }

  async function resized(rowOf: (el: Element) => number): Promise<void> {
    layOut(rowOf);
    RecordingResizeObserver.last!.fire();
    await settleZoneless(fixture);
  }

  it('watches its own size', () => {
    expect(seps().length).toBe(2);
    expect(RecordingResizeObserver.last!.observed.has(toolbar())).toBe(true);
  });

  it('shows every separator while the toolbar is one row', async () => {
    await resized(() => 0);
    expect(hidden()).toEqual([false, false]);
  });

  it('hides the separator left at the end of a row when Blur wraps alone', async () => {
    await resized((el) => (el === blur() ? 1 : 0));
    expect(hidden()).toEqual([false, true]);
  });

  it('hides the separator that wraps with Blur to the start of the next row', async () => {
    await resized((el) => (el === blur() || el === blurSep() ? 1 : 0));
    expect(hidden()).toEqual([false, true]);
  });

  it('hides the separator after Marquee when the row breaks beside it', async () => {
    await resized((el) => (el === toolbar().firstElementChild || el === togglesSep() ? 0 : 1));
    expect(hidden()).toEqual([true, false]);
  });

  it('shows a separator again once its row has room', async () => {
    await resized((el) => (el === blur() ? 1 : 0));
    expect(hidden()).toEqual([false, true]);

    await resized(() => 0);
    expect(hidden()).toEqual([false, false]);
  });

  it('stops watching when it goes', () => {
    const observer = RecordingResizeObserver.last!;
    fixture.destroy();
    expect(observer.observed.size).toBe(0);
  });
});
