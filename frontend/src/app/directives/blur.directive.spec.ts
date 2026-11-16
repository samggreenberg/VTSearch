import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { HttpTestingController, TestRequest } from '@angular/common/http/testing';

import { BlurDirective } from './blur.directive';
import { ActiveContextService } from '../services/active-context.service';
import { BlurStateService } from '../services/blur-state.service';
import { SettingsStateService } from '../services/settings-state.service';
import { activeContextInterceptor } from '../interceptors/active-context.interceptor';
import { provideHttpTesting } from '../testing/test-providers';
import { provideZoneless } from '../testing/zoneless-testbed';
import { settleZoneless } from '../testing/settle-resource';
import { makeSettingsStateStub } from '../testing/mocks';
import type { AppSettings } from '../generated/api-client/models/app-settings';

@Component({
  standalone: true,
  imports: [BlurDirective],
  template: `
    @if (shown()) {
      <div class="frame"><img class="pic" alt="" [vtBlur]="id()" [vtBlurDataset]="datasetId()" /></div>
    }
  `,
})
class HostComponent {
  readonly id = signal<number | null>(1);
  readonly datasetId = signal<string | undefined>(undefined);
  readonly shown = signal(true);
}

/** A ResizeObserver that records what it observes and lets the spec report a size. */
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
  report(element: Element, width: number, height: number): void {
    const entry = { target: element, contentRect: { width, height } } as unknown as ResizeObserverEntry;
    this.callback([entry], this as unknown as ResizeObserver);
  }
}

const BLURRED = '/api/medias/blurred';

describe('BlurDirective', () => {
  let fixture: ComponentFixture<HostComponent>;
  let host: HostComponent;
  let httpMock: HttpTestingController;
  let blurState: BlurStateService;
  let settings: ReturnType<typeof makeSettingsStateStub>['settings'];
  const realResizeObserver = globalThis.ResizeObserver;

  beforeEach(async () => {
    RecordingResizeObserver.last = null;
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = RecordingResizeObserver;
    const settingsStub = makeSettingsStateStub();
    settings = settingsStub.settings;
    settings.set({ blur_rules: [{ field: 'nsfw', op: 'truthy' }] } as AppSettings);

    await TestBed.configureTestingModule({
      imports: [HostComponent],
      providers: [
        ...provideZoneless(),
        ...provideHttpTesting(activeContextInterceptor),
        { provide: SettingsStateService, useValue: settingsStub.stub },
      ],
    }).compileComponents();
    httpMock = TestBed.inject(HttpTestingController);
    blurState = TestBed.inject(BlurStateService);
    TestBed.inject(ActiveContextService).setActivePair('active-ds', 'active-det');

    fixture = TestBed.createComponent(HostComponent);
    host = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  afterEach(() => {
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = realResizeObserver;
    httpMock.verify();
  });

  const img = (): HTMLImageElement => fixture.nativeElement.querySelector('.pic');
  const isBlurred = (): boolean => img().classList.contains('vt-blurred');
  const radius = (): string => img().style.getPropertyValue('--vt-blur-radius');

  function expectRead(datasetId: string): TestRequest {
    return httpMock.expectOne((r) => r.url === BLURRED && r.headers.get('X-Dataset-Id') === datasetId);
  }

  it('blurs before the set arrives, then follows the service, with no manual change detection', async () => {
    expect(isBlurred()).toBe(true); // fail closed: the instance has rules

    expectRead('active-ds').flush({ ids: [] });
    await settleZoneless(fixture);
    expect(isBlurred()).toBe(false);

    blurState.setBlurred('active-ds', 1, true);
    await settleZoneless(fixture);
    expect(isBlurred()).toBe(true);
    httpMock.expectOne('/api/medias/1/blur').flush({ ok: true, id: 1, blurred: true });

    blurState.setBlurred('active-ds', 1, false);
    await settleZoneless(fixture);
    expect(isBlurred()).toBe(false);
    httpMock.expectOne('/api/medias/1/blur').flush({ ok: true, id: 1, blurred: false });
  });

  it('follows its item: another id reads its own state', async () => {
    expectRead('active-ds').flush({ ids: [2] });
    await settleZoneless(fixture);
    expect(isBlurred()).toBe(false);

    host.id.set(2);
    await settleZoneless(fixture);
    expect(isBlurred()).toBe(true);
  });

  it('never blurs an image that names no item', async () => {
    host.id.set(null);
    await settleZoneless(fixture);
    expect(isBlurred()).toBe(false);
    expectRead('active-ds').flush({ ids: [] });
  });

  it('reads a named dataset instead of the active one', async () => {
    expectRead('active-ds').flush({ ids: [] });
    host.datasetId.set('sibling-ds');
    await settleZoneless(fixture);
    expect(isBlurred()).toBe(true); // the sibling's set has not arrived

    expectRead('sibling-ds').flush({ ids: [] });
    await settleZoneless(fixture);
    expect(isBlurred()).toBe(false);
  });

  it('sets the radius from the rendered size while blurred, and drops it once sharp', async () => {
    const observer = RecordingResizeObserver.last!;
    expect(observer.observed.has(img())).toBe(true);
    expect(radius()).toBe(''); // not measured yet: the CSS fallback applies

    observer.report(img(), 200, 150);
    await settleZoneless(fixture);
    expect(radius()).toBe('12px');

    observer.report(img(), 600, 500);
    await settleZoneless(fixture);
    expect(radius()).toBe('40px');

    expectRead('active-ds').flush({ ids: [] });
    await settleZoneless(fixture);
    expect(isBlurred()).toBe(false);
    expect(radius()).toBe('');
    expect(observer.observed.has(img())).toBe(false);
  });

  it('keeps the dataset loaded while mounted, and releases it when removed', async () => {
    const release = vi.fn();
    const retain = vi.spyOn(blurState, 'retain').mockReturnValue(release);
    expectRead('active-ds').flush({ ids: [] });

    host.datasetId.set('sibling-ds');
    await settleZoneless(fixture);
    expect(retain).toHaveBeenCalledWith('sibling-ds');

    host.shown.set(false);
    await settleZoneless(fixture);
    expect(release).toHaveBeenCalledTimes(1);
  });
});
