import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { SettingsModalComponent } from './settings-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('SettingsModalComponent', () => {
  let component: SettingsModalComponent;
  let fixture: ComponentFixture<SettingsModalComponent>;
  let httpMock: HttpTestingController;

  const mockSettings = {
    volume: 50,
    theme: 'dark',
    show_animations: 'os',
    enrich_descriptions: false,
    calibrate_count: 50,
    calibration_fraction: 0.5,
    autopilot_top_greens: 3,
    autopilot_hard_reds: 3,
    autopilot_goal_diversity: 40,
  };

  const mockMediaTypes = {
    media_types: [
      { type_id: 'audio', name: 'Sound', icon: 'audio' },
      { type_id: 'image', name: 'Image', icon: 'image' },
    ],
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SettingsModalComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(SettingsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  // Minimal matchMedia stub so the "Browser motion" status line can be driven
  // in jsdom (which omits matchMedia). Mirrors the shape ThemeService's spec
  // uses: a matches flag plus a fire-change helper for the live listener.
  interface StubMediaQueryList {
    matches: boolean;
    media: string;
    addEventListener: (type: string, l: (e: MediaQueryListEvent) => void) => void;
    removeEventListener: (type: string, l: (e: MediaQueryListEvent) => void) => void;
    _listeners: Array<(e: MediaQueryListEvent) => void>;
    _fireChange: (matches: boolean) => void;
  }
  let originalMatchMedia: typeof window.matchMedia;
  function installMatchMedia(initialMatches: boolean): StubMediaQueryList {
    const stub: StubMediaQueryList = {
      matches: initialMatches,
      media: '(prefers-reduced-motion: reduce)',
      _listeners: [],
      addEventListener(_type, l) {
        this._listeners.push(l);
      },
      removeEventListener(_type, l) {
        this._listeners = this._listeners.filter((x) => x !== l);
      },
      _fireChange(matches: boolean) {
        this.matches = matches;
        this._listeners.forEach((l) => l({ matches } as MediaQueryListEvent));
      },
    };
    (window as unknown as { matchMedia: (q: string) => MediaQueryList }).matchMedia = () =>
      stub as unknown as MediaQueryList;
    return stub;
  }

  beforeEach(() => {
    originalMatchMedia = window.matchMedia;
  });

  afterEach(() => {
    httpMock.verify();
    (window as unknown as { matchMedia: typeof window.matchMedia }).matchMedia =
      originalMatchMedia;
  });

  // Settle runs ngOnInit, whose forkJoin issues the four parallel GETs (plain
  // HTTP subscribes don't hold the app unstable, so whenStable resolves before
  // the flush). Settle again afterwards so the resolved signals repaint.
  async function flushInit(): Promise<void> {
    await settleZoneless(fixture);
    // ngOnInit also refreshes HuggingFace sign-in status for the HuggingFace tab.
    httpMock
      .expectOne('/api/auth/huggingface/status')
      .flush({ configured: false, authenticated: false, username: '', scopes: '' });
    // forkJoin makes all requests in parallel: settings, media types,
    // embedders, and version.
    const settingsReq = httpMock.expectOne('/api/settings');
    const mediaTypesReq = httpMock.expectOne('/api/media-types');
    const embeddersReq = httpMock.expectOne('/api/embedders');
    const versionReq = httpMock.expectOne('/api/version');
    // Deep-clone so per-test component mutations (onGridIconSizeChange, etc.)
    // never leak back into the shared mockSettings constant.
    settingsReq.flush(structuredClone(mockSettings));
    mediaTypesReq.flush(mockMediaTypes);
    embeddersReq.flush({ embedders: [] });
    versionReq.flush({ version: '2026-05-07T00:00:00Z' });
    await settleZoneless(fixture);
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
  });

  it('should load settings on init', async () => {
    await flushInit();
    expect(component.settings().theme).toBe('dark');
    expect(component.loading()).toBe(false);
  });

  it('should load media types and set active view tab', async () => {
    await flushInit();
    expect(component.mediaTypes().length).toBe(2);
    expect(component.activeViewTab()).toBe('audio');
  });

  it('should update theme and save', async () => {
    await flushInit();
    component.onThemeChange('light');
    expect(component.settings().theme).toBe('light');
    // onThemeChange persists twice: ThemeService.setTheme issues its own
    // PUT /api/settings, and save() issues another.
    const reqs = httpMock.match('/api/settings');
    expect(reqs.length).toBe(2);
    reqs.forEach((r) => r.flush(mockSettings));
  });

  it('should toggle boolean setting and save', async () => {
    await flushInit();
    component.onToggle('show_metadata', false);
    expect(component.settings().show_metadata).toBe(false);
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should update the Show Animations pulldown and save', async () => {
    await flushInit();
    component.onAnimationModeChange('show');
    expect(component.settings().show_animations).toBe('show');
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should default the Graphics pulldown to auto when the setting is unset', async () => {
    await flushInit();
    expect(component.browseGraphics).toBe('auto');
  });

  it('should update the Graphics pulldown and save', async () => {
    await flushInit();
    component.onBrowseGraphicsChange('reduced');
    expect(component.settings().browse_graphics).toBe('reduced');
    expect(component.browseGraphics).toBe('reduced');
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should fall back to auto for an unrecognized stored Graphics value', async () => {
    await flushInit();
    // The typed union rules this out at compile time, so cast through unknown:
    // the guard exists for a settings file hand-edited (or written by a newer
    // build) with a mode this client doesn't know.
    component.settings.update(
      (s) => ({ ...s, browse_graphics: 'turbo' }) as unknown as typeof s,
    );
    expect(component.browseGraphics).toBe('auto');
  });

  it('should update number setting and save', async () => {
    await flushInit();
    component.onNumberChange('calibrate_count', 100);
    expect(component.settings().calibrate_count).toBe(100);
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should reset to defaults after confirmation', async () => {
    await flushInit();
    vi.spyOn(component['dialog'], 'confirmDestructiveWithEscape').mockReturnValue(Promise.resolve('confirm'));
    component.resetDefaults();
    // Drain the confirm() promise continuation that issues the GET.
    await new Promise<void>((resolve) => setTimeout(resolve));
    httpMock.expectOne('/api/settings/defaults').flush({ ...mockSettings, theme: 'light' });
    // Applying the defaults persists twice: ThemeService.setTheme PUTs the
    // new theme, and save() PUTs the full settings object.
    const reqs = httpMock.match('/api/settings');
    expect(reqs.length).toBe(2);
    reqs.forEach((r) => r.flush(mockSettings));
    expect(component.settings().theme).toBe('light');
  });

  it('should not reset when confirmation is declined', async () => {
    await flushInit();
    vi.spyOn(component['dialog'], 'confirmDestructiveWithEscape').mockReturnValue(Promise.resolve('cancel'));
    component.resetDefaults();
    await new Promise<void>((resolve) => setTimeout(resolve));
    httpMock.expectNone('/api/settings/defaults');
  });

  it('should open the exporter and skip reset when the escape hatch is chosen', async () => {
    await flushInit();
    vi.spyOn(component['dialog'], 'confirmDestructiveWithEscape').mockReturnValue(Promise.resolve('escape'));
    expect(component.showExporterModal).toBe(false);
    component.resetDefaults();
    await new Promise<void>((resolve) => setTimeout(resolve));
    expect(component.showExporterModal).toBe(true);
    httpMock.expectNone('/api/settings/defaults');
  });

  it('should use preselectedViewTab when valid', async () => {
    fixture.componentRef.setInput('preselectedViewTab', 'image');
    await flushInit();
    expect(component.activeViewTab()).toBe('image');
    expect(component.activeSettingsTab()).toBe('appearance');
  });

  it('should ignore preselectedViewTab when not in mediaTypes', async () => {
    fixture.componentRef.setInput('preselectedViewTab', 'video');
    await flushInit();
    expect(component.activeViewTab()).toBe('audio');
  });

  it('should ignore empty preselectedViewTab', async () => {
    fixture.componentRef.setInput('preselectedViewTab', '');
    await flushInit();
    expect(component.activeViewTab()).toBe('audio');
  });

  it('should emit closed on close', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('reports browser motion as allowed when matchMedia is unavailable', async () => {
    (window as unknown as { matchMedia: unknown }).matchMedia = undefined;
    await flushInit();
    expect(component.browserBlocksMotion()).toBe(false);
  });

  it('reports browser motion as blocked when the OS prefers reduced motion', async () => {
    installMatchMedia(true);
    await flushInit();
    expect(component.browserBlocksMotion()).toBe(true);
  });

  it('updates the status live when the OS preference flips', async () => {
    const stub = installMatchMedia(false);
    await flushInit();
    expect(component.browserBlocksMotion()).toBe(false);

    stub._fireChange(true);
    expect(component.browserBlocksMotion()).toBe(true);

    stub._fireChange(false);
    expect(component.browserBlocksMotion()).toBe(false);
  });

  it('removes the matchMedia listener on destroy', async () => {
    const stub = installMatchMedia(false);
    await flushInit();
    expect(stub._listeners.length).toBe(1);
    component.ngOnDestroy();
    expect(stub._listeners.length).toBe(0);
  });

  it('should handle init error gracefully', async () => {
    await settleZoneless(fixture);
    httpMock
      .expectOne('/api/auth/huggingface/status')
      .flush({ configured: false, authenticated: false, username: '', scopes: '' });
    const settingsReq = httpMock.expectOne('/api/settings');
    const mediaTypesReq = httpMock.expectOne('/api/media-types');
    const embeddersReq = httpMock.expectOne('/api/embedders');
    const versionReq = httpMock.expectOne('/api/version');
    // Flush the successful siblings first; erroring settings last avoids
    // forkJoin cancelling the in-flight siblings before they're flushed.
    mediaTypesReq.flush({ media_types: [] });
    embeddersReq.flush({ embedders: [] });
    versionReq.flush({ version: '2026-05-07T00:00:00Z' });
    settingsReq.flush({}, { status: 500, statusText: 'Error' });
    expect(component.loading()).toBe(false);
    expect(component.error()).toBe('Failed to load settings');
  });
});
