import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { SettingsModalComponent } from './settings-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('SettingsModalComponent', () => {
  let component: SettingsModalComponent;
  let fixture: ComponentFixture<SettingsModalComponent>;
  let httpMock: HttpTestingController;

  const mockSettings = {
    volume: 50,
    theme: 'dark',
    show_animations: true,
    enrich_descriptions: false,
    safe_thresholds: false,
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
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(SettingsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
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
    component.onToggle('show_animations', false);
    expect(component.settings().show_animations).toBe(false);
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should update number setting and save', async () => {
    await flushInit();
    component.onNumberChange('calibrate_count', 100);
    expect(component.settings().calibrate_count).toBe(100);
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should reset to defaults after confirmation', async () => {
    await flushInit();
    vi.spyOn(component['dialog'], 'confirmDestructive').mockReturnValue(Promise.resolve(true));
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
    vi.spyOn(component['dialog'], 'confirmDestructive').mockReturnValue(Promise.resolve(false));
    component.resetDefaults();
    await new Promise<void>((resolve) => setTimeout(resolve));
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

  it('should handle init error gracefully', async () => {
    await settleZoneless(fixture);
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
