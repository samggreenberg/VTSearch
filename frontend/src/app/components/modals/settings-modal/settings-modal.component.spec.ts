import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { SettingsModalComponent } from './settings-modal.component';

describe('SettingsModalComponent', () => {
  let component: SettingsModalComponent;
  let fixture: ComponentFixture<SettingsModalComponent>;
  let httpMock: HttpTestingController;

  const mockSettings = {
    volume: 50,
    theme: 'dark',
    show_animations: true,
    view_mode_left: { audio: 'list', image: 'grid' },
    view_mode_right: { audio: 'grid', image: 'list' },
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
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(SettingsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushInit(): void {
    fixture.detectChanges();
    // forkJoin makes all requests in parallel: settings, media types,
    // embedders, and version.
    const settingsReq = httpMock.expectOne('/api/settings');
    const mediaTypesReq = httpMock.expectOne('/api/media-types');
    const embeddersReq = httpMock.expectOne('/api/embedders');
    const versionReq = httpMock.expectOne('/api/version');
    // Deep-clone so per-test component mutations (onViewModeChange, etc.)
    // never leak back into the shared mockSettings constant.
    settingsReq.flush(structuredClone(mockSettings));
    mediaTypesReq.flush(mockMediaTypes);
    embeddersReq.flush({ embedders: [] });
    versionReq.flush({ version: '2026-05-07T00:00:00Z' });
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should load settings on init', () => {
    flushInit();
    expect(component.settings().theme).toBe('dark');
    expect(component.loading()).toBe(false);
  });

  it('should load media types and set active view tab', () => {
    flushInit();
    expect(component.mediaTypes().length).toBe(2);
    expect(component.activeViewTab()).toBe('audio');
  });

  it('should update theme and save', () => {
    flushInit();
    component.onThemeChange('light');
    expect(component.settings().theme).toBe('light');
    // onThemeChange persists twice: ThemeService.setTheme issues its own
    // PUT /api/settings, and save() issues another.
    const reqs = httpMock.match('/api/settings');
    expect(reqs.length).toBe(2);
    reqs.forEach((r) => r.flush(mockSettings));
  });

  it('should toggle boolean setting and save', () => {
    flushInit();
    component.onToggle('show_animations', false);
    expect(component.settings().show_animations).toBe(false);
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should update number setting and save', () => {
    flushInit();
    component.onNumberChange('calibrate_count', 100);
    expect(component.settings().calibrate_count).toBe(100);
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should update per-media-type view mode and save', () => {
    flushInit();
    component.onViewModeChange('view_mode_left', 'audio', 'grid');
    const dict = component.settings().view_mode_left as Record<string, string>;
    expect(dict['audio']).toBe('grid');
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should get view mode for a media type', () => {
    flushInit();
    expect(component.getViewMode('view_mode_left', 'audio')).toBe('list');
    expect(component.getViewMode('view_mode_right', 'audio')).toBe('grid');
  });

  it('should reset to defaults after confirmation', fakeAsync(() => {
    flushInit();
    vi.spyOn(component['dialog'], 'confirmDestructive').mockReturnValue(Promise.resolve(true));
    component.resetDefaults();
    tick();
    httpMock.expectOne('/api/settings/defaults').flush({ ...mockSettings, theme: 'light' });
    // Applying the defaults persists twice: ThemeService.setTheme PUTs the
    // new theme, and save() PUTs the full settings object.
    const reqs = httpMock.match('/api/settings');
    expect(reqs.length).toBe(2);
    reqs.forEach((r) => r.flush(mockSettings));
    expect(component.settings().theme).toBe('light');
  }));

  it('should not reset when confirmation is declined', fakeAsync(() => {
    flushInit();
    vi.spyOn(component['dialog'], 'confirmDestructive').mockReturnValue(Promise.resolve(false));
    component.resetDefaults();
    tick();
    httpMock.expectNone('/api/settings/defaults');
  }));

  it('should use preselectedViewTab when valid', () => {
    component.preselectedViewTab = 'image';
    flushInit();
    expect(component.activeViewTab()).toBe('image');
    expect(component.activeSettingsTab()).toBe('appearance');
  });

  it('should ignore preselectedViewTab when not in mediaTypes', () => {
    component.preselectedViewTab = 'video';
    flushInit();
    expect(component.activeViewTab()).toBe('audio');
  });

  it('should ignore empty preselectedViewTab', () => {
    component.preselectedViewTab = '';
    flushInit();
    expect(component.activeViewTab()).toBe('audio');
  });

  it('should emit closed on close', () => {
    flushInit();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should handle init error gracefully', () => {
    fixture.detectChanges();
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
