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
    // forkJoin makes all requests in parallel
    const settingsReq = httpMock.expectOne('/api/settings');
    const mediaTypesReq = httpMock.expectOne('/api/media-types');
    const versionReq = httpMock.expectOne('/api/version');
    settingsReq.flush(mockSettings);
    mediaTypesReq.flush(mockMediaTypes);
    versionReq.flush({ version: '2026-05-07T00:00:00Z' });
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should load settings on init', () => {
    flushInit();
    expect(component.settings.theme).toBe('dark');
    expect(component.loading).toBeFalse();
  });

  it('should load media types and set active view tab', () => {
    flushInit();
    expect(component.mediaTypes.length).toBe(2);
    expect(component.activeViewTab).toBe('audio');
  });

  it('should update theme and save', () => {
    flushInit();
    component.onThemeChange('light');
    expect(component.settings.theme).toBe('light');
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should toggle boolean setting and save', () => {
    flushInit();
    component.onToggle('show_animations', false);
    expect(component.settings.show_animations).toBeFalse();
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should update number setting and save', () => {
    flushInit();
    component.onNumberChange('calibrate_count', 100);
    expect(component.settings.calibrate_count).toBe(100);
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should update per-media-type view mode and save', () => {
    flushInit();
    component.onViewModeChange('view_mode_left', 'audio', 'grid');
    const dict = component.settings.view_mode_left as Record<string, string>;
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
    spyOn(component['dialog'], 'confirmDestructive').and.returnValue(Promise.resolve(true));
    component.resetDefaults();
    tick();
    httpMock.expectOne('/api/settings/defaults').flush({ ...mockSettings, theme: 'light' });
    httpMock.expectOne('/api/settings').flush(mockSettings);
    expect(component.settings.theme).toBe('light');
  }));

  it('should not reset when confirmation is declined', fakeAsync(() => {
    flushInit();
    spyOn(component['dialog'], 'confirmDestructive').and.returnValue(Promise.resolve(false));
    component.resetDefaults();
    tick();
    httpMock.expectNone('/api/settings/defaults');
  }));

  it('should use preselectedViewTab when valid', () => {
    component.preselectedViewTab = 'image';
    flushInit();
    expect(component.activeViewTab).toBe('image');
    expect(component.activeSettingsTab).toBe('appearance');
  });

  it('should ignore preselectedViewTab when not in mediaTypes', () => {
    component.preselectedViewTab = 'video';
    flushInit();
    expect(component.activeViewTab).toBe('audio');
  });

  it('should ignore empty preselectedViewTab', () => {
    component.preselectedViewTab = '';
    flushInit();
    expect(component.activeViewTab).toBe('audio');
  });

  it('should emit closed on close', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should handle init error gracefully', () => {
    fixture.detectChanges();
    const settingsReq = httpMock.expectOne('/api/settings');
    const mediaTypesReq = httpMock.expectOne('/api/media-types');
    const versionReq = httpMock.expectOne('/api/version');
    settingsReq.flush({}, { status: 500, statusText: 'Error' });
    mediaTypesReq.flush({ media_types: [] });
    versionReq.flush({ version: '2026-05-07T00:00:00Z' });
    expect(component.loading).toBeFalse();
    expect(component.error).toBe('Failed to load settings');
  });
});
