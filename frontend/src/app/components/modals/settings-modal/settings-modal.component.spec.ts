import { ComponentFixture, TestBed } from '@angular/core/testing';
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
    swipe_animation: true,
    view_mode_left: { audio: 'list', image: 'grid' },
    view_mode_right: { audio: 'grid', image: 'list' },
    enrich_descriptions: false,
    safe_thresholds: false,
    calibrate_count: 50,
    calibration_fraction: 0.5,
    autopilot_top_greens: 3,
    autopilot_hard_reds: 3,
    autopilot_goal_diversity: 40,
    autoload_media_types: ['audio'],
    autoload_media_embedders: ['audio'],
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
    const embeddersReq = httpMock.expectOne('/api/embedders');
    const mediaTypesReq = httpMock.expectOne('/api/media-types');
    settingsReq.flush(mockSettings);
    embeddersReq.flush({ embedders: [{ name: 'audio', media_type_id: 'audio' }, { name: 'images', media_type_id: 'image' }, { name: 'text', media_type_id: 'text' }] });
    mediaTypesReq.flush(mockMediaTypes);
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should load settings and embedders on init', () => {
    flushInit();
    expect(component.settings.theme).toBe('dark');
    expect(component.embedders.length).toBe(3);
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
    component.onToggle('swipe_animation', false);
    expect(component.settings.swipe_animation).toBeFalse();
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

  it('should check if embedder is autoloaded', () => {
    flushInit();
    expect(component.isEmbedderAutoloaded({ name: 'audio', media_type_id: 'audio' })).toBeTrue();
    expect(component.isEmbedderAutoloaded({ name: 'clip', media_type_id: 'image' })).toBeFalse();
  });

  it('should toggle embedder in autoload list', () => {
    flushInit();
    component.toggleEmbedder({ name: 'clip', media_type_id: 'image' });
    expect(component.settings.autoload_media_embedders).toContain('clip');
    httpMock.expectOne('/api/settings').flush(mockSettings);

    component.toggleEmbedder({ name: 'audio', media_type_id: 'audio' });
    expect(component.settings.autoload_media_embedders).not.toContain('audio');
    httpMock.expectOne('/api/settings').flush(mockSettings);
  });

  it('should reset to defaults', () => {
    flushInit();
    component.resetDefaults();
    httpMock.expectOne('/api/settings/defaults').flush({ ...mockSettings, theme: 'light' });
    httpMock.expectOne('/api/settings').flush(mockSettings);
    expect(component.settings.theme).toBe('light');
  });

  it('should sort embedders by media_type_id then name', () => {
    fixture.detectChanges();
    const settingsReq = httpMock.expectOne('/api/settings');
    const embeddersReq = httpMock.expectOne('/api/embedders');
    const mediaTypesReq = httpMock.expectOne('/api/media-types');
    settingsReq.flush(mockSettings);
    embeddersReq.flush({
      embedders: [
        { name: 'zebra', media_type_id: 'image' },
        { name: 'alpha', media_type_id: 'text' },
        { name: 'beta', media_type_id: 'audio' },
        { name: 'alpha', media_type_id: 'audio' },
        { name: 'clip', media_type_id: 'image' },
      ],
    });
    mediaTypesReq.flush(mockMediaTypes);
    expect(component.embedders.map((e) => `${e.media_type_id}:${e.name}`)).toEqual([
      'audio:alpha',
      'audio:beta',
      'image:clip',
      'image:zebra',
      'text:alpha',
    ]);
  });

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

  it('should return media type icon for embedder', () => {
    flushInit();
    expect(component.getMediaTypeIcon('audio')).toBe('audio');
    expect(component.getMediaTypeIcon('image')).toBe('image');
    expect(component.getMediaTypeIcon('unknown')).toBe('');
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
    const embeddersReq = httpMock.expectOne('/api/embedders');
    const mediaTypesReq = httpMock.expectOne('/api/media-types');
    settingsReq.flush({}, { status: 500, statusText: 'Error' });
    embeddersReq.flush({ embedders: [] });
    mediaTypesReq.flush({ media_types: [] });
    expect(component.loading).toBeFalse();
    expect(component.error).toBe('Failed to load settings');
  });
});
