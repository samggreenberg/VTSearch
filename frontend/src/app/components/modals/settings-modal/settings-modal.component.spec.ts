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
    view_mode_left: 'grid',
    view_mode_right: 'list',
    enrich_descriptions: false,
    safe_thresholds: false,
    calibrate_count: 50,
    calibration_fraction: 0.5,
    autopilot_top_greens: 3,
    autopilot_hard_reds: 3,
    autoload_media_types: ['audio'],
    autoload_media_embedders: ['audio'],
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
    // forkJoin makes both requests in parallel
    const settingsReq = httpMock.expectOne('/api/settings');
    const embeddersReq = httpMock.expectOne('/api/embedders');
    settingsReq.flush(mockSettings);
    embeddersReq.flush({ embedders: [{ name: 'audio', media_type_id: 'audio' }, { name: 'images', media_type_id: 'image' }, { name: 'text', media_type_id: 'text' }] });
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
    expect(component.embedders.map((e) => `${e.media_type_id}:${e.name}`)).toEqual([
      'audio:alpha',
      'audio:beta',
      'image:clip',
      'image:zebra',
      'text:alpha',
    ]);
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
    settingsReq.flush({}, { status: 500, statusText: 'Error' });
    embeddersReq.flush({ embedders: [] });
    expect(component.loading).toBeFalse();
    expect(component.error).toBe('Failed to load settings');
  });
});
