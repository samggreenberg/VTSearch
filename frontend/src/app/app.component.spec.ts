import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { AppComponent } from './app.component';
import { provideRouter } from '@angular/router';
import { MediaStateService } from './services/media-state.service';
import { DatasetStateService } from './services/dataset-state.service';
import { provideZoneless } from './testing/zoneless-testbed';

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [
        ...provideZoneless(),
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it(`should have the 'VTSearch' title`, () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app.title).toEqual('VTSearch');
  });

  it('should render header with title', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.tick();
    const compiled = fixture.nativeElement as HTMLElement;
    // Branding is rendered as the logo image (alt text), not an <h1>.
    const logo = compiled.querySelector('header .header-logo') as HTMLImageElement;
    expect(logo.alt).toContain('VTSearch');
  });

  it('should render logo in header', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.tick();
    const compiled = fixture.nativeElement as HTMLElement;
    const logo = compiled.querySelector('header .header-logo') as HTMLImageElement;
    expect(logo).toBeTruthy();
    expect(logo.src).toContain('logo.png');
    expect(logo.alt).toBe('VTSearch logo');
  });

  it('should render main content area with router outlet', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.tick();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.main-content')).toBeTruthy();
  });

  it('should start with menu closed', () => {
    const fixture = TestBed.createComponent(AppComponent);
    expect(fixture.componentInstance.menuOpen).toBe(false);
  });

  it('should toggle menu open on burger button click', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.tick();
    const btn = fixture.nativeElement.querySelector('.burger-btn') as HTMLElement;
    btn.click();
    expect(fixture.componentInstance.menuOpen).toBe(true);
  });

  it('should toggle menu closed on second burger button click', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.tick();
    const btn = fixture.nativeElement.querySelector('.burger-btn') as HTMLElement;
    btn.click();
    btn.click();
    expect(fixture.componentInstance.menuOpen).toBe(false);
  });

  it('should close menu on document click', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.tick();
    fixture.componentInstance.menuOpen = true;
    fixture.componentInstance.onDocumentClick(new Event('click'));
    expect(fixture.componentInstance.menuOpen).toBe(false);
  });

  it('should show the Recent sessions burger with an empty state when none exist', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.tick();
    const dropdown = fixture.nativeElement.querySelector('.burger-dropdown') as HTMLElement;
    // The burger no longer duplicates the top-bar destinations; it is a
    // Recent-sessions menu, empty by default in the test harness.
    expect(dropdown.querySelector('.burger-section-header')?.textContent).toContain(
      'Recent sessions',
    );
    expect(dropdown.querySelectorAll('.burger-subitem').length).toBe(0);
    expect(dropdown.querySelector('.burger-status')?.textContent).toContain('No recent sessions');
  });

  it('should render the header icon buttons (Achievements, Help, Settings)', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.tick();
    const buttons = fixture.nativeElement.querySelectorAll('header .header-icon-btn');
    const labels = Array.from(buttons).map((b) => (b as HTMLElement).getAttribute('aria-label'));
    expect(labels).toEqual(['Achievements', 'Help', 'Settings']);
  });

  it('should disable the Dashboard logo/button when not on label view', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.componentInstance.isOnLabelView.set(false);
    TestBed.tick();
    const logo = fixture.nativeElement.querySelector('.logo-btn') as HTMLElement;
    const dashBtn = fixture.nativeElement.querySelector('.top-bar-btn') as HTMLButtonElement;
    expect(logo.classList).toContain('disabled');
    expect(dashBtn.disabled).toBe(true);
  });

  it('should enable the Dashboard logo/button when on label view', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.componentInstance.isOnLabelView.set(true);
    TestBed.tick();
    const logo = fixture.nativeElement.querySelector('.logo-btn') as HTMLElement;
    const dashBtn = fixture.nativeElement.querySelector('.top-bar-btn') as HTMLButtonElement;
    expect(logo.classList).not.toContain('disabled');
    expect(dashBtn.disabled).toBe(false);
  });

  it('should not navigate to dashboard when disabled', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const router = TestBed.inject(Router);
    vi.spyOn(router, 'navigate').mockResolvedValue(true);
    fixture.componentInstance.isOnLabelView.set(false);
    fixture.componentInstance.onDashboard();
    expect(router.navigate).not.toHaveBeenCalled();
  });

  it('should navigate to dashboard when enabled', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const router = TestBed.inject(Router);
    vi.spyOn(router, 'navigate').mockResolvedValue(true);
    fixture.componentInstance.isOnLabelView.set(true);
    fixture.componentInstance.onDashboard();
    expect(router.navigate).toHaveBeenCalledWith(['/dashboard']);
    expect(fixture.componentInstance.menuOpen).toBe(false);
  });

  it('should close menu on Escape keydown', () => {
    const fixture = TestBed.createComponent(AppComponent);
    TestBed.tick();
    fixture.componentInstance.menuOpen = true;
    const dropdown = fixture.nativeElement.querySelector('.burger-dropdown') as HTMLElement;
    const event = new KeyboardEvent('keydown', { key: 'Escape' });
    dropdown.dispatchEvent(event);
    expect(fixture.componentInstance.menuOpen).toBe(false);
  });

  it('should open Settings modal when Settings clicked', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.componentInstance.menuOpen = true;
    fixture.componentInstance.onSettings();
    expect(fixture.componentInstance.showSettings).toBe(true);
    expect(fixture.componentInstance.menuOpen).toBe(false);
  });

  it('should set settingsViewTab from labeling media type', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const mediaState = TestBed.inject(MediaStateService);
    vi.spyOn(mediaState, 'mediasSignal').mockReturnValue([
      { id: 1, media_type: 'image' },
    ]);
    fixture.componentInstance.isOnLabelView.set(true);
    fixture.componentInstance.onSettings();
    expect(fixture.componentInstance.settingsViewTab).toBe('image');
  });

  it('should set settingsViewTab from dashboard when all same media type', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const datasetState = TestBed.inject(DatasetStateService);
    vi.spyOn(datasetState, 'datasets', 'get').mockReturnValue([
      { id: 'd1', name: 'DS1', media_type: 'audio' },
      { id: 'd2', name: 'DS2', media_type: 'audio' },
    ]);
    vi.spyOn(datasetState, 'detectors', 'get').mockReturnValue([
      { id: 'm1', name: 'M1', media_type: 'audio' },
    ]);
    fixture.componentInstance.isOnLabelView.set(false);
    fixture.componentInstance.onSettings();
    expect(fixture.componentInstance.settingsViewTab).toBe('audio');
  });

  it('should set empty settingsViewTab from dashboard when mixed media types', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const datasetState = TestBed.inject(DatasetStateService);
    vi.spyOn(datasetState, 'datasets', 'get').mockReturnValue([
      { id: 'd1', name: 'DS1', media_type: 'audio' },
      { id: 'd2', name: 'DS2', media_type: 'image' },
    ]);
    vi.spyOn(datasetState, 'detectors', 'get').mockReturnValue([]);
    fixture.componentInstance.isOnLabelView.set(false);
    fixture.componentInstance.onSettings();
    expect(fixture.componentInstance.settingsViewTab).toBe('');
  });
});
