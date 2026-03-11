import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { AppComponent } from './app.component';
import { provideRouter } from '@angular/router';
import { VtDialogService } from './services/dialog.service';
import { MediaStateService } from './services/media-state.service';
import { DatasetStateService } from './services/dataset-state.service';

describe('AppComponent', () => {
  let dialogSpy: jasmine.SpyObj<VtDialogService>;

  beforeEach(async () => {
    dialogSpy = jasmine.createSpyObj('VtDialogService', ['alert'], {
      dialogOpen: false,
    });
    dialogSpy.alert.and.returnValue(Promise.resolve(true));

    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [
        provideRouter([]),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: VtDialogService, useValue: dialogSpy },
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
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('VTSearch');
  });

  it('should render logo in header', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    const logo = compiled.querySelector('header .header-logo') as HTMLImageElement;
    expect(logo).toBeTruthy();
    expect(logo.src).toContain('logo.png');
    expect(logo.alt).toBe('VTSearch logo');
  });

  it('should render main content area with router outlet', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.main-content')).toBeTruthy();
  });

  it('should start with menu closed', () => {
    const fixture = TestBed.createComponent(AppComponent);
    expect(fixture.componentInstance.menuOpen).toBeFalse();
  });

  it('should toggle menu open on burger button click', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const btn = fixture.nativeElement.querySelector('.burger-btn') as HTMLElement;
    btn.click();
    expect(fixture.componentInstance.menuOpen).toBeTrue();
  });

  it('should toggle menu closed on second burger button click', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const btn = fixture.nativeElement.querySelector('.burger-btn') as HTMLElement;
    btn.click();
    btn.click();
    expect(fixture.componentInstance.menuOpen).toBeFalse();
  });

  it('should close menu on document click', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    fixture.componentInstance.menuOpen = true;
    fixture.componentInstance.onDocumentClick(new Event('click'));
    expect(fixture.componentInstance.menuOpen).toBeFalse();
  });

  it('should render all menu items', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const items = fixture.nativeElement.querySelectorAll('.burger-item');
    expect(items.length).toBe(5);
    expect(items[0].textContent).toContain('Dashboard');
    expect(items[1].textContent).toContain('Import Labels');
    expect(items[2].textContent).toContain('Export Detector');
    expect(items[3].textContent).toContain('Export Labels');
    expect(items[4].textContent).toContain('Settings');
  });

  it('should disable dataset-dependent items when not on label view', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.componentInstance.isOnLabelView = false;
    fixture.detectChanges();
    const items = fixture.nativeElement.querySelectorAll('.burger-item');
    expect(items[0].classList).toContain('disabled');
    expect(items[1].classList).toContain('disabled');
    expect(items[2].classList).toContain('disabled');
    expect(items[3].classList).toContain('disabled');
    // Settings is always enabled
    expect(items[4].classList).not.toContain('disabled');
  });

  it('should enable dataset-dependent items when on label view', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.componentInstance.isOnLabelView = true;
    fixture.detectChanges();
    const items = fixture.nativeElement.querySelectorAll('.burger-item');
    expect(items[0].classList).not.toContain('disabled');
    expect(items[1].classList).not.toContain('disabled');
    expect(items[2].classList).not.toContain('disabled');
    expect(items[3].classList).not.toContain('disabled');
  });

  it('should not navigate to dashboard when disabled', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const router = TestBed.inject(Router);
    spyOn(router, 'navigate');
    fixture.componentInstance.isOnLabelView = false;
    fixture.componentInstance.onDashboard();
    expect(router.navigate).not.toHaveBeenCalled();
  });

  it('should navigate to dashboard when enabled', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const router = TestBed.inject(Router);
    spyOn(router, 'navigate');
    fixture.componentInstance.isOnLabelView = true;
    fixture.componentInstance.onDashboard();
    expect(router.navigate).toHaveBeenCalledWith(['/dashboard']);
    expect(fixture.componentInstance.menuOpen).toBeFalse();
  });

  it('should close menu on Escape keydown', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    fixture.componentInstance.menuOpen = true;
    const dropdown = fixture.nativeElement.querySelector('.burger-dropdown') as HTMLElement;
    const event = new KeyboardEvent('keydown', { key: 'Escape' });
    dropdown.dispatchEvent(event);
    expect(fixture.componentInstance.menuOpen).toBeFalse();
  });

  it('should open Settings modal when Settings clicked', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.componentInstance.menuOpen = true;
    fixture.componentInstance.onSettings();
    expect(fixture.componentInstance.showSettings).toBeTrue();
    expect(fixture.componentInstance.menuOpen).toBeFalse();
  });

  it('should set settingsViewTab from labeling media type', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const mediaState = TestBed.inject(MediaStateService);
    spyOnProperty(mediaState, 'medias', 'get').and.returnValue([
      { id: 1, type: 'image', filename: 'a.png', md5: 'abc', custom_metadata: {} },
    ]);
    fixture.componentInstance.isOnLabelView = true;
    fixture.componentInstance.onSettings();
    expect(fixture.componentInstance.settingsViewTab).toBe('image');
  });

  it('should set settingsViewTab from dashboard when all same media type', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const datasetState = TestBed.inject(DatasetStateService);
    spyOnProperty(datasetState, 'datasets', 'get').and.returnValue([
      { media_type: 'audio' },
      { media_type: 'audio' },
    ]);
    spyOnProperty(datasetState, 'models', 'get').and.returnValue([
      { media_type: 'audio' },
    ]);
    fixture.componentInstance.isOnLabelView = false;
    fixture.componentInstance.onSettings();
    expect(fixture.componentInstance.settingsViewTab).toBe('audio');
  });

  it('should set empty settingsViewTab from dashboard when mixed media types', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const datasetState = TestBed.inject(DatasetStateService);
    spyOnProperty(datasetState, 'datasets', 'get').and.returnValue([
      { media_type: 'audio' },
      { media_type: 'image' },
    ]);
    spyOnProperty(datasetState, 'models', 'get').and.returnValue([]);
    fixture.componentInstance.isOnLabelView = false;
    fixture.componentInstance.onSettings();
    expect(fixture.componentInstance.settingsViewTab).toBe('');
  });
});
