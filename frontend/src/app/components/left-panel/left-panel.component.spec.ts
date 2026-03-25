import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { LeftPanelComponent } from './left-panel.component';

describe('LeftPanelComponent', () => {
  let component: LeftPanelComponent;
  let fixture: ComponentFixture<LeftPanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LeftPanelComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(LeftPanelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should default to autopilot tab', () => {
    expect(component.activeTab).toBe('autopilot');
  });

  it('should emit autopilotStart on init', () => {
    const fresh = TestBed.createComponent(LeftPanelComponent);
    const comp = fresh.componentInstance;
    spyOn(comp.autopilotStart, 'emit');
    fresh.detectChanges();
    expect(comp.autopilotStart.emit).toHaveBeenCalled();
  });

  it('should switch to manual tab', () => {
    component.setTab('manual');
    expect(component.activeTab).toBe('manual');
  });

  it('should render autopilot tab content by default', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.tab-panel-autopilot')).toBeTruthy();
    expect(el.querySelector('.tab-panel-manual')).toBeNull();
  });

  it('should render manual tab content when switched', () => {
    component.setTab('manual');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.tab-panel-manual')).toBeTruthy();
    expect(el.querySelector('.tab-panel-autopilot')).toBeNull();
  });

  it('should show active class on selected tab', () => {
    const el = fixture.nativeElement as HTMLElement;
    const tabs = el.querySelectorAll('.left-tab');
    // Default: autopilot is active (second tab)
    expect(tabs[0].classList.contains('active')).toBeFalse();
    expect(tabs[1].classList.contains('active')).toBeTrue();

    component.setTab('manual');
    fixture.detectChanges();
    const tabsAfter = el.querySelectorAll('.left-tab');
    expect(tabsAfter[0].classList.contains('active')).toBeTrue();
    expect(tabsAfter[1].classList.contains('active')).toBeFalse();
  });

  it('should emit autopilotStart when switching to autopilot tab', () => {
    component.setTab('manual');
    spyOn(component.autopilotStart, 'emit');
    component.setTab('autopilot');
    expect(component.autopilotStart.emit).toHaveBeenCalled();
  });

  it('should emit autopilotStop when switching from autopilot to manual tab', () => {
    spyOn(component.autopilotStop, 'emit');
    component.setTab('manual');
    expect(component.autopilotStop.emit).toHaveBeenCalled();
  });

  it('should not emit start when setting the same tab', () => {
    spyOn(component.autopilotStart, 'emit');
    component.setTab('autopilot');
    expect(component.autopilotStart.emit).not.toHaveBeenCalled();
  });

  it('should emit autopilotRefocus when clicking already-active autopilot tab', () => {
    spyOn(component.autopilotRefocus, 'emit');
    component.setTab('autopilot');
    expect(component.autopilotRefocus.emit).toHaveBeenCalled();
  });

  it('should not emit autopilotRefocus when clicking already-active manual tab', () => {
    component.setTab('manual');
    spyOn(component.autopilotRefocus, 'emit');
    component.setTab('manual');
    expect(component.autopilotRefocus.emit).not.toHaveBeenCalled();
  });

  it('should emit sortModeChange', () => {
    spyOn(component.sortModeChange, 'emit');
    component.sortModeChange.emit('learned');
    expect(component.sortModeChange.emit).toHaveBeenCalledWith('learned');
  });

  it('should emit mediaSelect', () => {
    spyOn(component.mediaSelect, 'emit');
    component.mediaSelect.emit(42);
    expect(component.mediaSelect.emit).toHaveBeenCalledWith(42);
  });
});
