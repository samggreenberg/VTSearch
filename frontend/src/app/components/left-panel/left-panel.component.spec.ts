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

  it('should default to manual tab', () => {
    expect(component.activeTab).toBe('manual');
  });

  it('should switch to autopilot tab', () => {
    component.setTab('autopilot');
    expect(component.activeTab).toBe('autopilot');
  });

  it('should render manual tab content by default', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.tab-panel-manual')).toBeTruthy();
    expect(el.querySelector('.tab-panel-autopilot')).toBeNull();
  });

  it('should render autopilot tab content when switched', () => {
    component.setTab('autopilot');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.tab-panel-autopilot')).toBeTruthy();
    expect(el.querySelector('.tab-panel-manual')).toBeNull();
  });

  it('should show active class on selected tab', () => {
    const el = fixture.nativeElement as HTMLElement;
    const tabs = el.querySelectorAll('.left-tab');
    expect(tabs[0].classList.contains('active')).toBeTrue();
    expect(tabs[1].classList.contains('active')).toBeFalse();

    component.setTab('autopilot');
    fixture.detectChanges();
    const tabsAfter = el.querySelectorAll('.left-tab');
    expect(tabsAfter[0].classList.contains('active')).toBeFalse();
    expect(tabsAfter[1].classList.contains('active')).toBeTrue();
  });

  it('should emit autopilotStart when switching to autopilot tab', () => {
    spyOn(component.autopilotStart, 'emit');
    component.setTab('autopilot');
    expect(component.autopilotStart.emit).toHaveBeenCalled();
  });

  it('should emit autopilotStop when switching from autopilot to manual tab', () => {
    component.setTab('autopilot');
    spyOn(component.autopilotStop, 'emit');
    component.setTab('manual');
    expect(component.autopilotStop.emit).toHaveBeenCalled();
  });

  it('should not emit when setting the same tab', () => {
    spyOn(component.autopilotStart, 'emit');
    component.setTab('manual');
    expect(component.autopilotStart.emit).not.toHaveBeenCalled();
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
