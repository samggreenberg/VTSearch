import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ProgressIndicatorsComponent } from './progress-indicators.component';

describe('ProgressIndicatorsComponent', () => {
  let component: ProgressIndicatorsComponent;
  let fixture: ComponentFixture<ProgressIndicatorsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ProgressIndicatorsComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(ProgressIndicatorsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should render three indicators', () => {
    const buttons = fixture.nativeElement.querySelectorAll('.labeling-indicator');
    expect(buttons.length).toBe(3);
  });

  it('should show empty status by default', () => {
    expect(component.smartStatus).toBe('');
    expect(component.stableStatus).toBe('');
    expect(component.spanStatus).toBe('');
  });

  it('should reflect labeling status', () => {
    component.labelingStatus = {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'yellow' },
      span: { status: '' },
    };
    fixture.detectChanges();
    expect(component.smartStatus).toBe('green');
    expect(component.stableStatus).toBe('yellow');
    expect(component.spanStatus).toBe('');
  });

  it('should emit indicatorClick on button click', () => {
    spyOn(component.indicatorClick, 'emit');
    const buttons = fixture.nativeElement.querySelectorAll('.labeling-indicator');
    buttons[0].click();
    expect(component.indicatorClick.emit).toHaveBeenCalledWith('smart');
    buttons[1].click();
    expect(component.indicatorClick.emit).toHaveBeenCalledWith('stable');
    buttons[2].click();
    expect(component.indicatorClick.emit).toHaveBeenCalledWith('span');
  });

  it('should show smart subtext with cost', () => {
    component.labelingStatus = {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'yellow', cost: 0.123 },
      stable: { status: '' },
      span: { status: '' },
    };
    expect(component.smartSubtext).toBe('Cost: 0.123');
  });

  it('should show stable subtext with flips', () => {
    component.labelingStatus = {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: '' },
      stable: { status: 'yellow', flips: 5 },
      span: { status: '' },
    };
    expect(component.stableSubtext).toBe('Flips: 5');
  });

  it('should show span subtext with diversity level', () => {
    component.labelingStatus = {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: '' },
      stable: { status: '' },
      span: { status: 'green', diversity_level: 2.5, max_level: 4 },
    };
    expect(component.spanSubtext).toBe('3/4');
  });

  it('should show sort overlay when sortBusy is true', () => {
    component.sortBusy = true;
    component.sortStatus = 'Sorting...';
    fixture.detectChanges();
    const overlay = fixture.nativeElement.querySelector('.sort-overlay');
    expect(overlay).toBeTruthy();
    expect(overlay.textContent).toContain('Sorting...');
    expect(fixture.nativeElement.querySelector('.labeling-indicator')).toBeNull();
  });

  it('should show progress bar inside sort overlay when busy', () => {
    component.sortBusy = true;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.sort-overlay vt-progress-bar')).toBeTruthy();
  });

  it('should show indicators when sortBusy is false', () => {
    component.sortBusy = false;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.sort-overlay')).toBeNull();
    expect(fixture.nativeElement.querySelectorAll('.labeling-indicator').length).toBe(3);
  });

  it('should set data-status attribute on indicators', () => {
    component.labelingStatus = {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'yellow' },
      span: { status: 'red' },
    };
    fixture.detectChanges();
    const buttons = fixture.nativeElement.querySelectorAll('.labeling-indicator');
    expect(buttons[0].getAttribute('data-status')).toBe('green');
    expect(buttons[1].getAttribute('data-status')).toBe('yellow');
    expect(buttons[2].getAttribute('data-status')).toBe('red');
  });
});
