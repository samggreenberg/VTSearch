import { ComponentFixture, TestBed } from '@angular/core/testing';
import { InclusionSliderComponent } from './inclusion-slider.component';

describe('InclusionSliderComponent', () => {
  let component: InclusionSliderComponent;
  let fixture: ComponentFixture<InclusionSliderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [InclusionSliderComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(InclusionSliderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should default to 0', () => {
    expect(component.value).toBe(0);
  });

  it('should emit valueChange on input', () => {
    spyOn(component.valueChange, 'emit');
    const input = fixture.nativeElement.querySelector('input[type="range"]');
    input.value = '7';
    input.dispatchEvent(new Event('input'));
    expect(component.valueChange.emit).toHaveBeenCalledWith(7);
  });

  it('should render a range input with correct min/max', () => {
    const input = fixture.nativeElement.querySelector('input[type="range"]');
    expect(input.getAttribute('min')).toBe('-10');
    expect(input.getAttribute('max')).toBe('10');
  });
});
