import { ComponentFixture, TestBed } from '@angular/core/testing';
import { LabelSortComponent, LabelSortMode } from './label-sort.component';

describe('LabelSortComponent', () => {
  let component: LabelSortComponent;
  let fixture: ComponentFixture<LabelSortComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LabelSortComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelSortComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should default to time-desc', () => {
    expect(component.mode).toBe('time-desc');
  });

  it('should emit modeChange on sort change', () => {
    spyOn(component.modeChange, 'emit');
    component.onSortChange('name-asc');
    expect(component.mode).toBe('name-asc');
    expect(component.modeChange.emit).toHaveBeenCalledWith('name-asc');
  });

  it('should render all sort options', () => {
    const el = fixture.nativeElement as HTMLElement;
    const options = el.querySelectorAll('option');
    expect(options.length).toBe(7);
    const values = Array.from(options).map(o => o.value);
    expect(values).toContain('time-desc');
    expect(values).toContain('time-asc');
    expect(values).toContain('name-asc');
    expect(values).toContain('name-desc');
    expect(values).toContain('confidence-desc');
    expect(values).toContain('confidence-asc');
    expect(values).toContain('id-asc');
  });

  it('should accept mode input', () => {
    component.mode = 'confidence-desc';
    fixture.detectChanges();
    const select = fixture.nativeElement.querySelector('select') as HTMLSelectElement;
    expect(select.value).toBe('confidence-desc');
  });
});
