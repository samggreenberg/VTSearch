import { ComponentFixture, TestBed } from '@angular/core/testing';
import { LabelSortComponent, LabelSortMode } from './label-sort.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('LabelSortComponent', () => {
  let component: LabelSortComponent;
  let fixture: ComponentFixture<LabelSortComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LabelSortComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelSortComponent);
    component = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should default to time-desc', () => {
    expect(component.mode()).toBe('time-desc');
  });

  it('should emit modeChange on sort change', () => {
    let emitted: LabelSortMode | undefined;
    component.mode.subscribe((v) => (emitted = v));
    component.onSortChange('name-asc');
    expect(component.mode()).toBe('name-asc');
    expect(emitted).toBe('name-asc');
  });

  it('should render all sort options', () => {
    const el = fixture.nativeElement as HTMLElement;
    const options = el.querySelectorAll('option');
    expect(options.length).toBe(7);
    const values = Array.from(options).map(o => o.getAttribute('value'));
    expect(values).toContain('time-desc');
    expect(values).toContain('time-asc');
    expect(values).toContain('name-asc');
    expect(values).toContain('name-desc');
    expect(values).toContain('confidence-desc');
    expect(values).toContain('confidence-asc');
    expect(values).toContain('id-asc');
  });

  it('should accept mode input', async () => {
    fixture.componentRef.setInput('mode', 'confidence-desc');
    await settleZoneless(fixture);
    expect(component.mode()).toBe('confidence-desc');
  });
});
