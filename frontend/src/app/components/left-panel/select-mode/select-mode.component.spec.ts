import { ComponentFixture, TestBed } from '@angular/core/testing';
import { SelectModeComponent } from './select-mode.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('SelectModeComponent', () => {
  let component: SelectModeComponent;
  let fixture: ComponentFixture<SelectModeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SelectModeComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(SelectModeComponent);
    component = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should default to top select mode', () => {
    expect(component.selectMode).toBe('top');
  });

  it('should render three radio options', () => {
    const radios = fixture.nativeElement.querySelectorAll('input[type="radio"]');
    expect(radios.length).toBe(3);
  });

  it('should emit selectModeChange on radio change', () => {
    vi.spyOn(component.selectModeChange, 'emit');
    component.onChange('hard');
    expect(component.selectModeChange.emit).toHaveBeenCalledWith('hard');
  });

  it('should mark active radio with active class', async () => {
    fixture.componentRef.setInput('selectMode', 'hard');
    await settleZoneless(fixture);
    const labels = fixture.nativeElement.querySelectorAll('.sort-radio');
    expect(labels[1].classList.contains('active')).toBe(true);
  });
});
