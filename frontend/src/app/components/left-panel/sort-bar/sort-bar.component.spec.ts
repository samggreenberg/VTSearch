import { ComponentFixture, TestBed } from '@angular/core/testing';
import { SortBarComponent } from './sort-bar.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('SortBarComponent', () => {
  let component: SortBarComponent;
  let fixture: ComponentFixture<SortBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SortBarComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(SortBarComponent);
    component = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should default to text sort mode', () => {
    expect(component.sortMode).toBe('text');
  });

  it('should show text input when in text mode', async () => {
    fixture.componentRef.setInput('sortMode', 'text');
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.text-sort-input')).toBeTruthy();
  });

  it('should show learned sort wrap when in learned mode', async () => {
    fixture.componentRef.setInput('sortMode', 'learned');
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.learned-sort-wrap')).toBeTruthy();
  });

  it('should show load sort wrap when in load mode', async () => {
    fixture.componentRef.setInput('sortMode', 'load');
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.load-sort-wrap')).toBeTruthy();
  });

  it('should emit sortModeChange on radio click', () => {
    vi.spyOn(component.sortModeChange, 'emit');
    component.onSortModeChange('learned');
    expect(component.sortModeChange.emit).toHaveBeenCalledWith('learned');
  });

  it('should emit learnedSort when switching to learned mode', () => {
    vi.spyOn(component.learnedSort, 'emit');
    component.onSortModeChange('learned');
    expect(component.learnedSort.emit).toHaveBeenCalled();
  });

  it('should emit loadSort when switching to load mode', () => {
    vi.spyOn(component.loadSort, 'emit');
    component.onSortModeChange('load');
    expect(component.loadSort.emit).toHaveBeenCalled();
  });

  it('should not emit textSort on input alone', () => {
    vi.spyOn(component.textSort, 'emit');
    component.onTextInput('hello world');
    expect(component.textSort.emit).not.toHaveBeenCalled();
  });

  it('should emit textSort on submitTextSort', () => {
    vi.spyOn(component.textSort, 'emit');
    component.onTextInput('hello world');
    component.submitTextSort();
    expect(component.textSort.emit).toHaveBeenCalledWith('hello world');
  });

  it('should not emit textSort for whitespace-only input on submit', () => {
    vi.spyOn(component.textSort, 'emit');
    component.onTextInput('   ');
    component.submitTextSort();
    expect(component.textSort.emit).not.toHaveBeenCalled();
  });

  it('should render a Search button alongside the text input', async () => {
    fixture.componentRef.setInput('sortMode', 'text');
    await settleZoneless(fixture);
    const btn = fixture.nativeElement.querySelector('.text-sort-btn');
    expect(btn).toBeTruthy();
    expect(btn.textContent).toContain('Search');
  });

  it('should disable Search button when query is empty or whitespace', () => {
    // searchDisabled is a getter over the internal textQuery field; no render.
    component.textQuery = '   ';
    expect(component.searchDisabled).toBe(true);
    component.textQuery = 'cats';
    expect(component.searchDisabled).toBe(false);
  });

  it('should show hint when learned is disabled', async () => {
    fixture.componentRef.setInput('sortMode', 'learned');
    fixture.componentRef.setInput('learnedSortAvailable', false);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.sort-hint')?.textContent).toContain('Need at least');
  });

  it('should show load sort label', async () => {
    fixture.componentRef.setInput('sortMode', 'load');
    fixture.componentRef.setInput('loadSortLabel', 'Detector A');
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.sort-desc')?.textContent).toContain('Detector A');
  });
});
