import { ComponentFixture, TestBed } from '@angular/core/testing';
import { SortBarComponent } from './sort-bar.component';

describe('SortBarComponent', () => {
  let component: SortBarComponent;
  let fixture: ComponentFixture<SortBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SortBarComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(SortBarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should default to text sort mode', () => {
    expect(component.sortMode).toBe('text');
  });

  it('should show text input when in text mode', () => {
    component.sortMode = 'text';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.text-sort-input')).toBeTruthy();
  });

  it('should show learned sort wrap when in learned mode', () => {
    component.sortMode = 'learned';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.learned-sort-wrap')).toBeTruthy();
  });

  it('should show load sort wrap when in load mode', () => {
    component.sortMode = 'load';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.load-sort-wrap')).toBeTruthy();
  });

  it('should emit sortModeChange on radio click', () => {
    spyOn(component.sortModeChange, 'emit');
    component.onSortModeChange('learned');
    expect(component.sortModeChange.emit).toHaveBeenCalledWith('learned');
  });

  it('should emit learnedSort when switching to learned mode', () => {
    spyOn(component.learnedSort, 'emit');
    component.onSortModeChange('learned');
    expect(component.learnedSort.emit).toHaveBeenCalled();
  });

  it('should emit loadSort when switching to load mode', () => {
    spyOn(component.loadSort, 'emit');
    component.onSortModeChange('load');
    expect(component.loadSort.emit).toHaveBeenCalled();
  });

  it('should not emit textSort on input alone', () => {
    spyOn(component.textSort, 'emit');
    component.onTextInput('hello world');
    expect(component.textSort.emit).not.toHaveBeenCalled();
  });

  it('should emit textSort on submitTextSort', () => {
    spyOn(component.textSort, 'emit');
    component.onTextInput('hello world');
    component.submitTextSort();
    expect(component.textSort.emit).toHaveBeenCalledWith('hello world');
  });

  it('should not emit textSort for whitespace-only input on submit', () => {
    spyOn(component.textSort, 'emit');
    component.onTextInput('   ');
    component.submitTextSort();
    expect(component.textSort.emit).not.toHaveBeenCalled();
  });

  it('should render a Search button alongside the text input', () => {
    component.sortMode = 'text';
    fixture.detectChanges();
    const btn = fixture.nativeElement.querySelector('.text-sort-btn');
    expect(btn).toBeTruthy();
    expect(btn.textContent).toContain('Search');
  });

  it('should disable Search button when query is empty or whitespace', () => {
    component.sortMode = 'text';
    component.textQuery = '   ';
    fixture.detectChanges();
    expect(component.searchDisabled).toBe(true);
    component.textQuery = 'cats';
    fixture.detectChanges();
    expect(component.searchDisabled).toBe(false);
  });

  it('should show hint when learned is disabled', () => {
    component.sortMode = 'learned';
    component.learnedSortAvailable = false;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.sort-hint')?.textContent).toContain('Need at least');
  });

  it('should show load sort label', () => {
    component.sortMode = 'load';
    component.loadSortLabel = 'Detector A';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.sort-desc')?.textContent).toContain('Detector A');
  });
});
