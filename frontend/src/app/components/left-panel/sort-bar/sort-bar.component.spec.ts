import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
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

  it('should debounce text input and emit textSort', fakeAsync(() => {
    spyOn(component.textSort, 'emit');
    component.onTextInput('hello world');
    expect(component.textSort.emit).not.toHaveBeenCalled();
    tick(400);
    expect(component.textSort.emit).toHaveBeenCalledWith('hello world');
  }));

  it('should not emit textSort for whitespace-only input', fakeAsync(() => {
    spyOn(component.textSort, 'emit');
    component.onTextInput('   ');
    tick(400);
    expect(component.textSort.emit).not.toHaveBeenCalled();
  }));

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
