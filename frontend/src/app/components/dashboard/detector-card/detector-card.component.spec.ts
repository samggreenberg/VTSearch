import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { DetectorCardComponent } from './detector-card.component';

describe('DetectorCardComponent', () => {
  let component: DetectorCardComponent;
  let fixture: ComponentFixture<DetectorCardComponent>;

  const mockDetector = {
    id: 'm1',
    name: 'Test Detector',
    media_type: 'audio',
    num_training: 50,
    last_trained_at: 1700000000,
    created_at: 1699000000,
    loaded: true,
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DetectorCardComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(DetectorCardComponent);
    component = fixture.componentInstance;
    component.detector = { ...mockDetector };
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display detector name', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('Test Detector');
  });

  it('should display capitalized media type with icon', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('Audio');
    expect(el.querySelector('.type-icon')).toBeTruthy();
  });

  it('should display training count', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('50');
  });

  it('should enter rename mode on rename button click', () => {
    const el = fixture.nativeElement as HTMLElement;
    const renameBtn = el.querySelector('.edit-btn') as HTMLElement;
    renameBtn.click();
    fixture.detectChanges();
    expect(component.editing).toBeTrue();
    expect(el.querySelector('.inline-edit')).toBeTruthy();
  });

  it('should focus the rename input after clicking rename', fakeAsync(() => {
    const el = fixture.nativeElement as HTMLElement;
    const renameBtn = el.querySelector('.edit-btn') as HTMLElement;
    renameBtn.click();
    fixture.detectChanges();
    tick();
    const input = el.querySelector('.inline-edit') as HTMLInputElement;
    expect(document.activeElement).toBe(input);
  }));

  it('should emit rename on confirm', () => {
    spyOn(component.rename, 'emit');
    component.editing = true;
    component.editName = 'Renamed';
    component.confirmRename();
    expect(component.rename.emit).toHaveBeenCalledWith('Renamed');
  });

  it('should cancel rename on Escape', () => {
    spyOn(component.rename, 'emit');
    component.editing = true;
    component.onRenameKeydown(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(component.rename.emit).not.toHaveBeenCalled();
    expect(component.editing).toBeFalse();
  });

  it('should emit delete on delete button click', () => {
    spyOn(component.delete, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    const deleteBtn = el.querySelector('.delete-btn') as HTMLElement;
    deleteBtn.click();
    expect(component.delete.emit).toHaveBeenCalled();
  });

  it('should format dates', () => {
    expect(component.formatDate(1700000000)).toMatch(/\d/);
    expect(component.formatDate(null)).toBe('-');
  });

  it('should apply selected class via host binding', () => {
    component.selected = true;
    fixture.detectChanges();
    expect(fixture.nativeElement.classList.contains('selected')).toBeTrue();
  });
});
