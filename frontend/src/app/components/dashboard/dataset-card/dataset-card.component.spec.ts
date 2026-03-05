import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DatasetCardComponent } from './dataset-card.component';

describe('DatasetCardComponent', () => {
  let component: DatasetCardComponent;
  let fixture: ComponentFixture<DatasetCardComponent>;

  const mockDataset = {
    id: 'd1',
    name: 'Test Dataset',
    media_type: 'audio',
    num_items: 100,
    num_dupes: 5,
    created_at: 1700000000,
    origin: 'folder:/data/audio',
    loaded: true,
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DatasetCardComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(DatasetCardComponent);
    component = fixture.componentInstance;
    component.dataset = { ...mockDataset };
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display dataset name', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('Test Dataset');
  });

  it('should display media type', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('audio');
  });

  it('should display item count', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('100');
  });

  it('should show checkmark when loaded', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.check')).toBeTruthy();
  });

  it('should show Load button when not loaded', () => {
    component.dataset = { ...mockDataset, loaded: false };
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const loadBtn = el.querySelector('button.btn--secondary');
    expect(loadBtn?.textContent).toContain('Load');
  });

  it('should enter rename mode on rename button click', () => {
    const el = fixture.nativeElement as HTMLElement;
    const renameBtn = el.querySelector('.edit-btn') as HTMLElement;
    renameBtn.click();
    fixture.detectChanges();
    expect(component.editing).toBeTrue();
    expect(component.editName).toBe('Test Dataset');
    expect(el.querySelector('.inline-edit')).toBeTruthy();
  });

  it('should emit rename on Enter key', () => {
    spyOn(component.rename, 'emit');
    component.editing = true;
    component.editName = 'New Name';
    component.onRenameKeydown(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(component.rename.emit).toHaveBeenCalledWith('New Name');
    expect(component.editing).toBeFalse();
  });

  it('should cancel rename on Escape key', () => {
    spyOn(component.rename, 'emit');
    component.editing = true;
    component.editName = 'New Name';
    component.onRenameKeydown(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(component.rename.emit).not.toHaveBeenCalled();
    expect(component.editing).toBeFalse();
  });

  it('should not emit rename when name unchanged', () => {
    spyOn(component.rename, 'emit');
    component.editing = true;
    component.editName = 'Test Dataset';
    component.confirmRename();
    expect(component.rename.emit).not.toHaveBeenCalled();
  });

  it('should emit delete on delete button click', () => {
    spyOn(component.delete, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    const deleteBtn = el.querySelector('.delete-btn') as HTMLElement;
    deleteBtn.click();
    expect(component.delete.emit).toHaveBeenCalled();
  });

  it('should emit load on load button click', () => {
    spyOn(component.load, 'emit');
    component.dataset = { ...mockDataset, loaded: false };
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const loadBtn = el.querySelector('.btn--secondary') as HTMLElement;
    loadBtn.click();
    expect(component.load.emit).toHaveBeenCalled();
  });

  it('should format dates', () => {
    expect(component.formatDate(1700000000)).toMatch(/\d/);
    expect(component.formatDate(null)).toBe('-');
    expect(component.formatDate(0)).toBe('-');
  });

  it('should apply selected class via host binding', () => {
    component.selected = true;
    fixture.detectChanges();
    expect(fixture.nativeElement.classList.contains('selected')).toBeTrue();
  });
});
