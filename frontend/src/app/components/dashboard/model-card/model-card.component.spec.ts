import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ModelCardComponent } from './model-card.component';

describe('ModelCardComponent', () => {
  let component: ModelCardComponent;
  let fixture: ComponentFixture<ModelCardComponent>;

  const mockModel = {
    id: 'm1',
    name: 'Test Model',
    media_type: 'audio',
    trainable: true,
    num_training: 50,
    last_trained_at: 1700000000,
    created_at: 1699000000,
    loaded: true,
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ModelCardComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(ModelCardComponent);
    component = fixture.componentInstance;
    component.model = { ...mockModel };
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display model name', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('Test Model');
  });

  it('should display media type', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('audio');
  });

  it('should display training count', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('50');
  });

  it('should show trainable checkmark', () => {
    const el = fixture.nativeElement as HTMLElement;
    const checks = el.querySelectorAll('.check');
    expect(checks.length).toBeGreaterThan(0);
  });

  it('should show dash when not trainable', () => {
    component.model = { ...mockModel, trainable: false };
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const dims = el.querySelectorAll('.dim');
    expect(dims.length).toBeGreaterThan(0);
  });

  it('should enter rename mode on rename button click', () => {
    const el = fixture.nativeElement as HTMLElement;
    const renameBtn = el.querySelector('.edit-btn') as HTMLElement;
    renameBtn.click();
    fixture.detectChanges();
    expect(component.editing).toBeTrue();
    expect(el.querySelector('.inline-edit')).toBeTruthy();
  });

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
