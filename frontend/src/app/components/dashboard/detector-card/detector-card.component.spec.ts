import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DetectorCardComponent } from './detector-card.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

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
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(DetectorCardComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('detector', { ...mockDetector });
    // The card only renders middle columns listed in columnOrder; the
    // dashboard supplies this ordering. Provide a representative set so the
    // media-type and training-count cells render.
    fixture.componentRef.setInput('columnOrder', ['media_type', 'num_training', 'last_trained_at']);
    await settleZoneless(fixture);
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

  it('should enter rename mode on rename button click', async () => {
    const el = fixture.nativeElement as HTMLElement;
    const renameBtn = el.querySelector('.edit-btn') as HTMLElement;
    renameBtn.click();
    await settleZoneless(fixture);
    expect(component.editing).toBe(true);
    expect(el.querySelector('.inline-edit')).toBeTruthy();
  });

  it('should focus the rename input after clicking rename', async () => {
    const el = fixture.nativeElement as HTMLElement;
    const renameBtn = el.querySelector('.edit-btn') as HTMLElement;
    renameBtn.click();
    await settleZoneless(fixture);
    await settleZoneless(fixture);
    const input = el.querySelector('.inline-edit') as HTMLInputElement;
    expect(document.activeElement).toBe(input);
  });

  it('should emit rename on confirm', () => {
    vi.spyOn(component.rename, 'emit');
    component.editing = true;
    component.editName = 'Renamed';
    component.confirmRename();
    expect(component.rename.emit).toHaveBeenCalledWith('Renamed');
  });

  it('should cancel rename on Escape', () => {
    vi.spyOn(component.rename, 'emit');
    component.editing = true;
    component.onRenameKeydown(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(component.rename.emit).not.toHaveBeenCalled();
    expect(component.editing).toBe(false);
  });

  it('should emit delete on delete button click', () => {
    vi.spyOn(component.delete, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    const deleteBtn = el.querySelector('.delete-btn') as HTMLElement;
    deleteBtn.click();
    expect(component.delete.emit).toHaveBeenCalled();
  });

  it('should format dates', () => {
    expect(component.formatDate(1700000000)).toMatch(/\d/);
    expect(component.formatDate(null)).toBe('-');
  });

  it('should apply selected class via host binding', async () => {
    fixture.componentRef.setInput('selected', true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.classList.contains('selected')).toBe(true);
  });
});
