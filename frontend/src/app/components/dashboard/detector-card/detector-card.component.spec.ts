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

  it('should emit delete from the inline delete button', () => {
    vi.spyOn(component.delete, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    (el.querySelector('.delete-btn') as HTMLElement).click();
    expect(component.delete.emit).toHaveBeenCalled();
  });

  it('should drop the inline verbs from the overflow menu', async () => {
    const el = fixture.nativeElement as HTMLElement;
    (el.querySelector('.overflow-btn') as HTMLElement).click();
    await settleZoneless(fixture);
    const overflow = Array.from(el.querySelectorAll('.menu-item')).map((b) => b.textContent?.trim());
    // Browse and Delete are inline icons; the ⋯ menu omits them.
    expect(overflow).not.toContain('Browse positives');
    expect(overflow).not.toContain('Delete');
    expect(overflow).toContain('Rename');
  });

  it('should still list the inline verbs in the right-click context menu', async () => {
    const el = fixture.nativeElement as HTMLElement;
    el.dispatchEvent(new MouseEvent('contextmenu', { clientX: 10, clientY: 10, bubbles: true }));
    await settleZoneless(fixture);
    const full = Array.from(el.querySelectorAll('.menu-item')).map((b) => b.textContent?.trim());
    // Right-click stays complete: Browse and Delete return.
    expect(full).toContain('Browse positives');
    expect(full).toContain('Delete');
  });

  it('should surface export and import-labels actions in the overflow menu', async () => {
    vi.spyOn(component.export, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    (el.querySelector('.overflow-btn') as HTMLElement).click();
    await settleZoneless(fixture);
    const items = Array.from(el.querySelectorAll('.menu-item')) as HTMLElement[];
    expect(items.some((b) => b.textContent?.includes('Import Labels'))).toBe(true);
    const exportItem = items.find((b) => b.textContent?.includes('Export'));
    expect(exportItem).toBeTruthy();
    exportItem!.click();
    expect(component.export.emit).toHaveBeenCalled();
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
