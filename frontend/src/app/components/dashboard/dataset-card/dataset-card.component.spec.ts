import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DatasetCardComponent } from './dataset-card.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

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
    loaded: true,
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DatasetCardComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(DatasetCardComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('dataset', { ...mockDataset });
    // The card only renders middle columns listed in columnOrder; the
    // dashboard supplies this ordering. Provide a representative set so the
    // media-type and item-count cells render.
    fixture.componentRef.setInput('columnOrder', ['media_type', 'num_items', 'created_at']);
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display dataset name', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('Test Dataset');
  });

  it('should display capitalized media type with icon', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('Audio');
    expect(el.querySelector('.type-icon')).toBeTruthy();
  });

  it('should display item count', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('100');
  });

  it('should hide the load button when already loaded', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.load-btn')).toBeFalsy();
  });

  it('should show the load button when not loaded', async () => {
    fixture.componentRef.setInput('dataset', { ...mockDataset, loaded: false });
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.load-btn')).toBeTruthy();
  });

  it('should enter rename mode on rename button click', async () => {
    const el = fixture.nativeElement as HTMLElement;
    const renameBtn = el.querySelector('.edit-btn') as HTMLElement;
    renameBtn.click();
    await settleZoneless(fixture);
    expect(component.editing).toBe(true);
    expect(component.editName).toBe('Test Dataset');
    expect(el.querySelector('.inline-edit')).toBeTruthy();
  });

  it('should focus the rename input after clicking rename', async () => {
    const el = fixture.nativeElement as HTMLElement;
    const renameBtn = el.querySelector('.edit-btn') as HTMLElement;
    renameBtn.click();
    // Settle renders the .inline-edit input; the focus setTimeout queued in
    // beginRename() then runs against the live element.
    await settleZoneless(fixture);
    await settleZoneless(fixture);
    const input = el.querySelector('.inline-edit') as HTMLInputElement;
    expect(document.activeElement).toBe(input);
  });

  it('should emit rename on Enter key', () => {
    vi.spyOn(component.rename, 'emit');
    component.editing = true;
    component.editName = 'New Name';
    component.onRenameKeydown(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(component.rename.emit).toHaveBeenCalledWith('New Name');
    expect(component.editing).toBe(false);
  });

  it('should cancel rename on Escape key', () => {
    vi.spyOn(component.rename, 'emit');
    component.editing = true;
    component.editName = 'New Name';
    component.onRenameKeydown(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(component.rename.emit).not.toHaveBeenCalled();
    expect(component.editing).toBe(false);
  });

  it('should not emit rename when name unchanged', () => {
    vi.spyOn(component.rename, 'emit');
    component.editing = true;
    component.editName = 'Test Dataset';
    component.confirmRename();
    expect(component.rename.emit).not.toHaveBeenCalled();
  });

  it('should emit delete on delete button click', () => {
    vi.spyOn(component.delete, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    const deleteBtn = el.querySelector('.delete-btn') as HTMLElement;
    deleteBtn.click();
    expect(component.delete.emit).toHaveBeenCalled();
  });

  it('should emit browse on browse button click for audio datasets', () => {
    vi.spyOn(component.browse, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    const browseBtn = el.querySelector('.browse-btn') as HTMLButtonElement;
    expect(browseBtn.disabled).toBe(false);
    browseBtn.click();
    expect(component.browse.emit).toHaveBeenCalled();
  });

  it('should emit browse for non-audio datasets too', async () => {
    fixture.componentRef.setInput('dataset', { ...mockDataset, media_type: 'image' });
    await settleZoneless(fixture);
    vi.spyOn(component.browse, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    const browseBtn = el.querySelector('.browse-btn') as HTMLButtonElement;
    expect(browseBtn.disabled).toBe(false);
    browseBtn.click();
    expect(component.browse.emit).toHaveBeenCalled();
  });

  it('should format dates', () => {
    expect(component.formatDate(1700000000)).toMatch(/\d/);
    expect(component.formatDate(null)).toBe('-');
    expect(component.formatDate(0)).toBe('-');
  });

  it('should apply selected class via host binding', async () => {
    fixture.componentRef.setInput('selected', true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.classList.contains('selected')).toBe(true);
  });
});
