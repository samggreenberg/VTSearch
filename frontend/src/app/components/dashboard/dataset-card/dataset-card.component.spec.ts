import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DatasetCardComponent } from './dataset-card.component';
import { DashboardLoadingTasksService } from '../../../services/dashboard-loading-tasks.service';
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

  // Stub the loading-tasks service so the card doesn't drag in the real
  // SSE/HTTP DI chain; the card only reads `isCancelling` for the badge.
  let cancellingIds: Set<string>;

  beforeEach(async () => {
    cancellingIds = new Set<string>();
    await TestBed.configureTestingModule({
      imports: [DatasetCardComponent],
      providers: [
        ...provideZoneless(),
        {
          provide: DashboardLoadingTasksService,
          useValue: { isCancelling: (id: string) => cancellingIds.has(id) },
        },
      ],
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
    const labels = Array.from(el.querySelectorAll('.menu-item')).map((b) => b.textContent?.trim());
    // Loaded, single-user dataset. Only Delete is inline (plus Load when
    // unloaded), so the ⋯ overflow still carries Browse alongside the tail;
    // Delete is dropped, Load is hidden (loaded), Edit-access is single-user-hidden.
    expect(labels).toEqual(['Browse dataset', 'Rename', 'Stats']);
  });

  it('should still list the inline Delete verb in the right-click context menu', async () => {
    const el = fixture.nativeElement as HTMLElement;
    el.dispatchEvent(new MouseEvent('contextmenu', { clientX: 10, clientY: 10, bubbles: true }));
    await settleZoneless(fixture);
    const labels = Array.from(el.querySelectorAll('.menu-item')).map((b) => b.textContent?.trim());
    // Right-click stays complete: Delete returns alongside the rest.
    expect(labels).toEqual(['Browse dataset', 'Rename', 'Stats', 'Delete']);
  });

  it('should emit browse from the overflow menu', async () => {
    vi.spyOn(component.browse, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    (el.querySelector('.overflow-btn') as HTMLElement).click();
    await settleZoneless(fixture);
    const items = Array.from(el.querySelectorAll('.menu-item')) as HTMLElement[];
    const browseItem = items.find((b) => b.textContent?.includes('Browse dataset'));
    expect(browseItem).toBeTruthy();
    browseItem!.click();
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

  it('shows a disabled "Cancelling…" badge on the loading row once its task is cancelling', async () => {
    const task = { task_id: 'load-1', status: 'running', current: 0, total: 0 };
    fixture.componentRef.setInput('loadingTask', task);
    await settleZoneless(fixture);
    // Before cancel: a live Cancel button.
    let btn = (fixture.nativeElement as HTMLElement).querySelector('.jp__cancel') as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
    expect(btn.textContent?.trim()).toBe('Cancel');

    // The service now reports this task as cancelling; re-flow the input so
    // the getter re-reads the stub.
    cancellingIds.add('load-1');
    fixture.componentRef.setInput('loadingTask', { ...task });
    await settleZoneless(fixture);

    btn = (fixture.nativeElement as HTMLElement).querySelector('.jp__cancel') as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.textContent?.trim()).toBe('Cancelling…');
  });
});
