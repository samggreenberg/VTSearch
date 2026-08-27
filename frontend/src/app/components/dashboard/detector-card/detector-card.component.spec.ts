import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DetectorCardComponent } from './detector-card.component';
import { DashboardLoadingTasksService } from '../../../services/dashboard-loading-tasks.service';
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

  // Stub the loading-tasks service so the card doesn't drag in the real
  // SSE/HTTP DI chain; the card only reads `isCancelling` for the badge.
  let cancellingIds: Set<string>;

  beforeEach(async () => {
    cancellingIds = new Set<string>();
    await TestBed.configureTestingModule({
      imports: [DetectorCardComponent],
      providers: [
        ...provideZoneless(),
        {
          provide: DashboardLoadingTasksService,
          useValue: { isCancelling: (id: string) => cancellingIds.has(id) },
        },
      ],
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
    expect(el.querySelector('.empty-hint')).toBeNull();
  });

  it('should show an "Empty" hint instead of 0 on a zero-training detector', async () => {
    fixture.componentRef.setInput('detector', { ...mockDetector, num_training: 0, last_trained_at: null });
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    const hint = el.querySelector('.empty-hint');
    expect(hint).toBeTruthy();
    expect(hint?.textContent?.trim()).toBe('Empty');
    expect(component.isUntrained).toBe(true);
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

  it('should drop the inline Delete verb from the overflow menu but keep Browse', async () => {
    const el = fixture.nativeElement as HTMLElement;
    (el.querySelector('.overflow-btn') as HTMLElement).click();
    await settleZoneless(fixture);
    const overflow = Array.from(el.querySelectorAll('.menu-item')).map((b) => b.textContent?.trim());
    // Only Delete is inline (plus Load when unloaded); Browse stays in the ⋯ menu.
    expect(overflow).not.toContain('Delete');
    expect(overflow).toContain('Browse positives');
    expect(overflow).toContain('Rename');
  });

  it('should still list the inline Delete verb in the right-click context menu', async () => {
    const el = fixture.nativeElement as HTMLElement;
    el.dispatchEvent(new MouseEvent('contextmenu', { clientX: 10, clientY: 10, bubbles: true }));
    await settleZoneless(fixture);
    const full = Array.from(el.querySelectorAll('.menu-item')).map((b) => b.textContent?.trim());
    // Right-click stays complete: Delete returns alongside Browse.
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
    // "Export labels" and "Export model" are both in this menu and emit
    // different outputs, so match the full label rather than the prefix.
    const exportItem = items.find((b) =>
      b.textContent?.includes('Export labels'),
    );
    expect(exportItem).toBeTruthy();
    exportItem!.click();
    expect(component.export.emit).toHaveBeenCalled();
  });

  it('should offer "Move to AutoRun" on a draft detector and emit setAutorun(true)', async () => {
    vi.spyOn(component.setAutorun, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    (el.querySelector('.overflow-btn') as HTMLElement).click();
    await settleZoneless(fixture);
    const items = Array.from(el.querySelectorAll('.menu-item')) as HTMLElement[];
    const move = items.find((b) => b.textContent?.includes('Move to AutoRun'));
    expect(move).toBeTruthy();
    expect(items.some((b) => b.textContent?.includes('Move to Drafts'))).toBe(false);
    move!.click();
    expect(component.setAutorun.emit).toHaveBeenCalledWith(true);
  });

  describe('frozen (AutoRun) detector', () => {
    beforeEach(async () => {
      fixture.componentRef.setInput('detector', { ...mockDetector, autofind: true });
      await settleZoneless(fixture);
    });

    it('hides the rename pencil and inline delete button', () => {
      const el = fixture.nativeElement as HTMLElement;
      expect(component.frozen).toBe(true);
      expect(el.querySelector('.edit-btn')).toBeNull();
      expect(el.querySelector('.delete-btn')).toBeNull();
    });

    it('omits the editing verbs from the right-click menu but keeps read/use verbs', async () => {
      const el = fixture.nativeElement as HTMLElement;
      el.dispatchEvent(new MouseEvent('contextmenu', { clientX: 10, clientY: 10, bubbles: true }));
      await settleZoneless(fixture);
      const full = Array.from(el.querySelectorAll('.menu-item')).map((b) => b.textContent?.trim());
      expect(full).not.toContain('Rename');
      expect(full).not.toContain('Import Labels');
      expect(full).not.toContain('Delete');
      expect(full).toContain('Browse positives');
      expect(full).toContain('Stats');
      expect(full).toContain('Export labels');
    });

    it('offers "Move to Drafts" and emits setAutorun(false)', async () => {
      vi.spyOn(component.setAutorun, 'emit');
      const el = fixture.nativeElement as HTMLElement;
      (el.querySelector('.overflow-btn') as HTMLElement).click();
      await settleZoneless(fixture);
      const items = Array.from(el.querySelectorAll('.menu-item')) as HTMLElement[];
      const move = items.find((b) => b.textContent?.includes('Move to Drafts'));
      expect(move).toBeTruthy();
      expect(items.some((b) => b.textContent?.includes('Move to AutoRun'))).toBe(false);
      move!.click();
      expect(component.setAutorun.emit).toHaveBeenCalledWith(false);
    });
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

  it('shows a disabled "Cancelling…" badge on the loading row once its task is cancelling', async () => {
    const task = { task_id: 'det-1', status: 'running', current: 0, total: 0 };
    fixture.componentRef.setInput('loadingTask', task);
    await settleZoneless(fixture);
    let btn = (fixture.nativeElement as HTMLElement).querySelector('.jp__cancel') as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
    expect(btn.textContent?.trim()).toBe('Cancel');

    cancellingIds.add('det-1');
    fixture.componentRef.setInput('loadingTask', { ...task });
    await settleZoneless(fixture);

    btn = (fixture.nativeElement as HTMLElement).querySelector('.jp__cancel') as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.textContent?.trim()).toBe('Cancelling…');
  });
});
