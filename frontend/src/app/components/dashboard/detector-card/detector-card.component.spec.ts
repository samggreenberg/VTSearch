import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { provideHttpTesting } from '../../../testing/test-providers';
import { DetectorCardComponent } from './detector-card.component';
import { DashboardLoadingTasksService } from '../../../services/dashboard-loading-tasks.service';
import { MediaTypeCapabilityService } from '../../../services/media-type-capability.service';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { wireVerdict } from '../../../testing/line-test';

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

    // The Type cell's icon comes from the served media-type registry, which
    // the dashboard loads; seed it as the backend serves it.
    TestBed.inject(MediaTypeCapabilityService).infos.set([
      { type_id: 'audio', name: 'Audio', icon: 'audio' },
      { type_id: 'face', name: 'Face', icon: 'face' },
      { type_id: 'point_cloud', name: 'Point Cloud', icon: 'cubes' },
    ]);

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

  // #5178: the cell once hardcoded five types' SVGs, so Face rows had none.
  it('shows the served icon for every media type, face and a third-party one included', async () => {
    for (const mediaType of ['face', 'point_cloud']) {
      fixture.componentRef.setInput('detector', { ...mockDetector, media_type: mediaType });
      await settleZoneless(fixture);
      const el = fixture.nativeElement as HTMLElement;
      expect(el.querySelector('.type-icon'), mediaType).toBeTruthy();
      expect(component.typeIcon).toBe(mediaType === 'face' ? 'face' : 'cubes');
    }
  });

  it('shows no icon for a type the registry does not list', async () => {
    fixture.componentRef.setInput('detector', { ...mockDetector, media_type: 'mystery' });
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('Mystery');
    expect(el.querySelector('.type-icon')).toBeNull();
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
    const exportItem = items.find((b) =>
      b.textContent?.includes('Export labels'),
    );
    expect(exportItem).toBeTruthy();
    exportItem!.click();
    expect(component.export.emit).toHaveBeenCalled();

    // The portable ONNX bundle is an expert affordance called directly against
    // POST /api/detectors/<id>/portable-bundle, deliberately not a menu item:
    // it read as a second, near-identical "Export" to every user who had no
    // use for it.
    expect(items.some((b) => /onnx|export model/i.test(b.textContent ?? ''))).toBe(false);
  });

  it('should offer "Move to AutoFind" on a draft detector and emit setAutofind(true)', async () => {
    vi.spyOn(component.setAutofind, 'emit');
    const el = fixture.nativeElement as HTMLElement;
    (el.querySelector('.overflow-btn') as HTMLElement).click();
    await settleZoneless(fixture);
    const items = Array.from(el.querySelectorAll('.menu-item')) as HTMLElement[];
    const move = items.find((b) => b.textContent?.includes('Move to AutoFind'));
    expect(move).toBeTruthy();
    expect(items.some((b) => b.textContent?.includes('Move to Drafts'))).toBe(false);
    move!.click();
    expect(component.setAutofind.emit).toHaveBeenCalledWith(true);
  });

  describe('auto-export (#4913)', () => {
    const status = {
      source_name: 'server_json_file',
      display_name: 'Server JSON File',
      inherited: true,
      missing_fields: [] as string[],
    };

    async function openMenu(): Promise<HTMLElement[]> {
      const el = fixture.nativeElement as HTMLElement;
      (el.querySelector('.overflow-btn') as HTMLElement).click();
      await settleZoneless(fixture);
      return Array.from(el.querySelectorAll('.menu-item')) as HTMLElement[];
    }

    it('offers no Auto-export… for a detector that syncs nowhere', async () => {
      const items = await openMenu();
      expect(items.some((b) => b.textContent?.includes('Auto-export'))).toBe(false);
    });

    it('offers Auto-export… to the creator and emits it', async () => {
      vi.spyOn(component.autoExport, 'emit');
      fixture.componentRef.setInput('detector', { ...mockDetector, created_by: 'sam', labelset_source_status: status });
      fixture.componentRef.setInput('currentUser', 'sam');
      await settleZoneless(fixture);
      const item = (await openMenu()).find((b) => b.textContent?.includes('Auto-export'));
      expect(item).toBeTruthy();
      item!.click();
      expect(component.autoExport.emit).toHaveBeenCalled();
    });

    it('lists it disabled for someone who did not create the detector', () => {
      fixture.componentRef.setInput('detector', { ...mockDetector, created_by: 'alice', labelset_source_status: status });
      fixture.componentRef.setInput('currentUser', 'sam');
      component.onContextMenu(new MouseEvent('contextmenu'));
      const item = component.contextMenuItems.find((i) => i.id === 'auto-export');
      expect(item?.disabled).toBe(true);
    });

    it('warns under the name while a value is missing, and the warning opens the dialog', async () => {
      vi.spyOn(component.autoExport, 'emit');
      const el = fixture.nativeElement as HTMLElement;
      expect(el.querySelector('.autoexport-warning')).toBeFalsy();

      fixture.componentRef.setInput('detector', {
        ...mockDetector,
        created_by: 'sam',
        labelset_source_status: { ...status, missing_fields: ['filepath'] },
      });
      fixture.componentRef.setInput('currentUser', 'sam');
      await settleZoneless(fixture);
      const warning = el.querySelector('.autoexport-warning') as HTMLButtonElement;
      expect(warning.textContent).toContain('Auto-export not set up');
      expect(warning.title).toContain('filepath');
      warning.click();
      expect(component.autoExport.emit).toHaveBeenCalled();
    });
  });

  describe('frozen (AutoFind) detector', () => {
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

    it('offers "Move to Drafts" and emits setAutofind(false)', async () => {
      vi.spyOn(component.setAutofind, 'emit');
      const el = fixture.nativeElement as HTMLElement;
      (el.querySelector('.overflow-btn') as HTMLElement).click();
      await settleZoneless(fixture);
      const items = Array.from(el.querySelectorAll('.menu-item')) as HTMLElement[];
      const move = items.find((b) => b.textContent?.includes('Move to Drafts'));
      expect(move).toBeTruthy();
      expect(items.some((b) => b.textContent?.includes('Move to AutoFind'))).toBe(false);
      move!.click();
      expect(component.setAutofind.emit).toHaveBeenCalledWith(false);
    });

    it('reads Untested under the name when it keeps no verdict', () => {
      const line = (fixture.nativeElement as HTMLElement).querySelector('.verdict-line') as HTMLElement;
      expect(line.textContent?.trim()).toBe('Untested');
      expect(line.classList).toContain('verdict-line--untested');
    });

    it('shows the latest verdict under the name, and marks a stale one (#4526)', async () => {
      fixture.componentRef.setInput('detector', { ...mockDetector, autofind: true, test_verdict: wireVerdict() });
      await settleZoneless(fixture);
      let line = (fixture.nativeElement as HTMLElement).querySelector('.verdict-line') as HTMLElement;
      expect(line.textContent?.trim()).toBe('Tested on drawings-new: likely 78 ± 8% right, about half of them found (34 picks, 2026-10-05)');
      expect(line.title).toContain('34 random picks');

      fixture.componentRef.setInput('detector', { ...mockDetector, autofind: true, test_verdict: wireVerdict({ stale: true }) });
      await settleZoneless(fixture);
      line = (fixture.nativeElement as HTMLElement).querySelector('.verdict-line') as HTMLElement;
      expect(line.textContent?.trim()).toMatch(/^⚠ Out of date · Tested on drawings-new/);
      expect(line.classList).toContain('verdict-line--stale');
      expect(line.title).toContain('retrained since');
    });
  });

  it('shows no verdict line on a draft', () => {
    expect((fixture.nativeElement as HTMLElement).querySelector('.verdict-line')).toBeNull();
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

describe('DetectorCardComponent: detector sharing', () => {
  let component: DetectorCardComponent;
  let fixture: ComponentFixture<DetectorCardComponent>;
  let http: HttpTestingController;

  const base = {
    id: 'd1',
    name: 'Cats',
    media_type: 'image',
    num_training: 12,
    created_by: 'me',
    sync_mode: 'local',
    sharing: null,
    sharing_note: null,
  };

  const NOTE = 'Shared through Team Drive, but detector sharing is off on this server; this is a local copy until that changes.';

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DetectorCardComponent],
      providers: [
        ...provideZoneless(),
        ...provideHttpTesting(),
        { provide: DashboardLoadingTasksService, useValue: { isCancelling: () => false } },
      ],
    }).compileComponents();
    http = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(DetectorCardComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('currentUser', 'me');
    fixture.componentRef.setInput('columnOrder', []);
  });

  afterEach(() => http.verify());

  async function render(detector: Record<string, unknown>): Promise<HTMLElement> {
    fixture.componentRef.setInput('detector', { ...base, ...detector });
    await settleZoneless(fixture);
    return fixture.nativeElement as HTMLElement;
  }

  async function menuLabels(el: HTMLElement): Promise<string[]> {
    (el.querySelector('.overflow-btn') as HTMLElement).click();
    await settleZoneless(fixture);
    return Array.from(el.querySelectorAll('.menu-item')).map((b) => b.textContent?.trim() ?? '');
  }

  it('shows no badge on a detector sharing has not touched', async () => {
    const el = await render({});
    expect(el.querySelector('.sharing-badge')).toBeNull();
    expect(el.querySelector('.name-edit-btn')).toBeTruthy();
  });

  it('badges a received detector, says who from and that it is read-only, and hides the pencil', async () => {
    const el = await render({
      sync_mode: 'linked',
      sharing: {
        remote_ref: 'c-1',
        source_name: 'team',
        remote_owner: 'ana@example.org',
        last_synced: 1700000000,
        integrity: { status: 'clean', count: 0 },
      },
    });
    const badge = el.querySelector('.sharing-badge') as HTMLElement;
    expect(badge.textContent?.trim()).toBe('Received');
    expect(badge.classList.contains('badge-download')).toBe(true);
    expect(badge.title).toContain('Received from ana@example.org');
    expect(badge.title).toContain('Read-only');
    expect(badge.title).toContain('Last checked for updates');
    expect(badge.title).toContain('owner’s');
    expect(el.querySelector('.name-edit-btn')).toBeNull();
    // A received draft keeps its inline Delete, which removes the local copy only.
    expect((el.querySelector('.delete-btn') as HTMLElement).title).toContain('local copy');
  });

  it('turns the Received badge amber when someone other than the owner wrote to the container', async () => {
    const el = await render({
      sync_mode: 'linked',
      sharing: { remote_ref: 'c-1', source_name: 'team', integrity: { status: 'foreign_modifications', count: 3 } },
    });
    const badge = el.querySelector('.sharing-badge') as HTMLElement;
    expect(badge.classList.contains('badge-warning')).toBe(true);
    expect(badge.title).toContain('3 votes were modified by someone other than the owner');
  });

  it("badges a shared detector and reads who it reaches from the container on the badge's first hover", async () => {
    const el = await render({
      sharing: { remote_ref: 'c-2', source_name: 'team', last_synced: 1700000000 },
      labelset_source_status: { source_name: 'team', display_name: 'Team Drive', inherited: false, missing_fields: [] },
    });
    const badge = el.querySelector('.sharing-badge') as HTMLElement;
    expect(badge.textContent?.trim()).toBe('Shared');
    expect(badge.title).toContain('Shared through Team Drive');
    expect(badge.title).not.toContain('recipient');

    badge.dispatchEvent(new MouseEvent('mouseenter'));
    badge.dispatchEvent(new MouseEvent('mouseenter'));
    http.expectOne('/api/detectors/registry/d1/sharing').flush({
      detector_id: 'd1',
      sync_mode: 'local',
      is_owner: true,
      recipients: ['bo', 'cy'],
      public: false,
    });
    await settleZoneless(fixture);
    expect((el.querySelector('.sharing-badge') as HTMLElement).title).toContain('Shared with 2 recipients');
  });

  it('shows the note on a row whose container is not in use here, and disables the sharing verbs with it', async () => {
    const el = await render({ sharing: { remote_ref: 'c-2', source_name: 'team' }, sharing_note: NOTE });
    const badge = el.querySelector('.sharing-badge') as HTMLElement;
    expect(badge.classList.contains('badge-warning')).toBe(true);
    expect(badge.title).toContain(NOTE);
    badge.dispatchEvent(new MouseEvent('mouseenter'));
    http.expectNone('/api/detectors/registry/d1/sharing');

    (el.querySelector('.overflow-btn') as HTMLElement).click();
    await settleZoneless(fixture);
    const share = Array.from(el.querySelectorAll('.menu-item')).find((b) => b.textContent?.includes('Share…')) as
      | HTMLButtonElement
      | undefined;
    expect(share?.disabled).toBe(true);
    expect(share?.title).toBe(NOTE);
  });

  it('shows "Remote sync failed" with the error, and opens Share… for the creator', async () => {
    const shareSpy = vi.fn();
    component.share.subscribe(shareSpy);
    const el = await render({ sharing: { remote_ref: 'c-2', source_name: 'team', sync_error: 'quota exceeded' } });
    const link = el.querySelector('.sync-error') as HTMLButtonElement;
    expect(link.textContent).toContain('Remote sync failed');
    expect(link.title).toContain('quota exceeded');
    link.click();
    expect(shareSpy).toHaveBeenCalledTimes(1);
  });

  it('offers Share… and Fork only once sharing is on, and emits each', async () => {
    let el = await render({});
    expect(await menuLabels(el)).not.toContain('Share…');
    component.dismissContextMenu();

    fixture.componentRef.setInput('sharingEnabled', true);
    el = await render({});
    const labels = await menuLabels(el);
    expect(labels).toEqual(expect.arrayContaining(['Share…', 'Fork']));

    const forkSpy = vi.fn();
    component.fork.subscribe(forkSpy);
    (Array.from(el.querySelectorAll('.menu-item')).find((b) => b.textContent?.includes('Fork')) as HTMLElement).click();
    expect(forkSpy).toHaveBeenCalledTimes(1);
  });

  it('lists Check for updates and Fork on a received row, and none of the editing verbs', async () => {
    const el = await render({ sync_mode: 'linked', sharing: { remote_ref: 'c-1', source_name: 'team' } });
    const labels = await menuLabels(el);
    expect(labels).toEqual(expect.arrayContaining(['Check for updates', 'Fork']));
    for (const absent of ['Rename', 'Import Labels', 'Auto-export…', 'Share…']) expect(labels).not.toContain(absent);

    const checkSpy = vi.fn();
    component.checkUpdates.subscribe(checkSpy);
    (Array.from(el.querySelectorAll('.menu-item')).find((b) => b.textContent?.includes('Check for updates')) as HTMLElement).click();
    expect(checkSpy).toHaveBeenCalledTimes(1);
  });
});
