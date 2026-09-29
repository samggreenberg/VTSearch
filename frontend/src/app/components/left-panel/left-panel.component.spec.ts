import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { LeftPanelComponent } from './left-panel.component';
import type { Media } from '../../models/api.models';
import { settleResource } from '../../testing/settle-resource';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../testing/test-providers';
import { FLOOR_STATES, lineFloor } from '../../testing/line-floor';

describe('LeftPanelComponent', () => {
  let component: LeftPanelComponent;
  let fixture: ComponentFixture<LeftPanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LeftPanelComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(LeftPanelComponent);
    component = fixture.componentInstance;
    TestBed.tick();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should default to autopilot tab', () => {
    expect(component.activeTab()).toBe('autopilot');
  });

  it('should emit autopilotStart on init', () => {
    const fresh = TestBed.createComponent(LeftPanelComponent);
    const comp = fresh.componentInstance;
    vi.spyOn(comp.autopilotStart, 'emit');
    TestBed.tick();
    expect(comp.autopilotStart.emit).toHaveBeenCalled();
  });

  it('should switch to manual tab', () => {
    component.setTab('manual');
    expect(component.activeTab()).toBe('manual');
  });

  it('should default to manual tab when autopilotEnabled is false', () => {
    const fresh = TestBed.createComponent(LeftPanelComponent);
    const comp = fresh.componentInstance;
    fresh.componentRef.setInput('autopilotEnabled', false);
    vi.spyOn(comp.autopilotStart, 'emit');
    TestBed.tick();
    expect(comp.activeTab()).toBe('manual');
    expect(comp.autopilotStart.emit).not.toHaveBeenCalled();
  });

  it('should default to manual and not start autopilot when autopilotDisabled', () => {
    const fresh = TestBed.createComponent(LeftPanelComponent);
    const comp = fresh.componentInstance;
    fresh.componentRef.setInput('autopilotDisabled', true);
    vi.spyOn(comp.autopilotStart, 'emit');
    TestBed.tick();
    expect(comp.activeTab()).toBe('manual');
    expect(comp.autopilotStart.emit).not.toHaveBeenCalled();
  });

  it('should fall back to manual when autopilot becomes disabled after starting', () => {
    // Default init lands on the autopilot tab.
    expect(component.activeTab()).toBe('autopilot');
    vi.spyOn(component.autopilotStop, 'emit');
    fixture.componentRef.setInput('autopilotDisabled', true);
    TestBed.tick();
    expect(component.activeTab()).toBe('manual');
    expect(component.autopilotStop.emit).toHaveBeenCalled();
  });

  it('should ignore clicks on the autopilot tab while it is disabled', () => {
    component.setTab('manual');
    fixture.componentRef.setInput('autopilotDisabled', true);
    TestBed.tick();
    vi.spyOn(component.autopilotStart, 'emit');
    component.setTab('autopilot');
    expect(component.activeTab()).toBe('manual');
    expect(component.autopilotStart.emit).not.toHaveBeenCalled();
  });

  it('should disable the autopilot tab button when autopilotDisabled', () => {
    fixture.componentRef.setInput('autopilotDisabled', true);
    TestBed.tick();
    const el = fixture.nativeElement as HTMLElement;
    const tabs = el.querySelectorAll<HTMLButtonElement>('.left-tab');
    // Second tab is Autopilot.
    expect(tabs[1].disabled).toBe(true);
  });

  it('should render autopilot tab content by default', () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.tab-panel-autopilot')).toBeTruthy();
    expect(el.querySelector('.tab-panel-manual')).toBeNull();
  });

  it('should render manual tab content when switched', () => {
    const el = fixture.nativeElement as HTMLElement;
    (el.querySelectorAll<HTMLButtonElement>('.left-tab')[0]).click(); // Manual
    TestBed.tick();
    expect(el.querySelector('.tab-panel-manual')).toBeTruthy();
    expect(el.querySelector('.tab-panel-autopilot')).toBeNull();
  });

  it('should show active class on selected tab', () => {
    const el = fixture.nativeElement as HTMLElement;
    const tabs = el.querySelectorAll<HTMLButtonElement>('.left-tab');
    // Default: autopilot is active (second tab)
    expect(tabs[0].classList.contains('active')).toBe(false);
    expect(tabs[1].classList.contains('active')).toBe(true);

    tabs[0].click(); // Manual
    TestBed.tick();
    const tabsAfter = el.querySelectorAll('.left-tab');
    expect(tabsAfter[0].classList.contains('active')).toBe(true);
    expect(tabsAfter[1].classList.contains('active')).toBe(false);
  });

  it('should emit autopilotStart when switching to autopilot tab', () => {
    component.setTab('manual');
    vi.spyOn(component.autopilotStart, 'emit');
    component.setTab('autopilot');
    expect(component.autopilotStart.emit).toHaveBeenCalled();
  });

  it('should emit autopilotStop when switching from autopilot to manual tab', () => {
    vi.spyOn(component.autopilotStop, 'emit');
    component.setTab('manual');
    expect(component.autopilotStop.emit).toHaveBeenCalled();
  });

  it('should not emit start when setting the same tab', () => {
    vi.spyOn(component.autopilotStart, 'emit');
    component.setTab('autopilot');
    expect(component.autopilotStart.emit).not.toHaveBeenCalled();
  });

  it('should emit autopilotRefocus when clicking already-active autopilot tab', () => {
    vi.spyOn(component.autopilotRefocus, 'emit');
    component.setTab('autopilot');
    expect(component.autopilotRefocus.emit).toHaveBeenCalled();
  });

  it('should not emit autopilotRefocus when clicking already-active manual tab', () => {
    component.setTab('manual');
    vi.spyOn(component.autopilotRefocus, 'emit');
    component.setTab('manual');
    expect(component.autopilotRefocus.emit).not.toHaveBeenCalled();
  });

  it('should emit sortModeChange', () => {
    vi.spyOn(component.sortModeChange, 'emit');
    component.sortModeChange.emit('learned');
    expect(component.sortModeChange.emit).toHaveBeenCalledWith('learned');
  });

  it('should emit mediaSelect', () => {
    vi.spyOn(component.mediaSelect, 'emit');
    component.mediaSelect.emit(42);
    expect(component.mediaSelect.emit).toHaveBeenCalledWith(42);
  });

  /**
   * The grid lists every item, labeled or not, each with its own vote badge —
   * so the count in the header says nothing about whether any of them are
   * still worth clicking. The note is what distinguishes "there is nothing
   * left to pick" from "what I want is further down" (#4028).
   */
  describe('the "All labeled" note (#4028)', () => {
    const stub = (id: number): Media => ({ id, media_type: 'image' }) as Media;
    const note = () =>
      (fixture.nativeElement as HTMLElement).querySelector('.all-labeled-note');

    function show(inputs: Record<string, unknown>): void {
      component.setTab('manual');
      for (const [k, v] of Object.entries(inputs)) fixture.componentRef.setInput(k, v);
      TestBed.tick();
    }

    it('stays away while an item is unlabeled', () => {
      show({
        medias: [stub(1), stub(2)],
        goodVotes: new Set([1]),
        badVotes: new Set<number>(),
      });
      expect(component.allLabeled()).toBe(false);
      expect(note()).toBeNull();
    });

    it('appears once every item in the grid carries a label', () => {
      show({
        medias: [stub(1), stub(2)],
        goodVotes: new Set([1]),
        badVotes: new Set([2]),
      });
      expect(component.allLabeled()).toBe(true);
      expect(note()!.textContent).toContain('All labeled');
    });

    it('stays away in Find mode, where the queue is measured by verified', () => {
      show({
        panelMode: 'find',
        medias: [stub(1)],
        goodVotes: new Set([1]),
        badVotes: new Set<number>(),
      });
      expect(component.allLabeled()).toBe(false);
    });

    it('stays away on an empty grid — nothing loaded is not nothing left', () => {
      show({ medias: [], goodVotes: new Set<number>(), badVotes: new Set<number>() });
      expect(component.allLabeled()).toBe(false);
    });
  });

  /**
   * Before any sort runs, Manual's ranking is empty while the dataset is not.
   * The grid must say how to get items on screen, not "Nothing to show",
   * which reads as an empty dataset (#4157).
   */
  describe('Manual grid before a sort (#4157)', () => {
    const stub = (id: number): Media => ({ id, media_type: 'image' }) as Media;
    const empty = () =>
      (fixture.nativeElement as HTMLElement).querySelector('.empty-list');

    function show(inputs: Record<string, unknown>): void {
      component.setTab('manual');
      for (const [k, v] of Object.entries(inputs)) fixture.componentRef.setInput(k, v);
      TestBed.tick();
    }

    it('asks for a sort order instead of claiming there is nothing', () => {
      show({ medias: [stub(1), stub(2)], sortOrder: [] });
      expect(empty()!.textContent).toContain('Choose a sort order above');
      expect(empty()!.textContent).not.toContain('Nothing to show');
    });

    it('says it is sorting while a sort is in flight', () => {
      show({ medias: [stub(1)], sortOrder: [], sortBusy: true });
      expect(empty()!.textContent).toContain('Sorting');
    });

    it('still reports an empty dataset as such', () => {
      show({ medias: [], sortOrder: [] });
      expect(empty()!.textContent).toContain('No media loaded');
    });
  });

  describe('grid header (mediaTypeName)', () => {
    const stub = (media_type: string): Media => ({ id: 1, media_type }) as Media;

    function setMedias(medias: Media[]): void {
      fixture.componentRef.setInput('medias', medias);
    }

    it('derives the type label from the first grid item', () => {
      setMedias([stub('audio')]);
      expect(component.mediaTypeName()).toBe('Audio');
    });

    it('resets to "Media" when the grid empties (no stale type label)', () => {
      setMedias([stub('audio')]);
      expect(component.mediaTypeName()).toBe('Audio');
      // Switching to an empty grid must clear the previous type, not keep it.
      setMedias([]);
      expect(component.mediaTypeName()).toBe('Media');
    });

    it('re-derives when the grid switches media type', () => {
      setMedias([stub('audio')]);
      expect(component.mediaTypeName()).toBe('Audio');
      setMedias([stub('image')]);
      expect(component.mediaTypeName()).toBe('Image');
    });

    it('upgrades from the fallback to the display name when type metadata loads after the grid', async () => {
      const httpMock = TestBed.inject(HttpTestingController);
      // The media-types read rides `rxResource`, whose loader runs in a root
      // effect rather than during `detectChanges()`; tick so the GET is issued.
      TestBed.tick();
      // The grid populates before the getMediaTypes() request resolves, so the
      // header first shows the capitalized fallback.
      setMedias([stub('audio')]);
      expect(component.mediaTypeName()).toBe('Audio');

      // Metadata arrives late with a custom display name: the header must
      // upgrade instead of staying stuck on the fallback. The resource value
      // commits on a microtask, so settle before asserting. Two GETs match
      // here — the header's own rxResource read and the shared
      // MediaTypeCapabilityService.ensureLoaded() fired in ngOnInit — so flush
      // them all with the same payload.
      const reqs = httpMock.match((r) => r.url.includes('/api/media-types'));
      reqs.forEach((r) => r.flush({ media_types: [{ type_id: 'audio', name: 'Sound Clips' }] }));
      await settleResource();
      expect(component.mediaTypeName()).toBe('Sound Clips');
    });
  });

  /**
   * The line always keeps a set, checked or not (#4272): the Find work-queue
   * actions (Browse / To Dataset / Export) gate on the positives above it in
   * every state, and the line draws the same in each (#4273). A null cut used
   * to disable them silently (#4247).
   */
  describe('in every floor state (#4272)', () => {
    const stub = (id: number): Media => ({ id, media_type: 'image' }) as Media;
    const ranking = [
      { id: 1, score: 0.9 },
      { id: 2, score: 0.6 },
      { id: 3, score: 0.4 },
    ];

    function show(panelMode: 'label' | 'find', floor: ReturnType<typeof lineFloor>): HTMLElement {
      fixture.componentRef.setInput('panelMode', panelMode);
      fixture.componentRef.setInput('medias', ranking.map(({ id }) => stub(id)));
      fixture.componentRef.setInput('sortOrder', ranking);
      fixture.componentRef.setInput('threshold', 0.5);
      fixture.componentRef.setInput('floor', floor);
      if (panelMode === 'label') component.setTab('manual');
      TestBed.tick();
      return fixture.nativeElement as HTMLElement;
    }

    it.each(FLOOR_STATES)('counts the unverified positives above the line when %s', (status) => {
      show('find', lineFloor(status));
      expect(component.unverifiedGoodCount).toBe(2);
    });

    it.each(FLOOR_STATES)('draws the plain line in Find and Label when %s', (status) => {
      for (const mode of ['find', 'label'] as const) {
        const line = show(mode, lineFloor(status)).querySelector('.media-threshold-line')!;
        expect(line.className).toBe('media-threshold-line');
        expect(line.textContent!.trim().toLowerCase()).toBe('threshold');
      }
    });

    it.each(FLOOR_STATES)('offers the check in both places the floor control lives when %s', (status) => {
      const emitted = vi.spyOn(component.floorCheck, 'emit');
      for (const mode of ['find', 'label'] as const) {
        const btn = show(mode, lineFloor(status)).querySelector('.floor-check-btn') as HTMLButtonElement;
        expect(btn.textContent!.trim()).toBe('Check 5 picks');
        btn.click();
      }
      expect(emitted).toHaveBeenCalledTimes(2);
    });

    it('holds the check while a sort is running', () => {
      fixture.componentRef.setInput('sortBusy', true);
      const btn = show('find', lineFloor('unchecked')).querySelector('.floor-check-btn') as HTMLButtonElement;
      expect(btn.disabled).toBe(true);
    });
  });

  /**
   * The Find row (#4246): the precision floor beside the unverified-positives
   * work-queue actions. The actions scope over the unverified items above the
   * line, so they gate on `unverifiedGoodCount`, which counts exactly those.
   */
  describe('the Find row', () => {
    const stub = (id: number): Media => ({ id, media_type: 'image' }) as Media;
    const actions = ['Browse unverified positives', 'Unverified positives to dataset', 'Export unverified positives'];

    function find(sortOrder: { id: number; score: number }[] | null, threshold: number | null): HTMLElement {
      fixture.componentRef.setInput('panelMode', 'find');
      fixture.componentRef.setInput('medias', (sortOrder ?? []).map(({ id }) => stub(id)));
      fixture.componentRef.setInput('sortOrder', sortOrder);
      fixture.componentRef.setInput('threshold', threshold);
      TestBed.tick();
      return fixture.nativeElement as HTMLElement;
    }

    const button = (el: HTMLElement, label: string) =>
      el.querySelector(`.find-floor-row button[aria-label="${label}"]`) as HTMLButtonElement;

    describe('unverifiedGoodCount', () => {
      it('counts the items at or above the line', () => {
        find([{ id: 1, score: 0.9 }, { id: 2, score: 0.5 }, { id: 3, score: 0.49 }], 0.5);
        expect(component.unverifiedGoodCount).toBe(2);
      });

      it('is zero with no ranking or no line', () => {
        find(null, 0.5);
        expect(component.unverifiedGoodCount).toBe(0);
        find([{ id: 1, score: 0.9 }], null);
        expect(component.unverifiedGoodCount).toBe(0);
      });

      it('is zero when nothing clears the line', () => {
        find([{ id: 1, score: 0.2 }], 0.5);
        expect(component.unverifiedGoodCount).toBe(0);
      });
    });

    it('mounts the floor control, not the Inclusion stepper', () => {
      const el = find([{ id: 1, score: 0.9 }], 0.5);
      expect(el.querySelector('.find-floor-row vt-precision-floor')).not.toBeNull();
      expect(el.querySelector('#inclusion-input')).toBeNull();
    });

    it('enables the work-queue actions only while an unverified positive exists', () => {
      let el = find([{ id: 1, score: 0.2 }], 0.5);
      actions.forEach((label) => expect(button(el, label).disabled).toBe(true));
      el = find([{ id: 1, score: 0.9 }], 0.5);
      actions.forEach((label) => expect(button(el, label).disabled).toBe(false));
    });

    it('disables the work-queue actions while Find is waiting', () => {
      fixture.componentRef.setInput('disabled', true);
      const el = find([{ id: 1, score: 0.9 }], 0.5);
      actions.forEach((label) => expect(button(el, label).disabled).toBe(true));
    });

    it('emits each work-queue action', () => {
      const el = find([{ id: 1, score: 0.9 }], 0.5);
      const browse = vi.spyOn(component.browse, 'emit');
      const toDataset = vi.spyOn(component.toDataset, 'emit');
      const exported = vi.spyOn(component.unverifiedExport, 'emit');
      actions.forEach((label) => button(el, label).click());
      expect(browse).toHaveBeenCalledOnce();
      expect(toDataset).toHaveBeenCalledOnce();
      expect(exported).toHaveBeenCalledOnce();
    });

    it('shows the floor, its state and the count the line returns', () => {
      fixture.componentRef.setInput('minPrecision', 0.75);
      fixture.componentRef.setInput('floor', lineFloor('confirmed', { minPrecision: 0.75 }));
      fixture.componentRef.setInput('returned', 212);
      const el = find([{ id: 1, score: 0.9 }], 0.5);
      const select = el.querySelector('.find-floor-row select') as HTMLSelectElement;
      expect(select.value).toBe('0.75');
      expect(el.querySelector('.find-floor-row .floor-state')!.textContent).toContain('At least 75% right · likely 55–100% (checked 5) · 32 kept');
    });

    it('forwards a picked floor as minPrecisionChange', () => {
      const el = find([{ id: 1, score: 0.9 }], 0.5);
      const emitted = vi.spyOn(component.minPrecisionChange, 'emit');
      const select = el.querySelector('.find-floor-row select') as HTMLSelectElement;
      select.value = '0.9';
      select.dispatchEvent(new Event('change'));
      expect(emitted).toHaveBeenCalledWith(0.9);
    });
  });

  it('mounts the floor control in the Manual tab', () => {
    component.setTab('manual');
    TestBed.tick();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.tab-panel-manual vt-precision-floor')).not.toBeNull();
  });
});
