import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { LeftPanelComponent } from './left-panel.component';
import type { Media } from '../../models/api.models';
import { settleResource } from '../../testing/settle-resource';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../testing/test-providers';

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
});
