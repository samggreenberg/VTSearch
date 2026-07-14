import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { SimpleChange } from '@angular/core';
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
    expect(component.activeTab).toBe('autopilot');
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
    expect(component.activeTab).toBe('manual');
  });

  it('should default to manual tab when autopilotEnabled is false', () => {
    const fresh = TestBed.createComponent(LeftPanelComponent);
    const comp = fresh.componentInstance;
    comp.autopilotEnabled = false;
    vi.spyOn(comp.autopilotStart, 'emit');
    TestBed.tick();
    expect(comp.activeTab).toBe('manual');
    expect(comp.autopilotStart.emit).not.toHaveBeenCalled();
  });

  it('should default to manual and not start autopilot when autopilotDisabled', () => {
    const fresh = TestBed.createComponent(LeftPanelComponent);
    const comp = fresh.componentInstance;
    comp.autopilotDisabled = true;
    vi.spyOn(comp.autopilotStart, 'emit');
    TestBed.tick();
    expect(comp.activeTab).toBe('manual');
    expect(comp.autopilotStart.emit).not.toHaveBeenCalled();
  });

  it('should fall back to manual when autopilot becomes disabled after starting', () => {
    // Default init lands on the autopilot tab.
    expect(component.activeTab).toBe('autopilot');
    vi.spyOn(component.autopilotStop, 'emit');
    component.autopilotDisabled = true;
    component.ngOnChanges({
      autopilotDisabled: new SimpleChange(false, true, false),
    });
    expect(component.activeTab).toBe('manual');
    expect(component.autopilotStop.emit).toHaveBeenCalled();
  });

  it('should ignore clicks on the autopilot tab while it is disabled', () => {
    component.setTab('manual');
    component.autopilotDisabled = true;
    vi.spyOn(component.autopilotStart, 'emit');
    component.setTab('autopilot');
    expect(component.activeTab).toBe('manual');
    expect(component.autopilotStart.emit).not.toHaveBeenCalled();
  });

  it('should disable the autopilot tab button when autopilotDisabled', () => {
    // setInput fires ngOnChanges and marks the host dirty so the tick repaints
    // consistently (a direct field write leaves derived tab state unsettled,
    // which under zoneless surfaces as an NG0100 in the verify pass).
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
    // Drive the switch through the tab button's (click) — a bound listener that
    // marks the view dirty — so the panel repaints cleanly under zoneless
    // (calling setTab() directly leaves the host undirtied and the tab
    // conditional unsettled across the verify pass).
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

  describe('grid header (mediaTypeName)', () => {
    const stub = (media_type: string): Media => ({ id: 1, media_type }) as Media;

    function setMedias(medias: Media[]): void {
      component.medias = medias;
      component.ngOnChanges({
        medias: new SimpleChange(undefined, medias, false),
      });
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
      // commits on a microtask and the deriving effect runs on the next tick,
      // so settle before asserting.
      const req = httpMock.expectOne((r) => r.url.includes('/api/media-types'));
      req.flush({ media_types: [{ type_id: 'audio', name: 'Sound Clips' }] });
      await settleResource();
      expect(component.mediaTypeName()).toBe('Sound Clips');
    });
  });
});
