import { Subject } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { TestBed } from '@angular/core/testing';

import { BrowseBinMemberGridComponent } from './browse-bin-member-grid.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * Contracts the member grid took over from ``BrowseBinPopupComponent`` when it
 * was split out (issue #3423). Both describe blocks drive the component
 * directly rather than through a rendered fixture: the grid is virtualized and
 * does not render individual entries reliably under jsdom, which is exactly why
 * these were written against the instance in the first place.
 */
describe('BrowseBinMemberGridComponent (scroll-prefetch re-wiring)', () => {
  it('re-subscribes prefetch when the member-grid viewport is recreated', () => {
    // Regression: the grid subscribed scrolledIndexChange only once, but the
    // viewport lives behind an `@if` and the bin details are reused across
    // summons (right-clicking another bin only swaps inputs). An empty→
    // populated transition created a fresh viewport whose stream was never
    // subscribed, so scrolling the member grid never hydrated thumbnails beyond
    // the initially-prefetched window. In production a constructor effect
    // tracking the `viewport` view-query signal re-runs ensureScrollSubscription
    // whenever the instance changes; here the query is stubbed as a plain
    // function and the method driven directly.
    const component = Object.create(
      BrowseBinMemberGridComponent.prototype,
    ) as BrowseBinMemberGridComponent;
    const state = component as unknown as {
      viewport: () => unknown;
      scrollSub: unknown;
      scrollSubscribedViewport: unknown;
      prefetchVisible(): void;
      ensureScrollSubscription(): void;
    };
    state.scrollSub = null;
    state.scrollSubscribedViewport = null;
    const prefetchSpy = vi.fn();
    state.prefetchVisible = prefetchSpy;

    const vp1 = { scrolledIndexChange: new Subject<number>() };
    state.viewport = () => vp1;
    state.ensureScrollSubscription();
    vp1.scrolledIndexChange.next(0);
    const callsAfterFirstScroll = prefetchSpy.mock.calls.length;
    expect(callsAfterFirstScroll).toBeGreaterThan(0);

    // Viewport destroyed (the bin emptied, or a previewOnly summon) …
    vp1.scrolledIndexChange.complete();
    state.viewport = () => undefined;
    state.ensureScrollSubscription();

    // … then a new multi-member summon creates a fresh instance.
    const vp2 = { scrolledIndexChange: new Subject<number>() };
    state.viewport = () => vp2;
    state.ensureScrollSubscription();

    const callsBeforeSecondScroll = prefetchSpy.mock.calls.length;
    vp2.scrolledIndexChange.next(2);
    expect(prefetchSpy.mock.calls.length).toBe(callsBeforeSecondScroll + 1);
  });
});

/**
 * DOM-focus half of the keyboard walk.
 *
 * Arrow keys move the shell's viewed item, but before this they left DOM focus
 * behind on whatever entry the user last tabbed/clicked. Enter is only caught by
 * the focused entry's own handler, so it toggled the stale DOM-focused entry
 * rather than the arrow-walked one; Space fired both the shell's document
 * fallback and the focused entry, double-toggling. The fix keeps DOM focus glued
 * to the walked entry, and lets the focused entry own its activation without the
 * fallback double-firing. The shell drives this through `revealAndFocus`.
 */
describe('BrowseBinMemberGridComponent (keyboard focus sync)', () => {
  interface GridState {
    ids: () => number[];
    rows: () => number[][];
    columns: () => number;
    rowSize: () => number;
    viewport: () => unknown;
    host: { nativeElement: { querySelector: (sel: string) => HTMLElement | null } };
  }

  function makeGrid(): { component: BrowseBinMemberGridComponent; state: GridState } {
    const component = Object.create(
      BrowseBinMemberGridComponent.prototype,
    ) as BrowseBinMemberGridComponent;
    const state = component as unknown as GridState;
    state.ids = () => [10, 20, 30, 40];
    state.rows = () => [
      [10, 20],
      [30, 40],
    ];
    state.columns = () => 2;
    state.rowSize = () => 88;
    // No viewport: `revealAndFocus`'s scroll half is a no-op, leaving the focus
    // half — the part these tests are about — to run on its own.
    state.viewport = () => undefined;
    return { component, state };
  }

  let rafSpy: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    // Run rAF synchronously so `focusEntry`'s deferred (and retried) focus lands
    // within the test tick, deterministically.
    rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      cb(0);
      return 0;
    });
  });
  afterEach(() => rafSpy.mockRestore());

  it('moves DOM focus to the entry the shell walked to', () => {
    const { component, state } = makeGrid();
    const walked = { focus: vi.fn() } as unknown as HTMLElement;
    state.host = {
      nativeElement: {
        // Only the entry for id 20 (index 1) exists.
        querySelector: (sel: string) => (sel.includes('"20"') ? walked : null),
      },
    };

    component.revealAndFocus(1);

    expect(walked.focus).toHaveBeenCalledWith({ preventScroll: true });
  });

  it('retries the focus until the virtualized entry renders, then focuses it', () => {
    const { component, state } = makeGrid();
    const walked = { focus: vi.fn() } as unknown as HTMLElement;
    let calls = 0;
    state.host = {
      nativeElement: {
        // Absent for the first two frames (row still virtualizing in), then in.
        querySelector: (sel: string) => {
          if (!sel.includes('"20"')) return null;
          return ++calls >= 3 ? walked : null;
        },
      },
    };

    component.revealAndFocus(1);

    expect(walked.focus).toHaveBeenCalledTimes(1);
  });

  it('lets the focused entry own its activation, stopping the fallback double-toggle', () => {
    const { component } = makeGrid();
    const emitted: number[] = [];
    (component as unknown as { entryClick: { emit: (id: number) => void } }).entryClick = {
      emit: (id: number) => emitted.push(id),
    };
    const event = {
      key: 'Enter',
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    } as unknown as KeyboardEvent;

    component.onEntryKeydown(event, 42);

    expect(event.preventDefault).toHaveBeenCalled();
    // The bubble is stopped so the shell's document-level Space fallback (which
    // acts on the viewed item) can't also fire and cancel this toggle.
    expect(event.stopPropagation).toHaveBeenCalled();
    expect(emitted).toEqual([42]);
  });
});

/**
 * The People view (#5590) reuses the grid for a person's faces, where a click
 * shows a face large rather than selecting it, and there is no Browse
 * selection to select into.
 */
describe('BrowseBinMemberGridComponent (not selectable)', () => {
  it('draws no select-all, needs no selection service, and says what a click does', async () => {
    configureZoneless({ imports: [BrowseBinMemberGridComponent], providers: [...provideHttpTesting()] });
    const fixture = TestBed.createComponent(BrowseBinMemberGridComponent);
    fixture.componentRef.setInput('ids', [1, 2, 3]);
    fixture.componentRef.setInput('rows', [[1, 2, 3]]);
    fixture.componentRef.setInput('columns', 3);
    fixture.componentRef.setInput('selectable', false);
    fixture.componentRef.setInput('countNoun', 'face');
    // The grid asks for its entries' metadata, which keeps the app unstable
    // while it is out; drain a macrotask rather than wait for stability.
    await settleResource();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.bin-popup-select-all')).toBeNull();
    expect(el.querySelector('.bin-popup-count')?.textContent?.trim()).toBe('3 faces');
    const grid = fixture.componentInstance;
    expect(grid.isSelected(1)).toBe(false);
    expect(grid.selectionState).toBe('none');
    expect(grid.entryLabel(1)).toBe('Clip #1 (click to show it large)');
    fixture.destroy();
  });
});
