import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { vi } from 'vitest';

import { MediaListComponent } from './media-list.component';
import { Media } from '../../../models/api.models';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('MediaListComponent', () => {
  let component: MediaListComponent;
  let fixture: ComponentFixture<MediaListComponent>;

  const mockMedias: Media[] = [
    { id: 1, media_type: 'audio', filename: 'a.wav', md5: 'a1', custom_metadata: {} },
    { id: 2, media_type: 'audio', filename: 'b.wav', md5: 'b2', custom_metadata: {} },
    { id: 3, media_type: 'audio', filename: 'c.wav', md5: 'c3', custom_metadata: {} },
  ];

  /** Spy on the (test-setup-stubbed) scrollIntoView so the selection
   *  autoscroll — queued by the effect, performed in ngAfterViewChecked — is
   *  observable. Cleared on install because the prototype spy persists across
   *  tests in this file. */
  function spyOnScrollIntoView() {
    const spy = vi.spyOn(Element.prototype, 'scrollIntoView');
    spy.mockClear();
    return spy;
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaListComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(MediaListComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('medias', mockMedias);
    TestBed.tick();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should render media items in natural order when no sortOrder', () => {
    const items = component.cachedOrderedItems;
    expect(items.length).toBe(3);
    expect(items[0].media.id).toBe(1);
    expect(items[1].media.id).toBe(2);
    expect(items[2].media.id).toBe(3);
  });

  it('should render media items in sort order when provided', () => {
    fixture.componentRef.setInput('sortOrder', [
      { id: 3, score: 0.9 },
      { id: 1, score: 0.5 },
      { id: 2, score: 0.2 },
    ]);
    TestBed.tick();
    const items = component.cachedOrderedItems;
    expect(items[0].media.id).toBe(3);
    expect(items[1].media.id).toBe(1);
    expect(items[2].media.id).toBe(2);
  });

  it('should insert threshold line at correct position', () => {
    fixture.componentRef.setInput('sortOrder', [
      { id: 3, score: 0.9 },
      { id: 1, score: 0.5 },
      { id: 2, score: 0.2 },
    ]);
    fixture.componentRef.setInput('threshold', 0.4);
    TestBed.tick();
    const items = component.cachedOrderedItems;
    // Threshold is at 0.4, so item with score 0.2 should have showThreshold
    expect(items[0].showThreshold).toBe(false);
    expect(items[1].showThreshold).toBe(false);
    expect(items[2].showThreshold).toBe(true);
  });

  it('should return correct vote labels', () => {
    fixture.componentRef.setInput('goodVotes', new Set([1]));
    fixture.componentRef.setInput('badVotes', new Set([2]));
    expect(component.getVoteLabel(1)).toBe('good');
    expect(component.getVoteLabel(2)).toBe('bad');
    expect(component.getVoteLabel(3)).toBeNull();
  });

  it('should emit mediaSelect on item select', () => {
    vi.spyOn(component.mediaSelect, 'emit');
    component.onMediaSelect(2);
    expect(component.mediaSelect.emit).toHaveBeenCalledWith(2);
  });

  it('should show empty message when no medias', () => {
    fixture.componentRef.setInput('medias', []);
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('.empty-list')?.textContent).toContain('No media loaded');
  });

  it('should render threshold line in DOM', () => {
    fixture.componentRef.setInput('sortOrder', [
      { id: 1, score: 0.8 },
      { id: 2, score: 0.3 },
    ]);
    fixture.componentRef.setInput('threshold', 0.5);
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('.media-threshold-line')).toBeTruthy();
  });

  it('autoscrolls the newly-selected item into view in click mode', () => {
    const scrollSpy = spyOnScrollIntoView();
    fixture.componentRef.setInput('focusMode', 'click');
    TestBed.tick();
    fixture.componentRef.setInput('selectedId', 2);
    TestBed.tick();
    expect(scrollSpy).toHaveBeenCalled();
  });

  // Regression: clicking a card in this grid selects an item that is, by
  // definition, already on screen. Autoscrolling it into view is jarring and
  // pointless, so a selection that originated from onMediaSelect must not
  // trigger an autoscroll — even in click mode.
  it('does not autoscroll for a selection driven by clicking a card', () => {
    const scrollSpy = spyOnScrollIntoView();
    fixture.componentRef.setInput('focusMode', 'click');
    TestBed.tick();
    component.onMediaSelect(2);
    fixture.componentRef.setInput('selectedId', 2);
    TestBed.tick();
    expect(scrollSpy).not.toHaveBeenCalled();
  });

  // A click on card 2 must not suppress a *later* off-screen selection (e.g. a
  // vote auto-advance to card 3): only the clicked id is exempt from autoscroll.
  it('still autoscrolls a subsequent selection from elsewhere after a click', () => {
    const scrollSpy = spyOnScrollIntoView();
    fixture.componentRef.setInput('focusMode', 'click');
    TestBed.tick();
    component.onMediaSelect(2);
    fixture.componentRef.setInput('selectedId', 2);
    TestBed.tick();
    expect(scrollSpy).not.toHaveBeenCalled();
    fixture.componentRef.setInput('selectedId', 3);
    TestBed.tick();
    expect(scrollSpy).toHaveBeenCalled();
  });

  // Regression: in hover mode the selection follows the cursor. Scrolling the
  // hovered item to the top shifts what's under the mouse, which re-selects and
  // scrolls again — an infinite autoscroll loop. Hover selection must not
  // trigger an autoscroll.
  it('does not autoscroll on hover selection', () => {
    const scrollSpy = spyOnScrollIntoView();
    fixture.componentRef.setInput('focusMode', 'hover');
    TestBed.tick();
    fixture.componentRef.setInput('selectedId', 2);
    TestBed.tick();
    expect(scrollSpy).not.toHaveBeenCalled();
  });

  // Regression: small datasets used to never prefetch metadata, so the list
  // displayed ``#284`` placeholders forever; names only filled in when the
  // user clicked an item. Without virtual scroll active, every row is in the
  // DOM, so every row is visible and should hydrate eagerly.
  it('eagerly prefetches metadata for every row when virtual scroll is off', () => {
    const cache = TestBed.inject(MediaMetadataCacheService);
    const spy = vi.spyOn(cache, 'ensureLoaded').mockImplementation(() => {});
    fixture.componentRef.setInput('medias', [...mockMedias]);
    TestBed.tick();
    expect(spy).toHaveBeenCalled();
    const ids = spy.mock.lastCall![0] as number[];
    expect(ids).toEqual([1, 2, 3]);
  });
});

describe('MediaListComponent scroll-prefetch re-wiring', () => {
  let component: MediaListComponent;
  let fixture: ComponentFixture<MediaListComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaListComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();
    fixture = TestBed.createComponent(MediaListComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('medias', []);
    TestBed.tick();
  });

  function fakeViewport() {
    return {
      scrolledIndexChange: new Subject<number>(),
      elementRef: { nativeElement: document.createElement('div') },
    };
  }

  /** Replace the ``virtualViewport`` view-query signal with a stub returning
   *  the given fake (the readonly field is a plain property at runtime). */
  function setViewport(vp: unknown): void {
    (component as never as { virtualViewport: unknown }).virtualViewport = () => vp;
  }

  it('re-subscribes prefetch when the viewport instance is recreated', () => {
    // Regression: a one-shot `scrollSubscribed` flag outlived the CDK
    // viewport; after a dataset switch destroyed and recreated it, the new
    // instance's scrolledIndexChange was never subscribed, so scroll-driven
    // metadata prefetch silently died and rows showed stub names forever.
    const prefetchSpy = vi
      .spyOn(component as never as { prefetchVisibleMetadata(): void }, 'prefetchVisibleMetadata')
      .mockImplementation(() => undefined);

    const vp1 = fakeViewport();
    setViewport(vp1);
    component.ngAfterViewChecked();
    vp1.scrolledIndexChange.next(0);
    expect(prefetchSpy).toHaveBeenCalledTimes(1);

    // Simulate the @if branch destroying the viewport (its stream completes)
    // and a new instance appearing after the next dataset renders.
    vp1.scrolledIndexChange.complete();
    const vp2 = fakeViewport();
    setViewport(vp2);
    component.ngAfterViewChecked();

    vp2.scrolledIndexChange.next(3);
    expect(prefetchSpy).toHaveBeenCalledTimes(2);
  });

  describe('windowed "Load more"', () => {
    it('appends a trailing load-more grid row when hasMore is set', () => {
      fixture.componentRef.setInput('sortOrder', [
        { id: 1, score: 0.9 },
        { id: 2, score: 0.5 },
      ]);
      fixture.componentRef.setInput('hasMore', true);
      TestBed.tick();
      const rows = component.gridRows;
      expect(rows[rows.length - 1].kind).toBe('loadmore');
    });

    it('has no load-more row when hasMore is false', () => {
      fixture.componentRef.setInput('sortOrder', [{ id: 1, score: 0.9 }]);
      fixture.componentRef.setInput('hasMore', false);
      TestBed.tick();
      expect(component.gridRows.some((r) => r.kind === 'loadmore')).toBe(false);
    });

    it('onLoadMore emits when hasMore and not loading', () => {
      fixture.componentRef.setInput('hasMore', true);
      TestBed.tick();
      const spy = vi.fn();
      component.loadMore.subscribe(spy);
      component.onLoadMore();
      expect(spy).toHaveBeenCalledTimes(1);
    });

    it('onLoadMore is inert while a page fetch is in flight', () => {
      fixture.componentRef.setInput('hasMore', true);
      fixture.componentRef.setInput('loadingMore', true);
      TestBed.tick();
      const spy = vi.fn();
      component.loadMore.subscribe(spy);
      component.onLoadMore();
      expect(spy).not.toHaveBeenCalled();
    });
  });
});
