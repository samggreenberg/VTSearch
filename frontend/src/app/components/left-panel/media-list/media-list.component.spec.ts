import { ComponentFixture, TestBed } from '@angular/core/testing';
import { SimpleChange } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { MediaListComponent } from './media-list.component';
import { Media } from '../../../models/api.models';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { provideZoneless } from '../../../testing/zoneless-testbed';

describe('MediaListComponent', () => {
  let component: MediaListComponent;
  let fixture: ComponentFixture<MediaListComponent>;

  const mockMedias: Media[] = [
    { id: 1, media_type: 'audio', filename: 'a.wav', md5: 'a1', custom_metadata: {} },
    { id: 2, media_type: 'audio', filename: 'b.wav', md5: 'b2', custom_metadata: {} },
    { id: 3, media_type: 'audio', filename: 'c.wav', md5: 'c3', custom_metadata: {} },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaListComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(MediaListComponent);
    component = fixture.componentInstance;
    component.medias = mockMedias;
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
    component.sortOrder = [
      { id: 3, score: 0.9 },
      { id: 1, score: 0.5 },
      { id: 2, score: 0.2 },
    ];
    // Programmatic input assignment doesn't fire ngOnChanges; trigger the
    // rebuild explicitly the way Angular would for a [sortOrder] binding.
    component.ngOnChanges({
      sortOrder: new SimpleChange(null, component.sortOrder, false),
    });
    TestBed.tick();
    const items = component.cachedOrderedItems;
    expect(items[0].media.id).toBe(3);
    expect(items[1].media.id).toBe(1);
    expect(items[2].media.id).toBe(2);
  });

  it('should insert threshold line at correct position', () => {
    component.sortOrder = [
      { id: 3, score: 0.9 },
      { id: 1, score: 0.5 },
      { id: 2, score: 0.2 },
    ];
    component.threshold = 0.4;
    component.ngOnChanges({
      sortOrder: new SimpleChange(null, component.sortOrder, false),
      threshold: new SimpleChange(null, component.threshold, false),
    });
    TestBed.tick();
    const items = component.cachedOrderedItems;
    // Threshold is at 0.4, so item with score 0.2 should have showThreshold
    expect(items[0].showThreshold).toBe(false);
    expect(items[1].showThreshold).toBe(false);
    expect(items[2].showThreshold).toBe(true);
  });

  it('should return correct vote labels', () => {
    component.goodVotes = new Set([1]);
    component.badVotes = new Set([2]);
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
    // Drive the input through setInput so ngOnChanges rebuilds the ordered-items
    // cache to empty (a direct field write leaves the cache stale, which under
    // zoneless surfaces as an NG0100 during the verify pass) and the host view
    // is marked dirty so the tick repaints.
    fixture.componentRef.setInput('medias', []);
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('.empty-list')?.textContent).toContain('No media loaded');
  });

  it('should render threshold line in DOM', () => {
    component.sortOrder = [
      { id: 1, score: 0.8 },
      { id: 2, score: 0.3 },
    ];
    component.threshold = 0.5;
    component.ngOnChanges({
      sortOrder: new SimpleChange(null, component.sortOrder, false),
      threshold: new SimpleChange(null, component.threshold, false),
    });
    TestBed.tick();
    expect(fixture.nativeElement.querySelector('.media-threshold-line')).toBeTruthy();
  });

  // Regression: small datasets used to never prefetch metadata, so the list
  // displayed ``#284`` placeholders forever; names only filled in when the
  // user clicked an item. Without virtual scroll active, every row is in the
  // DOM, so every row is visible and should hydrate eagerly.
  it('eagerly prefetches metadata for every row when virtual scroll is off', () => {
    const cache = TestBed.inject(MediaMetadataCacheService);
    const spy = vi.spyOn(cache, 'ensureLoaded').mockImplementation(() => {});
    component.medias = mockMedias;
    component.ngOnChanges({
      medias: { currentValue: mockMedias, previousValue: [], firstChange: false, isFirstChange: () => false },
    });
    expect(spy).toHaveBeenCalled();
    const ids = spy.mock.lastCall![0] as number[];
    expect(ids).toEqual([1, 2, 3]);
  });
});
