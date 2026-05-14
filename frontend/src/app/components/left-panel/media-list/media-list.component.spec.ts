import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { MediaListComponent } from './media-list.component';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { MediaItem } from '../../../models/api.models';

describe('MediaListComponent', () => {
  let component: MediaListComponent;
  let fixture: ComponentFixture<MediaListComponent>;
  let cache: MediaMetadataCacheService;

  const mockMedias: MediaItem[] = [
    { id: 1, type: 'audio', filename: 'a.wav', md5: 'a1', custom_metadata: {} },
    { id: 2, type: 'audio', filename: 'b.wav', md5: 'b2', custom_metadata: {} },
    { id: 3, type: 'audio', filename: 'c.wav', md5: 'c3', custom_metadata: {} },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaListComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(MediaListComponent);
    component = fixture.componentInstance;
    cache = TestBed.inject(MediaMetadataCacheService);
    // Preload the cache so lookups return real items instead of placeholders.
    for (const m of mockMedias) {
      (cache as unknown as { cache: Map<number, MediaItem> }).cache.set(m.id, m);
    }
    component.mediaIds = mockMedias.map((m) => m.id);
    fixture.detectChanges();
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
    fixture.detectChanges();
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
    fixture.detectChanges();
    const items = component.cachedOrderedItems;
    expect(items[0].showThreshold).toBeFalse();
    expect(items[1].showThreshold).toBeFalse();
    expect(items[2].showThreshold).toBeTrue();
  });

  it('should return correct vote labels', () => {
    component.goodVotes = new Set([1]);
    component.badVotes = new Set([2]);
    expect(component.getVoteLabel(1)).toBe('good');
    expect(component.getVoteLabel(2)).toBe('bad');
    expect(component.getVoteLabel(3)).toBeNull();
  });

  it('should emit mediaSelect on item select', () => {
    spyOn(component.mediaSelect, 'emit');
    component.onMediaSelect(2);
    expect(component.mediaSelect.emit).toHaveBeenCalledWith(2);
  });

  it('should show empty message when no medias', () => {
    component.mediaIds = [];
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.empty-list')?.textContent).toContain('No media loaded');
  });

  it('should render threshold line in DOM', () => {
    component.sortOrder = [
      { id: 1, score: 0.8 },
      { id: 2, score: 0.3 },
    ];
    component.threshold = 0.5;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.media-threshold-line')).toBeTruthy();
  });
});
