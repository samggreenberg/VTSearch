import { ComponentFixture, TestBed } from '@angular/core/testing';
import { MediaListComponent } from './media-list.component';
import { MediaItem } from '../../../models/api.models';

describe('MediaListComponent', () => {
  let component: MediaListComponent;
  let fixture: ComponentFixture<MediaListComponent>;

  const mockMedias: MediaItem[] = [
    { id: 1, type: 'audio', duration: 5.0, file_size: 1024, filename: 'a.wav', category: '', md5: 'a1' },
    { id: 2, type: 'audio', duration: 3.0, file_size: 512, filename: 'b.wav', category: '', md5: 'b2' },
    { id: 3, type: 'audio', duration: 4.0, file_size: 768, filename: 'c.wav', category: '', md5: 'c3' },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaListComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(MediaListComponent);
    component = fixture.componentInstance;
    component.medias = mockMedias;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should render media items in natural order when no sortOrder', () => {
    const items = component.orderedItems;
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
    const items = component.orderedItems;
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
    const items = component.orderedItems;
    // Threshold is at 0.4, so item with score 0.2 should have showThreshold
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
    component.medias = [];
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
