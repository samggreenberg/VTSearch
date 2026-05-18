import { ComponentFixture, TestBed } from '@angular/core/testing';
import { MediaItemComponent } from './media-item.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';

describe('MediaItemComponent', () => {
  let component: MediaItemComponent;
  let fixture: ComponentFixture<MediaItemComponent>;

  const mockMedia: Media = {
    id: 1,
    type: 'audio',
    filename: 'test.wav',
    md5: 'abc123',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaItemComponent],
      providers: [ActiveContextService],
    }).compileComponents();

    fixture = TestBed.createComponent(MediaItemComponent);
    component = fixture.componentInstance;
    component.media = mockMedia;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display filename', () => {
    expect(component.displayName).toBe('test.wav');
  });

  it('should fall back to description when no filename', () => {
    component.media = { ...mockMedia, filename: '', description: 'A sound' };
    expect(component.displayName).toBe('A sound');
  });

  it('should fall back to id when no filename or description', () => {
    component.media = { ...mockMedia, filename: '', description: undefined };
    expect(component.displayName).toBe('#1');
  });

  it('should emit select on click', () => {
    spyOn(component.select, 'emit');
    fixture.nativeElement.querySelector('.media-item').click();
    expect(component.select.emit).toHaveBeenCalledWith(1);
  });

  it('should add active class when active', () => {
    component.active = true;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.media-item.active')).toBeTruthy();
  });

  it('should add labeled-good class when vote is good', () => {
    component.voteLabel = 'good';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.media-item.labeled-good')).toBeTruthy();
  });

  it('should add labeled-bad class when vote is bad', () => {
    component.voteLabel = 'bad';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.media-item.labeled-bad')).toBeTruthy();
  });

  it('should show thumbnail for audio type', () => {
    expect(component.thumbnailUrl).toBe('/api/medias/1/image');
  });

  it('should show thumbnail for image type', () => {
    component.media = { ...mockMedia, type: 'image' };
    expect(component.thumbnailUrl).toBe('/api/medias/1/image');
  });

  it('should not show thumbnail for text type', () => {
    component.media = { ...mockMedia, type: 'text' };
    expect(component.thumbnailUrl).toBeNull();
  });

  it('should fall back to placeholder when thumbnail fails to load', () => {
    component.viewMode = 'grid';
    expect(component.thumbnailUrl).toBe('/api/medias/1/image');
    component.onThumbnailError();
    expect(component.thumbnailUrl).toBeNull();
    expect(component.placeholderIcon).toBe('\u266B');
  });

  it('should reset thumbnailFailed when media changes', () => {
    component.onThumbnailError();
    expect(component.thumbnailUrl).toBeNull();
    component.media = { ...mockMedia, id: 2 };
    component.ngOnChanges({ media: {} as any });
    expect(component.thumbnailUrl).toBe('/api/medias/2/image');
  });

  it('should show score when provided', () => {
    component.score = 0.85;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.media-score')).toBeTruthy();
  });

  it('should not show score when null', () => {
    component.score = null;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.media-score')).toBeNull();
  });
});
