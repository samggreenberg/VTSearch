import { ComponentFixture, TestBed } from '@angular/core/testing';
import { MediaItemComponent } from './media-item.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('MediaItemComponent', () => {
  let component: MediaItemComponent;
  let fixture: ComponentFixture<MediaItemComponent>;

  const mockMedia: Media = {
    id: 1,
    media_type: 'audio',
    filename: 'test.wav',
    md5: 'abc123',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MediaItemComponent],
      providers: [...provideZoneless(), ActiveContextService],
    }).compileComponents();

    fixture = TestBed.createComponent(MediaItemComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);
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
    vi.spyOn(component.select, 'emit');
    fixture.nativeElement.querySelector('.media-item').click();
    expect(component.select.emit).toHaveBeenCalledWith(1);
  });

  it('should add active class when active', async () => {
    fixture.componentRef.setInput('active', true);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.media-item.active')).toBeTruthy();
  });

  it('should add labeled-good class when vote is good', async () => {
    fixture.componentRef.setInput('voteLabel', 'good');
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.media-item.labeled-good')).toBeTruthy();
  });

  it('should add labeled-bad class when vote is bad', async () => {
    fixture.componentRef.setInput('voteLabel', 'bad');
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.media-item.labeled-bad')).toBeTruthy();
  });

  it('should show thumbnail for audio type', () => {
    expect(component.thumbnailUrl).toBe('/api/medias/1/thumbnail');
  });

  it('should show thumbnail for image type', () => {
    component.media = { ...mockMedia, media_type: 'image' };
    expect(component.thumbnailUrl).toBe('/api/medias/1/thumbnail');
  });

  it('tints the audio waveform via a CSS mask, not a plain <img> (issue #2369)', async () => {
    // Default mockMedia is audio.
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    expect(component.isAudioThumbnail).toBe(true);
    expect(el.querySelector('img.media-thumbnail')).toBeNull();
    const wave = el.querySelector('.media-thumbnail-wave') as HTMLElement;
    expect(wave).toBeTruthy();
    expect(wave.style.maskImage || wave.style.webkitMaskImage).toContain('/api/medias/1/thumbnail');
  });

  it('renders a plain <img> for image thumbnails', async () => {
    fixture.componentRef.setInput('media', { ...mockMedia, media_type: 'image' });
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    expect(component.isAudioThumbnail).toBe(false);
    expect(el.querySelector('img.media-thumbnail')).toBeTruthy();
    expect(el.querySelector('.media-thumbnail-wave')).toBeNull();
  });

  it('should not show thumbnail for text type', () => {
    component.media = { ...mockMedia, media_type: 'text' };
    expect(component.thumbnailUrl).toBeNull();
  });

  it('should fall back to placeholder when thumbnail fails to load', () => {
    expect(component.thumbnailUrl).toBe('/api/medias/1/thumbnail');
    component.onThumbnailError();
    expect(component.thumbnailUrl).toBeNull();
    expect(component.placeholderIcon).toBe('\u266B');
  });

  it('should reset thumbnailFailed when media changes', () => {
    component.onThumbnailError();
    expect(component.thumbnailUrl).toBeNull();
    component.media = { ...mockMedia, id: 2 };
    component.ngOnChanges({ media: {} as any });
    expect(component.thumbnailUrl).toBe('/api/medias/2/thumbnail');
  });

  it('should show score when provided', async () => {
    fixture.componentRef.setInput('score', 0.85);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.media-score')).toBeTruthy();
  });

  it('should not show score when null', async () => {
    fixture.componentRef.setInput('score', null);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.media-score')).toBeNull();
  });
});
