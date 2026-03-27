import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ImageViewerComponent } from './image-viewer.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { MediaItem } from '../../../models/api.models';

describe('ImageViewerComponent', () => {
  let component: ImageViewerComponent;
  let fixture: ComponentFixture<ImageViewerComponent>;

  const mockMedia: MediaItem = {
    id: 2,
    type: 'image',
    filename: 'test.png',
    md5: 'def456',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImageViewerComponent],
      providers: [ActiveContextService],
    }).compileComponents();
    fixture = TestBed.createComponent(ImageViewerComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set imageSrc when media changes', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    expect(component.imageSrc).toBe('/api/medias/2/image');
  });

  it('should hide image until loaded to prevent flash of old image', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    expect(component.imageReady).toBeFalse();

    component.onImageLoad();
    expect(component.imageReady).toBeTrue();
  });

  it('should reset imageReady when media changes', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    component.onImageLoad();
    expect(component.imageReady).toBeTrue();

    const nextMedia = { ...mockMedia, id: 3, filename: 'next.png' };
    component.media = nextMedia;
    component.ngOnChanges({
      media: { currentValue: nextMedia, previousValue: mockMedia, firstChange: false, isFirstChange: () => false },
    });
    expect(component.imageReady).toBeFalse();
  });

  it('should show image on error to avoid stuck black screen', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    expect(component.imageReady).toBeFalse();

    component.onImageError();
    expect(component.imageReady).toBeTrue();
  });

  it('should render image element', () => {
    component.media = mockMedia;
    component.imageSrc = '/api/medias/2/image';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('img')).toBeTruthy();
  });

  it('should expose control methods and properties', () => {
    expect(component.rotateLeft).toBeDefined();
    expect(component.rotateRight).toBeDefined();
    expect(component.resetView).toBeDefined();
    expect(component.onZoomInput).toBeDefined();
    expect(component.minZoom).toBe(1);
    expect(component.maxZoom).toBe(5);
  });

  it('should reset view', () => {
    component.zoom = 2;
    component.rotation = 90;
    component.resetView();
    expect(component.zoom).toBe(1);
    expect(component.rotation).toBe(0);
  });

  it('should rotate left', () => {
    component.rotateLeft();
    expect(component.rotation).toBe(-90);
  });

  it('should rotate right', () => {
    component.rotateRight();
    expect(component.rotation).toBe(90);
  });

  it('should generate transform string', () => {
    expect(component.imageTransform).toContain('scale(1)');
    expect(component.imageTransform).toContain('rotate(0deg)');
  });
});
