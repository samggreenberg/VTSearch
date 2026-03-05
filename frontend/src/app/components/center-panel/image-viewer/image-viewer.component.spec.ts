import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ImageViewerComponent } from './image-viewer.component';
import { MediaItem } from '../../../models/api.models';

describe('ImageViewerComponent', () => {
  let component: ImageViewerComponent;
  let fixture: ComponentFixture<ImageViewerComponent>;

  const mockMedia: MediaItem = {
    id: 2,
    type: 'image',
    duration: 0,
    file_size: 2048,
    filename: 'test.png',
    category: 'test',
    md5: 'def456',
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImageViewerComponent],
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

  it('should render image element', () => {
    component.media = mockMedia;
    component.imageSrc = '/api/medias/2/image';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('img')).toBeTruthy();
  });

  it('should render image controls', () => {
    component.media = mockMedia;
    component.imageSrc = '/api/medias/2/image';
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.image-view-controls')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('input[type="range"]')).toBeTruthy();
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
