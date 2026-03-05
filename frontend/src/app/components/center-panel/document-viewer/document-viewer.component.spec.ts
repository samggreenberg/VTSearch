import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DocumentViewerComponent } from './document-viewer.component';
import { MediaItem } from '../../../models/api.models';

describe('DocumentViewerComponent', () => {
  let component: DocumentViewerComponent;
  let fixture: ComponentFixture<DocumentViewerComponent>;

  const mockMedia: MediaItem = {
    id: 5,
    type: 'document',
    duration: 0,
    file_size: 8192,
    filename: 'test.pdf',
    category: 'test',
    md5: 'mno345',
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DocumentViewerComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(DocumentViewerComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set mediaSrc when media changes', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    expect(component.mediaSrc).toBe('/api/medias/5/media');
  });
});
