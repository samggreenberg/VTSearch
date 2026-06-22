import { ComponentFixture, TestBed } from '@angular/core/testing';
import { SecurityContext } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { DocumentViewerComponent } from './document-viewer.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';

describe('DocumentViewerComponent', () => {
  let component: DocumentViewerComponent;
  let fixture: ComponentFixture<DocumentViewerComponent>;

  const mockMedia: Media = {
    id: 5,
    media_type: 'document',
    filename: 'test.pdf',
    md5: 'mno345',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DocumentViewerComponent],
      providers: [...provideZoneless(), ActiveContextService],
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
    // mediaSrc is a trusted SafeResourceUrl (required by the `<object [data]>`
    // RESOURCE_URL sink); unwrap it to assert the underlying media URL.
    const sanitizer = TestBed.inject(DomSanitizer);
    expect(sanitizer.sanitize(SecurityContext.RESOURCE_URL, component.mediaSrc)).toBe(
      '/api/medias/5/media',
    );
  });
});
