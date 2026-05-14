import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { LabelListComponent } from './label-list.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { MediaItem } from '../../../models/api.models';

describe('LabelListComponent', () => {
  let component: LabelListComponent;
  let fixture: ComponentFixture<LabelListComponent>;
  let cache: MediaMetadataCacheService;

  const sampleMedias: MediaItem[] = [
    { id: 1, type: 'audio', filename: 'song.wav', md5: 'aaa', custom_metadata: {} },
    { id: 2, type: 'image', filename: 'photo.jpg', md5: 'bbb', custom_metadata: {} },
    { id: 3, type: 'video', filename: 'clip.mp4', md5: 'ccc', custom_metadata: {} },
  ];

  function seedCache(): void {
    for (const m of sampleMedias) {
      (cache as unknown as { cache: Map<number, MediaItem> }).cache.set(m.id, m);
    }
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LabelListComponent],
      providers: [
        ActiveContextService,
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelListComponent);
    component = fixture.componentInstance;
    cache = TestBed.inject(MediaMetadataCacheService);
  });

  it('should create', () => {
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should display correct label heading and count', () => {
    seedCache();
    component.label = 'good';
    component.ids = [1, 2];
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const heading = el.querySelector('h3');
    expect(heading?.textContent).toContain('Goods');
    expect(heading?.textContent).toContain('(2)');
    expect(heading?.classList.contains('good')).toBeTrue();
  });

  it('should display bad label heading', () => {
    seedCache();
    component.label = 'bad';
    component.ids = [3];
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const heading = el.querySelector('h3');
    expect(heading?.textContent).toContain('Bads');
    expect(heading?.classList.contains('bad')).toBeTrue();
  });

  describe('sorting', () => {
    beforeEach(() => {
      seedCache();
      component.label = 'good';
      component.ids = [1, 2, 3];
      component.clickTimes = { '1': 3, '2': 1, '3': 2 };
      component.learnedScores = { '1': 0.9, '2': 0.5, '3': 0.7 };
    });

    it('should sort by time-desc (newest first)', () => {
      component.sortMode = 'time-desc';
      fixture.detectChanges();
      expect(component.sortedEntries.map(e => e.id)).toEqual([1, 3, 2]);
    });

    it('should sort by time-asc (oldest first)', () => {
      component.sortMode = 'time-asc';
      fixture.detectChanges();
      expect(component.sortedEntries.map(e => e.id)).toEqual([2, 3, 1]);
    });

    it('should sort by name-asc', () => {
      component.sortMode = 'name-asc';
      fixture.detectChanges();
      const names = component.sortedEntries.map(e => e.name);
      expect(names).toEqual(['clip.mp4', 'photo.jpg', 'song.wav']);
    });

    it('should sort by name-desc', () => {
      component.sortMode = 'name-desc';
      fixture.detectChanges();
      const names = component.sortedEntries.map(e => e.name);
      expect(names).toEqual(['song.wav', 'photo.jpg', 'clip.mp4']);
    });

    it('should sort by confidence-desc', () => {
      component.sortMode = 'confidence-desc';
      fixture.detectChanges();
      expect(component.sortedEntries.map(e => e.id)).toEqual([1, 3, 2]);
    });

    it('should sort by confidence-asc', () => {
      component.sortMode = 'confidence-asc';
      fixture.detectChanges();
      expect(component.sortedEntries.map(e => e.id)).toEqual([2, 3, 1]);
    });

    it('should sort by id-asc', () => {
      component.sortMode = 'id-asc';
      fixture.detectChanges();
      expect(component.sortedEntries.map(e => e.id)).toEqual([1, 2, 3]);
    });
  });

  describe('confidence calculation', () => {
    it('should use score directly for good label', () => {
      seedCache();
      component.label = 'good';
      component.ids = [1];
      component.learnedScores = { '1': 0.8 };
      fixture.detectChanges();

      expect(component.sortedEntries[0].confidence).toBeCloseTo(0.8);
    });

    it('should use 1-score for bad label', () => {
      seedCache();
      component.label = 'bad';
      component.ids = [1];
      component.learnedScores = { '1': 0.3 };
      fixture.detectChanges();

      expect(component.sortedEntries[0].confidence).toBeCloseTo(0.7);
    });

    it('should set confidence to -1 when no score', () => {
      seedCache();
      component.label = 'good';
      component.ids = [1];
      component.learnedScores = {};
      fixture.detectChanges();

      expect(component.sortedEntries[0].confidence).toBe(-1);
    });
  });

  describe('view modes and thumbnails', () => {
    beforeEach(() => {
      seedCache();
    });

    it('should have thumbnail URL for images', () => {
      expect(component.hasThumbnailUrl(2)).toBeTrue();
    });

    it('should have thumbnail URL for videos', () => {
      expect(component.hasThumbnailUrl(3)).toBeTrue();
    });

    it('should have thumbnail URL for audio', () => {
      expect(component.hasThumbnailUrl(1)).toBeTrue();
    });

    it('should generate correct thumbnail URLs', () => {
      expect(component.thumbnailUrl(1)).toBe('/api/medias/1/image');
      expect(component.thumbnailUrl(2)).toBe('/api/medias/2/image');
      expect(component.thumbnailUrl(3)).toBe('/api/medias/3/image');
    });

    it('should be in grid mode when viewMode is grid', () => {
      component.viewMode = 'grid';
      expect(component.isGrid).toBeTrue();
    });

    it('should not be in grid mode when viewMode is list', () => {
      component.viewMode = 'list';
      expect(component.isGrid).toBeFalse();
    });

    it('should not show placeholder icon for audio in grid mode (has thumbnail)', () => {
      component.viewMode = 'grid';
      expect(component.placeholderIcon(1)).toBeNull();
    });

    it('should not show placeholder icon for image in grid mode', () => {
      component.viewMode = 'grid';
      expect(component.placeholderIcon(2)).toBeNull();
    });

    it('should not show placeholder icon in list mode', () => {
      component.viewMode = 'list';
      expect(component.placeholderIcon(1)).toBeNull();
    });
  });

  it('should emit mediaSelected on entry click', () => {
    spyOn(component.mediaSelected, 'emit');
    component.onEntryClick(42);
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
  });

  it('should emit mediaSelected on Enter keydown', () => {
    spyOn(component.mediaSelected, 'emit');
    component.onEntryKeydown(new KeyboardEvent('keydown', { key: 'Enter' }), 42);
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
  });

  it('should emit mediaSelected on Space keydown', () => {
    spyOn(component.mediaSelected, 'emit');
    component.onEntryKeydown(new KeyboardEvent('keydown', { key: ' ' }), 42);
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
  });

  it('should not emit on other key', () => {
    spyOn(component.mediaSelected, 'emit');
    component.onEntryKeydown(new KeyboardEvent('keydown', { key: 'a' }), 42);
    expect(component.mediaSelected.emit).not.toHaveBeenCalled();
  });

  it('should use filename for entry name', () => {
    seedCache();
    component.ids = [1];
    fixture.detectChanges();
    expect(component.sortedEntries[0].name).toBe('song.wav');
  });

  it('should fallback to Clip #ID when no media found', () => {
    seedCache();
    component.ids = [999];
    fixture.detectChanges();
    expect(component.sortedEntries[0].name).toBe('Clip #999');
  });

  it('should render vote entries in the DOM', () => {
    seedCache();
    component.label = 'good';
    component.ids = [1, 2];
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const entries = el.querySelectorAll('.vote-entry');
    expect(entries.length).toBe(2);
  });
});
