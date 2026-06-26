import { ComponentFixture, TestBed } from '@angular/core/testing';
import { LabelListComponent } from './label-list.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('LabelListComponent', () => {
  let component: LabelListComponent;
  let fixture: ComponentFixture<LabelListComponent>;

  const sampleMedias: Media[] = [
    { id: 1, media_type: 'audio', filename: 'song.wav', md5: 'aaa', custom_metadata: {} },
    { id: 2, media_type: 'image', filename: 'photo.jpg', md5: 'bbb', custom_metadata: {} },
    { id: 3, media_type: 'video', filename: 'clip.mp4', md5: 'ccc', custom_metadata: {} },
  ];

  // Set any number of inputs via setInput (a CD trigger) then settle so
  // ngOnChanges rebuilds the lookup map and the sortedEntries signal.
  async function setInputs(inputs: Record<string, unknown>): Promise<void> {
    for (const [k, v] of Object.entries(inputs)) {
      fixture.componentRef.setInput(k, v);
    }
    await settleZoneless(fixture);
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LabelListComponent],
      providers: [...provideZoneless(), ActiveContextService],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelListComponent);
    component = fixture.componentInstance;
  });

  it('should create', async () => {
    await settleZoneless(fixture);
    expect(component).toBeTruthy();
  });

  it('should display correct label heading and count', async () => {
    await setInputs({ label: 'good', ids: [1, 2], medias: sampleMedias });

    const el = fixture.nativeElement as HTMLElement;
    const heading = el.querySelector('h3');
    expect(heading?.textContent).toContain('Goods');
    expect(heading?.textContent).toContain('(2)');
    expect(heading?.classList.contains('good')).toBe(true);
  });

  it('should display bad label heading', async () => {
    await setInputs({ label: 'bad', ids: [3], medias: sampleMedias });

    const el = fixture.nativeElement as HTMLElement;
    const heading = el.querySelector('h3');
    expect(heading?.textContent).toContain('Bads');
    expect(heading?.classList.contains('bad')).toBe(true);
  });

  describe('sorting', () => {
    beforeEach(() => {
      // Batch the base inputs; each test adds sortMode then settles once.
      fixture.componentRef.setInput('label', 'good');
      fixture.componentRef.setInput('ids', [1, 2, 3]);
      fixture.componentRef.setInput('medias', sampleMedias);
      fixture.componentRef.setInput('clickTimes', { '1': 3, '2': 1, '3': 2 });
      fixture.componentRef.setInput('learnedScores', { '1': 0.9, '2': 0.5, '3': 0.7 });
    });

    it('should sort by time-desc (newest first)', async () => {
      await setInputs({ sortMode: 'time-desc' });
      expect(component.sortedEntries().map(e => e.id)).toEqual([1, 3, 2]);
    });

    it('should sort by time-asc (oldest first)', async () => {
      await setInputs({ sortMode: 'time-asc' });
      expect(component.sortedEntries().map(e => e.id)).toEqual([2, 3, 1]);
    });

    it('should sort by name-asc', async () => {
      await setInputs({ sortMode: 'name-asc' });
      const names = component.sortedEntries().map(e => e.name);
      expect(names).toEqual(['clip.mp4', 'photo.jpg', 'song.wav']);
    });

    it('should sort by name-desc', async () => {
      await setInputs({ sortMode: 'name-desc' });
      const names = component.sortedEntries().map(e => e.name);
      expect(names).toEqual(['song.wav', 'photo.jpg', 'clip.mp4']);
    });

    it('should sort by confidence-desc', async () => {
      await setInputs({ sortMode: 'confidence-desc' });
      expect(component.sortedEntries().map(e => e.id)).toEqual([1, 3, 2]);
    });

    it('should sort by confidence-asc', async () => {
      await setInputs({ sortMode: 'confidence-asc' });
      expect(component.sortedEntries().map(e => e.id)).toEqual([2, 3, 1]);
    });

    it('should sort by id-asc', async () => {
      await setInputs({ sortMode: 'id-asc' });
      expect(component.sortedEntries().map(e => e.id)).toEqual([1, 2, 3]);
    });
  });

  describe('confidence calculation', () => {
    it('should use score directly for good label', async () => {
      await setInputs({ label: 'good', ids: [1], medias: sampleMedias, learnedScores: { '1': 0.8 } });
      expect(component.sortedEntries()[0].confidence).toBeCloseTo(0.8);
    });

    it('should use 1-score for bad label', async () => {
      await setInputs({ label: 'bad', ids: [1], medias: sampleMedias, learnedScores: { '1': 0.3 } });
      expect(component.sortedEntries()[0].confidence).toBeCloseTo(0.7);
    });

    it('should set confidence to -1 when no score', async () => {
      await setInputs({ label: 'good', ids: [1], medias: sampleMedias, learnedScores: {} });
      expect(component.sortedEntries()[0].confidence).toBe(-1);
    });
  });

  describe('view modes and thumbnails', () => {
    beforeEach(async () => {
      // setInput builds the media-id → media lookup map (in ngOnChanges) so
      // thumbnailUrl()/hasThumbnailUrl() resolve.
      await setInputs({ medias: sampleMedias });
    });

    it('should have thumbnail URL for images', () => {
      expect(component.hasThumbnailUrl(2)).toBe(true);
    });

    it('should have thumbnail URL for videos', () => {
      expect(component.hasThumbnailUrl(3)).toBe(true);
    });

    it('should have thumbnail URL for audio', () => {
      expect(component.hasThumbnailUrl(1)).toBe(true);
    });

    it('should generate correct thumbnail URLs', () => {
      expect(component.thumbnailUrl(1)).toBe('/api/medias/1/thumbnail');
      expect(component.thumbnailUrl(2)).toBe('/api/medias/2/thumbnail');
      expect(component.thumbnailUrl(3)).toBe('/api/medias/3/thumbnail');
    });

    it('should not show placeholder icon for audio (has thumbnail)', () => {
      expect(component.placeholderIcon(1)).toBeNull();
    });

    it('should not show placeholder icon for image (has thumbnail)', () => {
      expect(component.placeholderIcon(2)).toBeNull();
    });
  });

  it('should emit mediaSelected on entry click', () => {
    vi.spyOn(component.mediaSelected, 'emit');
    component.onEntryClick(42);
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
  });

  it('should emit mediaSelected on Enter keydown', () => {
    vi.spyOn(component.mediaSelected, 'emit');
    component.onEntryKeydown(new KeyboardEvent('keydown', { key: 'Enter' }), 42);
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
  });

  it('should emit mediaSelected on Space keydown', () => {
    vi.spyOn(component.mediaSelected, 'emit');
    component.onEntryKeydown(new KeyboardEvent('keydown', { key: ' ' }), 42);
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
  });

  it('should not emit on other key', () => {
    vi.spyOn(component.mediaSelected, 'emit');
    component.onEntryKeydown(new KeyboardEvent('keydown', { key: 'a' }), 42);
    expect(component.mediaSelected.emit).not.toHaveBeenCalled();
  });

  it('should use filename for entry name', async () => {
    await setInputs({ ids: [1], medias: sampleMedias });
    expect(component.sortedEntries()[0].name).toBe('song.wav');
  });

  it('should fallback to Clip #ID when no media found', async () => {
    await setInputs({ ids: [999], medias: sampleMedias });
    expect(component.sortedEntries()[0].name).toBe('Clip #999');
  });

  it('should render vote entries in the DOM', async () => {
    await setInputs({ label: 'good', ids: [1, 2], medias: sampleMedias });

    const el = fixture.nativeElement as HTMLElement;
    const entries = el.querySelectorAll('.vote-entry');
    expect(entries.length).toBe(2);
  });
});
