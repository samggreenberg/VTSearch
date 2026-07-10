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

  describe('precomputed entry display fields', () => {
    // Entries carry their thumbnail URL / fallback icon precomputed by
    // buildSortedEntries() (the template binds stored fields; there are no
    // per-change-detection getters anymore).
    beforeEach(async () => {
      await setInputs({ ids: [1, 2, 3], medias: sampleMedias });
    });

    function entryById(id: number) {
      const entry = component.sortedEntries().find((e) => e.id === id);
      if (!entry) throw new Error(`no entry for id ${id}`);
      return entry;
    }

    it('should precompute thumbnail URLs for audio/image/video', () => {
      expect(entryById(1).thumbnailUrl).toBe('/api/medias/1/thumbnail');
      expect(entryById(2).thumbnailUrl).toBe('/api/medias/2/thumbnail');
      expect(entryById(3).thumbnailUrl).toBe('/api/medias/3/thumbnail');
    });

    it('should fold a region box into the thumbnail URL', async () => {
      await setInputs({ regionBoxes: { '2': [0, 0.25, 0.5, 1] } });
      expect(entryById(2).thumbnailUrl).toBe('/api/medias/2/thumbnail?region=0.0000,0.2500,0.5000,1.0000');
      expect(entryById(1).thumbnailUrl).toBe('/api/medias/1/thumbnail');
    });

    it('should keep a media-type fallback icon for failed thumbnails', () => {
      expect(entryById(1).fallbackIcon).toBe('♫');
      expect(entryById(2).fallbackIcon).toBe('□');
    });

    it('should mark ids without a media as missing with no thumbnail or icon', async () => {
      await setInputs({ ids: [999] });
      const entry = component.sortedEntries()[0];
      expect(entry.thumbnailUrl).toBe('');
      expect(entry.fallbackIcon).toBeNull();
      expect(entry.missing).toBe(true);
    });
  });

  it('should emit mediaSelected with a numeric id on grid selection', () => {
    vi.spyOn(component.mediaSelected, 'emit');
    component.onEntrySelected({ key: '42', name: 'x', thumbnailUrl: '', fallbackIcon: null, missing: false });
    expect(component.mediaSelected.emit).toHaveBeenCalledWith(42);
  });

  it('should forward grid votes with a numeric id', () => {
    vi.spyOn(component.mediaVote, 'emit');
    component.onEntryVote({
      entry: { key: '42', name: 'x', thumbnailUrl: '', fallbackIcon: null, missing: false },
      vote: 'good',
    });
    expect(component.mediaVote.emit).toHaveBeenCalledWith({ id: 42, vote: 'good' });
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
