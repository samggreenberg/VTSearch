import { ComponentFixture, TestBed } from '@angular/core/testing';
import { VoteGridComponent, VoteGridEntry } from './vote-grid.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

function makeEntries(n: number, overrides: Partial<VoteGridEntry> = {}): VoteGridEntry[] {
  return Array.from({ length: n }, (_, i) => ({
    key: String(i + 1),
    name: `item-${i + 1}`,
    thumbnailUrl: '',
    fallbackIcon: '□',
    missing: false,
    ...overrides,
  }));
}

describe('VoteGridComponent', () => {
  let component: VoteGridComponent;
  let fixture: ComponentFixture<VoteGridComponent>;

  async function setInputs(inputs: Record<string, unknown>): Promise<void> {
    for (const [k, v] of Object.entries(inputs)) {
      fixture.componentRef.setInput(k, v);
    }
    await settleZoneless(fixture);
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VoteGridComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(VoteGridComponent);
    component = fixture.componentInstance;
  });

  it('should create', async () => {
    await settleZoneless(fixture);
    expect(component).toBeTruthy();
  });

  it('renders small piles as a plain grid (no CDK viewport)', async () => {
    await setInputs({ entries: makeEntries(5) });
    const el = fixture.nativeElement as HTMLElement;
    expect(component.useVirtual).toBe(false);
    expect(el.querySelector('cdk-virtual-scroll-viewport')).toBeNull();
    expect(el.querySelector('.vote-list-plain')).toBeTruthy();
    expect(el.querySelectorAll('.vote-entry').length).toBe(5);
  });

  it('switches to a CDK virtual-scroll viewport above the threshold', async () => {
    await setInputs({ entries: makeEntries(100) });
    const el = fixture.nativeElement as HTMLElement;
    expect(component.useVirtual).toBe(true);
    expect(el.querySelector('cdk-virtual-scroll-viewport')).toBeTruthy();
    expect(el.querySelector('.vote-list-plain')).toBeNull();
    // jsdom reports zero viewport width, so the column count stays 1 and each
    // entry becomes its own virtualized row.
    expect(component.rows.length).toBe(100);
    expect(component.rows[0].length).toBe(1);
  });

  it('drops back to the plain grid when the pile shrinks', async () => {
    await setInputs({ entries: makeEntries(100) });
    await setInputs({ entries: makeEntries(3) });
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('cdk-virtual-scroll-viewport')).toBeNull();
    expect(el.querySelectorAll('.vote-entry').length).toBe(3);
  });

  it('chunks entries into rows of the current column count', async () => {
    await setInputs({ entries: makeEntries(100) });
    component.columns = 3;
    component['rebuildRows']();
    expect(component.rows.length).toBe(Math.ceil(100 / 3));
    expect(component.rows[0].map((e) => e.key)).toEqual(['1', '2', '3']);
    expect(component.rows.at(-1)!.length).toBe(100 % 3 || 3);
  });

  describe('event routing', () => {
    const entry: VoteGridEntry = { key: '7', name: 'x', thumbnailUrl: '', fallbackIcon: null, missing: false };

    it('click selects in click mode', async () => {
      await setInputs({ focusMode: 'click' });
      vi.spyOn(component.entrySelected, 'emit');
      component.onEntryClick(entry);
      expect(component.entrySelected.emit).toHaveBeenCalledWith(entry);
    });

    it('click votes bad in hover mode', async () => {
      await setInputs({ focusMode: 'hover' });
      vi.spyOn(component.entryVote, 'emit');
      component.onEntryClick(entry);
      expect(component.entryVote.emit).toHaveBeenCalledWith({ entry, vote: 'bad' });
    });

    it('contextmenu votes good in hover mode', async () => {
      await setInputs({ focusMode: 'hover' });
      vi.spyOn(component.entryVote, 'emit');
      const event = new MouseEvent('contextmenu', { cancelable: true });
      component.onEntryContextMenu(event, entry);
      expect(event.defaultPrevented).toBe(true);
      expect(component.entryVote.emit).toHaveBeenCalledWith({ entry, vote: 'good' });
    });

    it('mouseenter selects in hover mode only', async () => {
      vi.spyOn(component.entrySelected, 'emit');
      await setInputs({ focusMode: 'click' });
      component.onEntryMouseEnter(entry);
      expect(component.entrySelected.emit).not.toHaveBeenCalled();
      await setInputs({ focusMode: 'hover' });
      component.onEntryMouseEnter(entry);
      expect(component.entrySelected.emit).toHaveBeenCalledWith(entry);
    });

    it('Enter and Space select; other keys do not', () => {
      vi.spyOn(component.entrySelected, 'emit');
      component.onEntryKeydown(new KeyboardEvent('keydown', { key: 'a' }), entry);
      expect(component.entrySelected.emit).not.toHaveBeenCalled();
      component.onEntryKeydown(new KeyboardEvent('keydown', { key: 'Enter' }), entry);
      component.onEntryKeydown(new KeyboardEvent('keydown', { key: ' ' }), entry);
      expect(component.entrySelected.emit).toHaveBeenCalledTimes(2);
    });
  });

  it('falls back to the type icon when a thumbnail URL fails', async () => {
    await setInputs({ entries: makeEntries(1, { thumbnailUrl: '/api/medias/1/thumbnail', fallbackIcon: '♫' }) });
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.vote-thumbnail')).toBeTruthy();
    expect(el.querySelector('.vote-placeholder')).toBeNull();

    component.onThumbnailError('/api/medias/1/thumbnail');
    await settleZoneless(fixture);
    expect(el.querySelector('.vote-thumbnail')).toBeNull();
    expect(el.querySelector('.vote-placeholder')?.textContent).toContain('♫');
  });

  it('renders a plain <img> for non-audio thumbnails', async () => {
    await setInputs({ entries: makeEntries(1, { thumbnailUrl: '/api/medias/1/thumbnail' }) });
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('img.vote-thumbnail')).toBeTruthy();
    expect(el.querySelector('.vote-thumbnail-wave')).toBeNull();
  });

  it('tints audio waveforms via a CSS mask instead of a plain <img> (issue #2369)', async () => {
    await setInputs({
      entries: makeEntries(1, { thumbnailUrl: '/api/medias/1/thumbnail', isAudio: true }),
    });
    const el = fixture.nativeElement as HTMLElement;
    // No raw <img> — a masked element carries the wave shape.
    expect(el.querySelector('img.vote-thumbnail')).toBeNull();
    const wave = el.querySelector('.vote-thumbnail-wave') as HTMLElement;
    expect(wave).toBeTruthy();
    expect(wave.style.maskImage || wave.style.webkitMaskImage).toContain('/api/medias/1/thumbnail');
  });

  it('marks missing entries and labels them in the aria text', async () => {
    await setInputs({ label: 'good', entries: makeEntries(1, { missing: true }) });
    const el = fixture.nativeElement as HTMLElement;
    const cell = el.querySelector('.vote-entry');
    expect(cell?.classList.contains('vote-entry-missing')).toBe(true);
    expect(cell?.getAttribute('aria-label')).toBe('Good: item-1 (not in current dataset)');
  });
});
