import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BehaviorSubject } from 'rxjs';
import { signal } from '@angular/core';

import { BrowseSelectionPanelComponent } from './browse-selection-panel.component';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { SettingsStateService } from '../../services/settings-state.service';
import type { NowPlaying } from '../browse-hover-preview/browse-hover-preview.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { makeActiveContextStub } from '../../testing/mocks';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Zoneless staleness canary for the browse selection panel.
 * Phase 2.6 signalized
 * `count`/`gridGoalWidth`/`sortedEntries`, which were template-bound
 * but written from the selection-refresh `effect()` and the metadata-cache
 * `version$` subscribe — neither of which schedules CD for a plain field under
 * zoneless (the effect-into-plain-field write was a latent staleness bug).
 *
 * Both tests run under a zoneless `TestBed`, drive state through the production
 * channel with NO manual `detectChanges()`, then assert on the rendered DOM.
 */
describe('BrowseSelectionPanelComponent (zoneless canary)', () => {
  let fixture: ComponentFixture<BrowseSelectionPanelComponent>;
  let selection: BrowseSelectionService;
  let metaVersion: BehaviorSubject<number>;
  let names: Map<number, string>;
  let playSpy: ReturnType<typeof vi.spyOn>;

  // Mirrors the component's dwell constant; the waits below clear it with margin.
  const DWELL_MS = 200;
  const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  beforeEach(async () => {
    metaVersion = new BehaviorSubject<number>(0);
    names = new Map<number, string>();

    const metadataStub: Partial<MediaMetadataCacheService> = {
      version$: metaVersion.asObservable(),
      ensureLoaded: () => {},
      get: ((id: number) =>
        names.has(id) ? { filename: names.get(id) } : undefined) as MediaMetadataCacheService['get'],
    };
    // jsdom has no real media pipeline; keep play()/load() quiet so the audio
    // hover path's audition kick-off doesn't spam "Not implemented".
    playSpy = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => {});

    const activeContextStub = makeActiveContextStub();
    const settingsStub: Partial<SettingsStateService> = {
      settingsSignal: signal(null) as SettingsStateService['settingsSignal'],
      load: () => {},
    };

    await configureZoneless({
      imports: [BrowseSelectionPanelComponent],
      providers: [
        BrowseSelectionService,
        { provide: MediaMetadataCacheService, useValue: metadataStub },
        { provide: ActiveContextService, useValue: activeContextStub },
        { provide: SettingsStateService, useValue: settingsStub },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(BrowseSelectionPanelComponent);
    selection = TestBed.inject(BrowseSelectionService);
    fixture.componentRef.setInput('mediaType', 'audio');
  });

  afterEach(() => {
    vi.restoreAllMocks();
    fixture.destroy();
  });

  it('repaints the count + list when the selection signal bumps, no manual detectChanges', async () => {
    await settleZoneless(fixture);

    // Empty selection: the empty hint shows and the title reads "Selection (0)".
    expect(fixture.nativeElement.querySelector('.bsp-empty')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('.bsp-title')!.textContent).toContain('0');
    expect(fixture.nativeElement.querySelectorAll('.bsp-entry').length).toBe(0);

    // Production channel: a selection mutation from outside any bound handler
    // bumps the service's `version` signal; the refresh effect must repaint.
    selection.addAll([1, 2, 3]);
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.bsp-empty')).toBeNull();
    expect(fixture.nativeElement.querySelector('.bsp-title')!.textContent).toContain('3');
    expect(fixture.nativeElement.querySelectorAll('.bsp-entry').length).toBe(3);
  });

  it('repaints the entry name tooltip when the metadata cache version$ emits, no manual detectChanges', async () => {
    selection.addAll([7]);
    await settleZoneless(fixture);

    // The list shows only thumbnails (no visible name text), so the name lives
    // in the entry's tooltip. Before it resolves, it falls back to the id.
    let entry = fixture.nativeElement.querySelector('.bsp-entry') as HTMLElement;
    expect(entry.getAttribute('title')).toContain('Clip #7');

    // Production channel: the metadata cache hydrates and pushes version$,
    // whose subscribe rebuilds `sortedEntries` (a signal) — must repaint.
    names.set(7, 'kestrel.wav');
    metaVersion.next(1);
    await settleZoneless(fixture);

    entry = fixture.nativeElement.querySelector('.bsp-entry') as HTMLElement;
    expect(entry.getAttribute('title')).toContain('kestrel.wav');
  });

  it('exposes the full name via the entry tooltip and no longer renders a visible name or copy button', async () => {
    selection.addAll([7]);
    names.set(7, 'a-very-long-filename-that-gets-truncated.wav');
    metaVersion.next(1);
    await settleZoneless(fixture);

    // The entry title carries the full name so hovering surfaces it even though
    // the list no longer prints names beneath the thumbnails.
    const entry = fixture.nativeElement.querySelector('.bsp-entry') as HTMLElement;
    expect(entry.getAttribute('title')).toBe(
      'a-very-long-filename-that-gets-truncated.wav — click to remove from selection',
    );

    // The visible name row and its Copy-name button are gone entirely.
    expect(entry.querySelector('.bsp-name-row')).toBeNull();
    expect(entry.querySelector('.bsp-copy')).toBeNull();
  });

  // --- Hover-to-play audio (issue #2455) -------------------------------------
  //
  // Selection entries audition their clip on hover the same way the canvas
  // hover-preview and bin-popup do: a dwell debounce arms on mouseenter, plays
  // through a private (never-mounted) audio element, and emits to the shared
  // top-left `nowPlaying` indicator; mouseleave stops it at once.

  function firstEntry(): HTMLElement {
    return fixture.nativeElement.querySelector('.bsp-entry') as HTMLElement;
  }

  it('auditions an audio entry only after the dwell elapses, emitting nowPlaying', async () => {
    let emitted: NowPlaying | null | undefined;
    fixture.componentInstance.nowPlaying.subscribe((e) => (emitted = e));
    selection.addAll([7]);
    await settleZoneless(fixture);

    firstEntry().dispatchEvent(new MouseEvent('mouseenter'));
    await settleZoneless(fixture);

    // Before the dwell: nothing auditioning yet.
    expect(playSpy).not.toHaveBeenCalled();
    expect(emitted).toBeUndefined();

    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);

    // After the dwell: playback starts and `nowPlaying` carries the clip's
    // waveform for the top-left indicator — flagged `loading` until it sounds.
    expect(playSpy).toHaveBeenCalled();
    // `progress` is null until a finite duration is known (jsdom has no media
    // pipeline, so it never is); the sweeping playhead stays hidden meanwhile.
    expect(emitted).toEqual({ mediaId: 7, waveUrl: '/api/medias/7/thumbnail', loading: true, progress: null });
  });

  it('stops the audition and emits nowPlaying(null) as soon as the cursor leaves', async () => {
    let emitted: NowPlaying | null | undefined;
    fixture.componentInstance.nowPlaying.subscribe((e) => (emitted = e));
    selection.addAll([5]);
    await settleZoneless(fixture);

    firstEntry().dispatchEvent(new MouseEvent('mouseenter'));
    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);
    expect(emitted?.mediaId).toBe(5);

    // No hover-bridge grace period: the clip stops the instant the entry is left.
    firstEntry().dispatchEvent(new MouseEvent('mouseleave'));
    await settleZoneless(fixture);
    expect(emitted).toBeNull();
  });

  it('does not audition for non-audio datasets', async () => {
    fixture.componentRef.setInput('mediaType', 'image');
    selection.addAll([3]);
    await settleZoneless(fixture);

    firstEntry().dispatchEvent(new MouseEvent('mouseenter'));
    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);

    expect(playSpy).not.toHaveBeenCalled();
  });

  it('silences a playing entry when it is clicked to remove', async () => {
    let emitted: NowPlaying | null | undefined;
    fixture.componentInstance.nowPlaying.subscribe((e) => (emitted = e));
    selection.addAll([8, 9]);
    await settleZoneless(fixture);

    // Audition the top entry, then click it to drop it from selection.
    const entry = firstEntry();
    entry.dispatchEvent(new MouseEvent('mouseenter'));
    await wait(DWELL_MS + 60);
    await settleZoneless(fixture);
    const playingId = emitted?.mediaId;
    expect(playingId).not.toBeUndefined();

    entry.dispatchEvent(new MouseEvent('click'));
    await settleZoneless(fixture);

    // Removal unmounts the entry (no mouseleave fires), so the click path must
    // silence it directly: nowPlaying clears and that item is gone.
    expect(emitted).toBeNull();
    expect(selection.ids()).not.toContain(playingId);
    expect(selection.ids().length).toBe(1);
  });
});
