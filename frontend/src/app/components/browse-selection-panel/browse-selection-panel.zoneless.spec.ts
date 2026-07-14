import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BehaviorSubject } from 'rxjs';
import { signal } from '@angular/core';

import { BrowseSelectionPanelComponent } from './browse-selection-panel.component';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { makeActiveContextStub } from '../../testing/mocks';
import { settleZoneless } from '../../testing/settle-resource';

/**
 * Zoneless staleness canary for the browse selection panel
 * (docs/plans/zoneless-migration.md, Phases 0.3/0.4 + 2.6). Phase 2.6 signalized
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

  beforeEach(async () => {
    metaVersion = new BehaviorSubject<number>(0);
    names = new Map<number, string>();

    const metadataStub: Partial<MediaMetadataCacheService> = {
      version$: metaVersion.asObservable(),
      ensureLoaded: () => {},
      get: ((id: number) =>
        names.has(id) ? { filename: names.get(id) } : undefined) as MediaMetadataCacheService['get'],
    };
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

  afterEach(() => fixture.destroy());

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

  it('repaints item names when the metadata cache version$ emits, no manual detectChanges', async () => {
    selection.addAll([7]);
    await settleZoneless(fixture);

    // Before the name resolves, the entry falls back to the id placeholder.
    let nameEl = fixture.nativeElement.querySelector('.bsp-name-grid, .bsp-name');
    expect(nameEl!.textContent).toContain('Clip #7');

    // Production channel: the metadata cache hydrates and pushes version$,
    // whose subscribe rebuilds `sortedEntries` (a signal) — must repaint.
    names.set(7, 'kestrel.wav');
    metaVersion.next(1);
    await settleZoneless(fixture);

    nameEl = fixture.nativeElement.querySelector('.bsp-name-grid, .bsp-name');
    expect(nameEl!.textContent).toContain('kestrel.wav');
  });

  it('exposes the full name via the entry tooltip and a per-entry copy button', async () => {
    selection.addAll([7]);
    names.set(7, 'a-very-long-filename-that-gets-truncated.wav');
    metaVersion.next(1);
    await settleZoneless(fixture);

    // The entry title carries the full name (not the old hardcoded string), so
    // hovering surfaces it even when the on-screen text is truncated/hidden.
    const entry = fixture.nativeElement.querySelector('.bsp-entry') as HTMLElement;
    expect(entry.getAttribute('title')).toBe(
      'a-very-long-filename-that-gets-truncated.wav — click to remove from selection',
    );

    // A copy-detail button sits beside each entry, wired to copy the full name.
    const copyBtn = entry.querySelector('.bsp-copy .copy-detail-btn') as HTMLButtonElement;
    expect(copyBtn).not.toBeNull();
    expect(copyBtn.getAttribute('title')).toBe('Copy name to clipboard');
  });
});
