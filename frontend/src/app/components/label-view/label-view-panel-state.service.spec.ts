import { Component, inject } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { LabelViewPanelStateService } from './label-view-panel-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * Zoneless staleness canary for the label view's per-media-type panel prefs.
 *
 * `label-view.component.html` binds `[gridGoalWidth]="gridGoalWidthLeft"` and
 * `[focusMode]="focusModeLeft"` through pass-through getters on the component
 * to the getters on this service. Those service getters used to return **plain
 * fields**, hydrated by a `loadFromSettings(settings)` call made from an HTTP
 * continuation — no signal anywhere in the chain, which `docs/FRONTEND.md`
 * section 5 names as the characteristic silent frontend bug here. It rendered
 * only because a co-located `effect()` in `LabelViewComponent` happened to
 * dirty the same view.
 *
 * This host binds the service getters directly, with no such effect nearby, and
 * drives the two state changes through their production channels — a settings
 * response landing, and a `setMediaType` switch. It asserts on rendered DOM
 * after `settleZoneless()` with NO manual `detectChanges()`, so a regression to
 * plain fields leaves the DOM at its defaults and fails here.
 */
@Component({
  selector: 'vt-panel-state-canary',
  standalone: true,
  template: `
    <span class="goal">{{ panelState.gridGoalWidthLeft }}</span>
    <span class="focus-left">{{ panelState.focusModeLeft }}</span>
    <span class="focus-right">{{ panelState.focusModeRight }}</span>
  `,
  providers: [LabelViewPanelStateService],
})
class PanelStateCanaryComponent {
  readonly panelState = inject(LabelViewPanelStateService);
}

describe('LabelViewPanelStateService (zoneless per-media-type canary)', () => {
  let fixture: ComponentFixture<PanelStateCanaryComponent>;
  let httpMock: HttpTestingController;
  let settingsState: SettingsStateService;
  let panelState: LabelViewPanelStateService;

  const settings = {
    grid_icon_size_left: { audio: 'L', image: 'XS' },
    focus_mode_left: { audio: 'hover', image: 'click' },
    focus_mode_right: { audio: 'click', image: 'hover' },
    panel_pct_left: { audio: 300, image: 420 },
    panel_pct_right: { audio: 250 },
  };

  beforeEach(async () => {
    await configureZoneless({
      imports: [PanelStateCanaryComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(PanelStateCanaryComponent);
    httpMock = TestBed.inject(HttpTestingController);
    settingsState = TestBed.inject(SettingsStateService);
    panelState = fixture.componentInstance.panelState;
  });

  afterEach(() => {
    httpMock.match(() => true).forEach((req) => {
      if (!req.cancelled) req.flush({});
    });
    fixture.destroy();
  });

  function text(sel: string): string | null {
    return fixture.nativeElement.querySelector(sel)?.textContent?.trim() ?? null;
  }

  /** Land a settings response through the production channel. */
  async function loadSettings(): Promise<void> {
    settingsState.load();
    TestBed.tick();
    httpMock.expectOne('/api/settings').flush(settings);
    await settleResource();
  }

  it('repaints the bound getters when settings land, no manual detectChanges', async () => {
    panelState.setMediaType('audio');
    await settleZoneless(fixture);
    // Nothing loaded yet — every pref sits at its fallback.
    expect(text('.goal')).toBe('80'); // iconSizeToGoalWidth('M')
    expect(text('.focus-left')).toBe('click');

    await loadSettings();
    await settleZoneless(fixture);

    expect(text('.goal')).toBe('130'); // iconSizeToGoalWidth('L')
    expect(text('.focus-left')).toBe('hover');
    expect(text('.focus-right')).toBe('click');
  });

  it('repaints on a media-type switch, no manual detectChanges', async () => {
    panelState.setMediaType('audio');
    await loadSettings();
    await settleZoneless(fixture);
    expect(text('.goal')).toBe('130');

    panelState.setMediaType('image');
    await settleZoneless(fixture);

    expect(text('.goal')).toBe('25'); // iconSizeToGoalWidth('XS')
    expect(text('.focus-left')).toBe('click');
    expect(text('.focus-right')).toBe('hover');
  });

  it('falls back for a media type with no saved entry', async () => {
    panelState.setMediaType('video');
    await loadSettings();
    await settleZoneless(fixture);

    expect(text('.goal')).toBe('80');
    expect(text('.focus-left')).toBe('click');
    expect(text('.focus-right')).toBe('click');
  });

  it('rejects a focus mode that is not one of the two real values', async () => {
    panelState.setMediaType('audio');
    settingsState.load();
    TestBed.tick();
    httpMock
      .expectOne('/api/settings')
      .flush({ ...settings, focus_mode_left: { audio: 'nonsense' } });
    await settleResource();
    await settleZoneless(fixture);

    expect(text('.focus-left')).toBe('click');
  });

  describe('saved panel widths', () => {
    it('reads the saved width for the active media type', async () => {
      panelState.setMediaType('audio');
      await loadSettings();

      expect(panelState.getPanelPx('left')).toBe(300);
      expect(panelState.getPanelPx('right')).toBe(250);
    });

    it('reports null when the side has no saved width', async () => {
      panelState.setMediaType('image');
      await loadSettings();

      expect(panelState.getPanelPx('left')).toBe(420);
      expect(panelState.getPanelPx('right')).toBeNull();
    });

    it('savePanelPx merges, preserving the other media type', async () => {
      panelState.setMediaType('audio');
      await loadSettings();

      panelState.savePanelPx('left', 512);
      const req = httpMock.expectOne('/api/settings');
      expect(req.request.method).toBe('PUT');
      // `image: 420` must survive: otherwise saving a width under audio
      // silently discards the width the user set under image.
      expect(req.request.body).toEqual({ panel_pct_left: { audio: 512, image: 420 } });
      req.flush({ ...settings, panel_pct_left: { audio: 512, image: 420 } });
      TestBed.tick();
      expect(panelState.getPanelPx('left')).toBe(512);
    });

    it('savePanelPx is a no-op before a media type is known', async () => {
      await loadSettings();
      panelState.savePanelPx('left', 512);
      httpMock.expectNone('/api/settings');
    });
  });
});
