import { ChangeDetectionStrategy, Component, computed, effect, inject, input, OnChanges, OnInit, signal, SimpleChanges } from '@angular/core';

import {
  SettingsStateService,
  type SettingsKey,
} from '../../services/settings-state.service';
import { coerceFocusMode, type FocusMode } from '../../utils/settings-coerce';
import { IconComponent } from '../icon/icon.component';

const ICON_SIZES = ['XS', 'S', 'M', 'L', 'XL'] as const;
type IconSize = (typeof ICON_SIZES)[number];

/** Accept only a size on this control's ladder; anything else falls back. */
function coerceIconSize(raw: unknown): IconSize | undefined {
  return (ICON_SIZES as readonly unknown[]).includes(raw) ? (raw as IconSize) : undefined;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-view-controls',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './view-controls.component.html',
  styleUrl: './view-controls.component.scss',
})
export class ViewControlsComponent implements OnInit, OnChanges {
  private settingsState = inject(SettingsStateService);

  readonly side = input<'left' | 'right' | 'popup'>('left');
  readonly currentMediaType = input('');
  /**
   * Whether to show the click/hover focus toggle. The VTSBrowse selection
   * panel reuses this control for its size buttons but has no good/bad voting,
   * so it hides the focus group with ``[showFocus]="false"``.
   */
  readonly showFocus = input(true);

  /**
   * The two preferences this control edits, keyed on the active media type and
   * on which panel it is mounted in (issue #3447). The key is a signal because
   * `side` is an input: one component instance reads `focus_mode_left` or
   * `focus_mode_popup` depending on where it was placed.
   */
  private readonly focusPref = this.settingsState.perMediaType<FocusMode>(
    computed<SettingsKey>(() => `focus_mode_${this.side()}` as SettingsKey),
    this.currentMediaType,
    { fallback: 'click', coerce: coerceFocusMode },
  );
  private readonly sizePref = this.settingsState.perMediaType<IconSize>(
    computed<SettingsKey>(() => `grid_icon_size_${this.side()}` as SettingsKey),
    this.currentMediaType,
    { fallback: 'M', coerce: coerceIconSize },
  );

  // Local, and written optimistically on click rather than bound straight to
  // the preferences above: the buttons compute the next size from the current
  // one and grey themselves out at the ends of the ladder, so waiting for the
  // PUT to round-trip would make a rapid second click read a stale size and
  // leave the control looking stuck. Signals (not plain fields) so the OnPush
  // template repaints on a write from `refresh()`, which runs outside any bound
  // handler.
  readonly focusMode = signal<FocusMode>('click');
  readonly gridIconSize = signal<IconSize>('M');

  constructor() {
    // `refresh()` reads both preferences, so this effect depends on them and
    // re-runs whenever either resolves to something new: settings landing, a
    // change made in the Settings modal or by the sibling control on the other
    // panel, or a `side` / media-type switch.
    effect(() => this.refresh());
  }

  ngOnInit(): void {
    this.settingsState.load();
  }

  /** The constructor effect already re-resolves on a `side` / media-type
   *  change; this keeps the update synchronous with the input write, as it was
   *  before the preferences became signals. */
  ngOnChanges(changes: SimpleChanges): void {
    if (changes['currentMediaType'] || changes['side']) {
      this.refresh();
    }
  }

  private refresh(): void {
    this.focusMode.set(this.focusPref.value());
    this.gridIconSize.set(this.sizePref.value());
  }

  setFocusMode(mode: FocusMode): void {
    if (!this.currentMediaType() || this.focusMode() === mode) return;
    this.focusMode.set(mode);
    this.focusPref.set(mode)?.subscribe();
  }

  bumpSize(delta: 1 | -1): void {
    if (!this.currentMediaType()) return;
    const idx = ICON_SIZES.indexOf(this.gridIconSize());
    const next = ICON_SIZES[Math.max(0, Math.min(ICON_SIZES.length - 1, idx + delta))];
    if (next === this.gridIconSize()) return;
    this.gridIconSize.set(next);
    this.sizePref.set(next)?.subscribe();
  }

  readonly atMaxSize = computed(() => this.gridIconSize() === ICON_SIZES[ICON_SIZES.length - 1]);

  readonly atMinSize = computed(() => this.gridIconSize() === ICON_SIZES[0]);
}
