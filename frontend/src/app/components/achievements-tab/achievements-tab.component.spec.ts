import { signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BehaviorSubject, Observable, of } from 'rxjs';

import { AchievementsTabComponent } from './achievements-tab.component';
import { AchievementsService } from '../../services/achievements.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';
import type { AchievementState } from '../../generated/api-client/models/achievement-state';
import type { AchievementEntry } from '../../generated/api-client/models/achievement-entry';
import type { CheckPhraseResponse } from '../../generated/api-client/models/check-phrase-response';
import type { AppSettings } from '../../generated/api-client/models/app-settings';

/**
 * The achievements tab renders the service snapshot: a score header, one row
 * per achievement (badge, tier label, progress bar), expandable detail panels
 * for the docs/media-type/hours rows, and the Readme-Reader phrase form. These
 * tests drive it through a fake AchievementsService and assert on the rendered
 * DOM. Run under the zoneless TestBed (the app is zoneless), so a state update
 * that forgets to nudge change detection would surface as stale markup.
 */
describe('AchievementsTabComponent', () => {
  let fixture: ComponentFixture<AchievementsTabComponent>;
  let state$: BehaviorSubject<AchievementState>;
  let checkPhrase: ReturnType<typeof vi.fn>;
  let refresh: ReturnType<typeof vi.fn>;
  let enableSetting: ReturnType<typeof signal<AppSettings | null>>;

  const EMPTY_STATE: AchievementState = {
    tier_names: [],
    achievements: [],
    pending_announcements: [],
    pending_toasts: [],
    docs: [],
    media_types: [],
    hours: [],
  };

  function makeEntry(over: Partial<AchievementEntry> = {}): AchievementEntry {
    return {
      id: 'votes_cast',
      name: 'Prolific Voter',
      description: 'Cast votes to train detectors.',
      icon: 'thumb',
      counter: 5,
      tier_idx: 0,
      tiers: [3, 25, 100, 500],
      next_threshold: 25,
      ...over,
    };
  }

  function el(selector: string): HTMLElement | null {
    return fixture.nativeElement.querySelector(selector);
  }

  function els(selector: string): HTMLElement[] {
    return Array.from(fixture.nativeElement.querySelectorAll(selector));
  }

  /** Push a new snapshot through the service stream and let CD settle. */
  async function pushState(state: Partial<AchievementState>): Promise<void> {
    state$.next({ ...EMPTY_STATE, ...state });
    await settleZoneless(fixture);
  }

  /** Create + first render. Kept out of beforeEach so `enableSetting` can be
   * seeded per-test before the constructor effect runs. */
  async function build(): Promise<void> {
    fixture = TestBed.createComponent(AchievementsTabComponent);
    await settleZoneless(fixture);
  }

  /** Open a row's detail panel by clicking its Expand toggle. */
  async function expandRow(): Promise<void> {
    el('.expand-toggle')!.click();
    await settleZoneless(fixture);
  }

  /** Type into the phrase input (updating the ngModel) and submit the form. */
  async function submitPhrase(value: string): Promise<void> {
    const input = el('.docs-phrase-input') as HTMLInputElement;
    input.value = value;
    input.dispatchEvent(new Event('input'));
    await settleZoneless(fixture);
    el('.docs-phrase-form')!.dispatchEvent(new Event('submit'));
    await settleZoneless(fixture);
  }

  beforeEach(() => {
    state$ = new BehaviorSubject<AchievementState>(EMPTY_STATE);
    refresh = vi.fn();
    checkPhrase = vi.fn<(phrase: string) => Observable<CheckPhraseResponse>>();
    enableSetting = signal<AppSettings | null>(null);

    const fakeAchievements = {
      state: state$.asObservable(),
      refresh,
      checkPhrase,
      docRawUrl: (id: string) => `/api/achievements/docs/${id}/raw`,
    };

    configureZoneless({
      imports: [AchievementsTabComponent],
      providers: [
        { provide: AchievementsService, useValue: fakeAchievements },
        { provide: SettingsStateService, useValue: { settingsSignal: enableSetting } },
      ],
    });
  });

  it('shows the loading placeholder until the first non-empty state arrives', async () => {
    await build();
    expect(el('.empty-state')?.textContent).toContain('Loading achievements');
    expect(el('.achievements-list')).toBeNull();
  });

  it('requests a refresh on init', async () => {
    await build();
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it('renders one row per achievement with name, tier, and description', async () => {
    await build();
    await pushState({ achievements: [makeEntry()] });

    expect(els('.achievement-row').length).toBe(1);
    expect(el('.achievement-name')?.textContent).toContain('Prolific Voter');
    expect(el('.achievement-tier')?.textContent?.trim()).toBe('Bronze');
    expect(el('.achievement-desc')?.textContent).toContain('Cast votes');
  });

  it('computes the total score as a sum of tier bit-values', async () => {
    await build();
    await pushState({
      achievements: [
        makeEntry({ id: 'a', tier_idx: 0 }), // 1
        makeEntry({ id: 'b', tier_idx: 2 }), // 4
        makeEntry({ id: 'c', tier_idx: -1 }), // 0 (locked)
      ],
    });
    expect(el('.achievements-total-value')?.textContent?.trim()).toBe('5');
  });

  it('formats the progress label toward the next tier', async () => {
    await build();
    await pushState({ achievements: [makeEntry({ counter: 5, tier_idx: 0, next_threshold: 25 })] });
    const label = el('.achievement-progress-label')?.textContent ?? '';
    expect(label).toContain('5 / 25');
    expect(label).toContain('Silver');
  });

  it('labels a maxed-out achievement instead of showing a next tier', async () => {
    await build();
    await pushState({
      achievements: [makeEntry({ counter: 500, tier_idx: 3, next_threshold: null })],
    });
    expect(el('.achievement-progress-label')?.textContent).toContain('maxed out');
  });

  it('marks a locked row and surfaces the unlock hint as a tooltip', async () => {
    await build();
    await pushState({ achievements: [makeEntry({ counter: 0, tier_idx: -1, next_threshold: 3 })] });
    expect(el('.achievement-row--locked')).not.toBeNull();
    expect(el('.achievement-tier')?.getAttribute('title')).toContain('Reach 3 to unlock Bronze');
  });

  it('expands the docs panel and lists the docs with read state', async () => {
    await build();
    await pushState({
      achievements: [makeEntry({ id: 'docs_read', name: 'Readme Reader', tier_idx: -1 })],
      docs: [
        { id: 'readme', name: 'README', path: 'README.md', read: true },
        { id: 'arch', name: 'Architecture', path: 'docs/ARCHITECTURE.md', read: false },
      ],
    });

    // Collapsed by default: no docs panel yet.
    expect(el('.docs-panel')).toBeNull();

    await expandRow();

    expect(el('.docs-panel')).not.toBeNull();
    expect(els('.docs-list-item').length).toBe(2);
    expect(el('.docs-list-item--read')).not.toBeNull();
    const link = el('.docs-list-link') as HTMLAnchorElement;
    expect(link.getAttribute('href')).toBe('/api/achievements/docs/readme/raw');
  });

  it('submits a phrase and reports a correct match', async () => {
    checkPhrase.mockReturnValue(
      of<CheckPhraseResponse>({
        matched: true,
        doc_id: 'readme',
        doc_name: 'README',
        already_read: false,
      }),
    );
    await build();
    await pushState({
      achievements: [makeEntry({ id: 'docs_read', tier_idx: -1 })],
      docs: [{ id: 'readme', name: 'README', path: 'README.md', read: false }],
    });
    await expandRow();
    await submitPhrase('all systems nominal');

    expect(checkPhrase).toHaveBeenCalledWith('all systems nominal');
    const status = el('.docs-phrase-status');
    expect(status?.getAttribute('data-kind')).toBe('success');
    expect(status?.textContent).toContain('README');
  });

  it('reports an already-credited phrase', async () => {
    checkPhrase.mockReturnValue(
      of<CheckPhraseResponse>({
        matched: true,
        doc_id: 'readme',
        doc_name: 'README',
        already_read: true,
      }),
    );
    await build();
    await pushState({
      achievements: [makeEntry({ id: 'docs_read', tier_idx: -1 })],
      docs: [{ id: 'readme', name: 'README', path: 'README.md', read: true }],
    });
    await expandRow();
    await submitPhrase('already known');

    expect(el('.docs-phrase-status')?.getAttribute('data-kind')).toBe('already');
  });

  it('reports a wrong phrase', async () => {
    checkPhrase.mockReturnValue(
      of<CheckPhraseResponse>({
        matched: false,
        doc_id: null,
        doc_name: null,
        already_read: false,
      }),
    );
    await build();
    await pushState({
      achievements: [makeEntry({ id: 'docs_read', tier_idx: -1 })],
      docs: [{ id: 'readme', name: 'README', path: 'README.md', read: false }],
    });
    await expandRow();
    await submitPhrase('nope');

    expect(el('.docs-phrase-status')?.getAttribute('data-kind')).toBe('wrong');
  });

  it('renders the media-types detail panel when expanded', async () => {
    await build();
    await pushState({
      achievements: [makeEntry({ id: 'media_types_touched', tier_idx: 0 })],
      media_types: [
        { id: 'audio', name: 'Audio', seen: true },
        { id: 'image', name: 'Image', seen: false },
      ],
    });
    await expandRow();

    expect(els('.ticks-list-item').length).toBe(2);
    expect(el('.ticks-list-item--on')).not.toBeNull();
  });

  it('renders the hours grid when expanded', async () => {
    await build();
    await pushState({
      achievements: [makeEntry({ id: 'hours_voted', tier_idx: 0 })],
      hours: [
        { hour: 0, seen: true },
        { hour: 13, seen: false },
      ],
    });
    await expandRow();

    const chips = els('.hour-chip');
    expect(chips.length).toBe(2);
    expect(chips[0].textContent?.trim()).toBe('00');
    expect(chips[1].textContent?.trim()).toBe('13');
    expect(el('.hour-chip--on')).not.toBeNull();
  });

  it('explains that counters are frozen when achievements are disabled', async () => {
    enableSetting.set({ enable_achievements: false } as AppSettings);
    await build();
    await pushState({
      achievements: [makeEntry({ id: 'votes_cast', counter: 0, tier_idx: -1, next_threshold: 3 })],
    });
    expect(el('.achievement-tier')?.getAttribute('title')).toContain('disabled in Settings');
  });
});
