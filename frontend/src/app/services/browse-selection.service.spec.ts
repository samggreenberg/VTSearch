import { Component, computed, inject } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BrowseSelectionService } from './browse-selection.service';
import { configureZoneless } from '../testing/zoneless-testbed';
import { settleZoneless } from '../testing/settle-resource';

describe('BrowseSelectionService', () => {
  let service: BrowseSelectionService;

  beforeEach(() => {
    service = new BrowseSelectionService();
  });

  it('starts empty', () => {
    expect(service.size).toBe(0);
    expect(service.ids()).toEqual([]);
    expect(service.version()).toBe(0);
  });

  it('addAll adds new ids and bumps the version once per changing call', () => {
    service.addAll([1, 2, 3]);
    expect(service.size).toBe(3);
    expect(service.ids()).toEqual([1, 2, 3]);
    expect(service.version()).toBe(1);

    // Re-adding ids already present is a no-op and must NOT bump the version.
    service.addAll([2, 3]);
    expect(service.size).toBe(3);
    expect(service.version()).toBe(1);
  });

  it('remove/removeAll only bump when something actually leaves the set', () => {
    service.addAll([1, 2, 3]);
    const v = service.version();

    service.remove(99); // not present
    expect(service.version()).toBe(v);

    service.remove(2);
    expect(service.has(2)).toBe(false);
    expect(service.version()).toBe(v + 1);

    service.removeAll([1, 1, 1]); // present once; dedup of the request
    expect(service.size).toBe(1);
    expect(service.version()).toBe(v + 2);
  });

  it('toggleBin selects a fully-unselected bin and clears a partial/full one', () => {
    service.toggleBin([1, 2, 3]); // none selected → select all
    expect(service.selectedCountIn([1, 2, 3])).toBe(3);

    service.remove(2); // make it partial
    service.toggleBin([1, 2, 3]); // partial → clear all members
    expect(service.selectedCountIn([1, 2, 3])).toBe(0);

    service.toggleBin([]); // empty bin is a no-op
    expect(service.size).toBe(0);
  });

  it('clear drops everything and only bumps when non-empty', () => {
    const v0 = service.version();
    service.clear(); // already empty → no bump
    expect(service.version()).toBe(v0);

    service.addAll([4, 5]);
    const v1 = service.version();
    service.clear();
    expect(service.size).toBe(0);
    expect(service.version()).toBe(v1 + 1);
  });

  it('arms and consumes the one-shot survive-projection-change mark', () => {
    expect(service.consumeSurviveProjectionChange()).toBe(false);
    service.markSurviveProjectionChange();
    expect(service.consumeSurviveProjectionChange()).toBe(true);
    // One-shot: a second consume returns false.
    expect(service.consumeSurviveProjectionChange()).toBe(false);
  });
});

/**
 * Zoneless staleness canary (docs/plans/zoneless-migration.md, Phases 0.3/0.4 +
 * 1.4). `version` is a signal as of Phase 1.4, so a view that reads it tracks
 * every selection mutation. This component mirrors how the real consumers
 * (browse-canvas redraw, bin-popup re-highlight, selection panel refresh) react:
 * it derives state from `version()` and renders it. Driving the service through
 * its production API and asserting the rendered DOM after `settleZoneless()` —
 * with no manual `detectChanges()` — proves the signal write schedules CD.
 */
@Component({
  selector: 'app-selection-canary',
  standalone: true,
  template: `<span class="count">{{ count() }}</span>`,
})
class SelectionCanaryComponent {
  private selection = inject(BrowseSelectionService);
  // Read `version()` so a selection mutation schedules CD; `size` is plain.
  readonly count = computed(() => {
    this.selection.version();
    return this.selection.size;
  });
}

describe('BrowseSelectionService (zoneless view canary)', () => {
  let fixture: ComponentFixture<SelectionCanaryComponent>;
  let selection: BrowseSelectionService;

  beforeEach(async () => {
    configureZoneless({
      imports: [SelectionCanaryComponent],
      providers: [BrowseSelectionService],
    });
    fixture = TestBed.createComponent(SelectionCanaryComponent);
    selection = TestBed.inject(BrowseSelectionService);
    await settleZoneless(fixture);
  });

  function countText(): string | null {
    return fixture.nativeElement.querySelector('.count')?.textContent?.trim() ?? null;
  }

  it('renders the initial empty selection', () => {
    expect(countText()).toBe('0');
  });

  it('repaints when the selection changes, with no manual detectChanges', async () => {
    selection.addAll([1, 2, 3]);
    await settleZoneless(fixture);
    expect(countText()).toBe('3');

    selection.remove(2);
    await settleZoneless(fixture);
    expect(countText()).toBe('2');

    selection.clear();
    await settleZoneless(fixture);
    expect(countText()).toBe('0');
  });
});
