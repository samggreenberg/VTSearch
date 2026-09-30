import type { FloorState } from '../generated/api-client/models/floor-state';
import { SAMPLE_RANGE, lineFloor } from '../testing/line-floor';
import {
  FLOOR_PRESETS,
  checkLabel,
  checkTitle,
  floorExplanation,
  floorSummary,
  isFloorPreset,
  lineFloorFrom,
  nearestFloorPreset,
  rangePercent,
  rangeTitle,
  staleNote,
  type FloorStatus,
} from './line-floor';

describe('line-floor (#4272, #4273)', () => {
  describe('lineFloorFrom', () => {
    it('reads the wire object, schedule included', () => {
      const wire: FloorState = {
        min_precision: 0.25,
        status: 'confirmed',
        count: 64,
        range: { lo: 0.55, hi: 1, labelled: 5, right: 5, stale: true },
        schedule: { candidate: 64, rounds: 2, picks: 5 },
      };
      expect(lineFloorFrom(wire)).toEqual({
        minPrecision: 0.25,
        status: 'confirmed',
        count: 64,
        range: { lo: 0.55, hi: 1, labelled: 5, right: 5, stale: true },
        schedule: { candidate: 64, rounds: 2, picks: 5 },
      });
    });

    it('reads an unchecked line with no range', () => {
      const wire = { min_precision: 0.1, status: 'unchecked', count: 128, range: null, schedule: { candidate: 128, rounds: 3, picks: 5 } };
      expect(lineFloorFrom(wire as unknown as FloorState)).toEqual({
        minPrecision: 0.1,
        status: 'unchecked',
        count: 128,
        range: null,
        schedule: { candidate: 128, rounds: 3, picks: 5 },
      });
    });

    it('reads a range with no stale flag as current', () => {
      const wire = { min_precision: 0.5, status: 'short', count: 32, range: { lo: 0.11, hi: 0.73, labelled: 5, right: 2 }, schedule: null };
      const floor = lineFloorFrom(wire as unknown as FloorState)!;
      expect(floor.range!.stale).toBe(false);
      expect(floor.schedule).toBeNull();
    });

    it('is null for a response with no detector behind it', () => {
      expect(lineFloorFrom(null)).toBeNull();
      expect(lineFloorFrom(undefined)).toBeNull();
    });

    it('reads a status outside the three states as no verdict', () => {
      const wire = { min_precision: 0.5, status: 'promised', count: 32, range: null, schedule: { candidate: 32, rounds: 1, picks: 5 } };
      expect(lineFloorFrom(wire as unknown as FloorState)).toBeNull();
    });
  });

  describe('the floor presets (#4298, #4317)', () => {
    it('offers three floors, left to right from false positives to false negatives, symmetric about the middle', () => {
      expect(FLOOR_PRESETS.map((p) => p.value)).toEqual([0.1, 0.5, 0.9]);
      expect(FLOOR_PRESETS[0].hint).toMatch(/^Toward false positives/);
      expect(FLOOR_PRESETS[2].hint).toMatch(/^Toward false negatives/);
    });

    it('knows a preset from a floor off the list', () => {
      expect(FLOOR_PRESETS.every((p) => isFloorPreset(p.value))).toBe(true);
      expect(isFloorPreset(0.25)).toBe(false);
      expect(isFloorPreset(0.75)).toBe(false);
    });

    it.each<[number, number]>([
      [0.01, 0.1],
      [0.25, 0.1],
      [0.29, 0.1],
      [0.31, 0.5],
      [0.6, 0.5],
      [0.75, 0.9],
      [1, 0.9],
    ])('snaps %s to the nearest preset, %s', (stored, snapped) => {
      expect(nearestFloorPreset(stored).value).toBe(snapped);
    });
  });

  describe('the floor control copy (#4246, #4273)', () => {
    it("shows a check's range as a number", () => {
      expect(rangePercent(SAMPLE_RANGE)).toBe('11–73%');
    });

    it.each<[FloorStatus, string]>([
      ['confirmed', 'Confirmed · likely 55–100% right (checked 5) · 32 kept'],
      ['short', 'Fell short · likely 11–73% right (checked 5) · top 32 kept'],
      ['unchecked', 'Top 32 kept, unchecked'],
    ])('summarises %s', (status, expected) => {
      expect(floorSummary(lineFloor(status))).toBe(expected);
    });

    it.each<FloorStatus>(['confirmed', 'short', 'unchecked'])('neither names nor numbers the floor (%s, #4298, #4317)', (status) => {
      for (const minPrecision of [0.1, 0.25, 0.5, 0.9]) {
        const floor = lineFloor(status, { minPrecision });
        const target = `${Math.round(minPrecision * 100)}%`;
        expect(floorSummary(floor)).not.toContain(target);
        expect(floorExplanation(floor)).not.toContain(target);
        expect(floorSummary(floor)).not.toMatch(/Centered|Complete|Correct/);
        expect(floorExplanation(floor)).not.toMatch(/Centered|Complete|Correct/);
      }
    });

    it('reads the count from the result, never from the preset', () => {
      expect(floorSummary(lineFloor('confirmed', { minPrecision: 0.1, count: 64 }))).toContain('64 kept');
      expect(floorSummary(lineFloor('confirmed', { minPrecision: 0.1, count: 128 }))).toContain('128 kept');
      expect(floorSummary(lineFloor('unchecked', { minPrecision: 0.1, count: 128 }))).toBe(
        'Top 128 kept, unchecked',
      );
    });

    it('carries the range wide as the check left it: a five-pick range at 50% is 0.62 wide', () => {
      const text = floorSummary(lineFloor('short'))!;
      expect(text).toContain('11–73%');
      expect(text).toContain('(checked 5)');
    });

    it('has nothing to summarise, explain or check without a verdict', () => {
      expect(floorSummary(null)).toBeNull();
      expect(floorExplanation(null)).toBeNull();
      expect(checkLabel(null)).toBeNull();
    });

    it('explains a confirmed line by its check, with the range from the picks alone', () => {
      const why = floorExplanation(lineFloor('confirmed'))!;
      expect(why).toContain('5 random picks from the 32 items the line keeps found 5 right');
      expect(why).toContain('likely 55–100% of them are: enough for the threshold');
      expect(why).not.toMatch(/estimate/i);
    });

    it('explains an unchecked line as unmeasured, pointing at no check (Find offers none, #4317)', () => {
      const why = floorExplanation(lineFloor('unchecked', { count: 128, schedule: { candidate: 128, rounds: 3, picks: 5 } }))!;
      expect(why).toContain('Unchecked: the line keeps the top 128');
      expect(why).not.toMatch(/likely|a check of/i);
    });

    it('explains a short check by how close it got, naming no cause', () => {
      const why = floorExplanation(lineFloor('short'))!;
      expect(why).toContain('5 random picks from the top 32');
      expect(why).toContain('likely 11–73% of them are: short of the threshold');
      // The app can't tell a sparse corpus from a weak model (owner, 2026-09-29).
      expect(why).not.toMatch(/sparse|weak|evidence|too few|model/i);
    });

    it.each<FloorStatus>(['short', 'confirmed'])('says a stale %s range is stale in the tooltip, and only there', (status) => {
      const fresh = lineFloor(status);
      const stale = lineFloor(status, { range: { ...fresh.range!, stale: true } });
      expect(floorSummary(stale)).toBe(floorSummary(fresh));
      expect(floorExplanation(stale)).toBe(floorExplanation(fresh) + staleNote(stale.range));
      expect(floorExplanation(stale)).toContain('Measured before your later votes');
      expect(floorExplanation(fresh)).not.toContain('later votes');
    });
  });

  describe('the check affordance (#4273)', () => {
    it.each<[number, number, string]>([
      [0.1, 5, 'Check 5 picks'],
      [0.5, 5, 'Check 5 picks'],
      [0.9, 5, 'Check 5 picks'],
    ])('at %s reads "%s picks" off the schedule', (x, picks, label) => {
      const floor = lineFloor('unchecked', { minPrecision: x, schedule: { candidate: 32, rounds: 3, picks } });
      expect(checkLabel(floor)).toBe(label);
    });

    it('is offered in every state: after a finished check it runs a fresh one', () => {
      for (const status of ['unchecked', 'confirmed', 'short'] as const) {
        expect(checkLabel(lineFloor(status))).toBe('Check 5 picks');
      }
    });

    it('says what a check does and that its votes are votes', () => {
      const title = checkTitle(lineFloor('unchecked', { schedule: { candidate: 64, rounds: 4, picks: 5 } }));
      expect(title).toContain('5 random picks a band, walking the list from the top 64');
      expect(title).toContain('deeper while the list stays right enough');
      expect(title).toContain('ordinary votes');
    });
  });

  describe('rangeTitle', () => {
    it('names the picks the range comes from', () => {
      expect(rangeTitle(SAMPLE_RANGE)).toBe('Likely 11–73% right, from 5 random picks (2 right).');
    });

    it('adds the stale sentence for a stale range', () => {
      expect(rangeTitle({ ...SAMPLE_RANGE, stale: true })).toContain('Measured before your later votes');
    });
  });
});
