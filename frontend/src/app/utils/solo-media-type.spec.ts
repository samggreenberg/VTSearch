import { describe, expect, it } from 'vitest';
import { allSoloMediaType, isSoloMediaType } from './solo-media-type';

describe('isSoloMediaType', () => {
  it('is true for the type the server is locked to', () => {
    expect(isSoloMediaType('image', 'image')).toBe(true);
  });

  it('is false for another type, which the lock does not account for', () => {
    expect(isSoloMediaType('image', 'audio')).toBe(false);
  });

  it('is false with no lock, or no type to name', () => {
    expect(isSoloMediaType(null, 'image')).toBe(false);
    expect(isSoloMediaType(undefined, 'image')).toBe(false);
    expect(isSoloMediaType('', 'image')).toBe(false);
    expect(isSoloMediaType('image', undefined)).toBe(false);
    expect(isSoloMediaType('image', '')).toBe(false);
  });
});

describe('allSoloMediaType', () => {
  it('is true when every row is of the locked type', () => {
    expect(allSoloMediaType('image', ['image', 'image'])).toBe(true);
  });

  it('is false when one row is of another type', () => {
    expect(allSoloMediaType('image', ['image', 'audio', 'image'])).toBe(false);
  });

  it('is true for no rows, whose column would show nothing either way', () => {
    expect(allSoloMediaType('image', [])).toBe(true);
  });

  it('is false with no lock', () => {
    expect(allSoloMediaType(null, ['image'])).toBe(false);
    expect(allSoloMediaType('', [])).toBe(false);
  });
});
