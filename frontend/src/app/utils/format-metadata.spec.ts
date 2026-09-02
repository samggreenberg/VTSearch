import { describe, expect, it } from 'vitest';
import { formatBytes, formatMetadataValue } from './format-metadata';

describe('formatBytes', () => {
  it('renders B / KB / MB across the two thresholds', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1023)).toBe('1023 B');
    expect(formatBytes(1024)).toBe('1.0 KB');
    expect(formatBytes(2048)).toBe('2.0 KB');
    expect(formatBytes(1024 * 1024 - 1)).toBe('1024.0 KB');
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB');
  });

  it('blanks a missing size rather than claiming zero', () => {
    expect(formatBytes(undefined)).toBe('');
    expect(formatBytes(null)).toBe('');
    expect(formatBytes(0)).toBe('0 B');
  });
});

describe('formatMetadataValue', () => {
  it('formats the three recognised numeric categories', () => {
    expect(formatMetadataValue('File Size', 2048)).toBe('2.0 KB');
    expect(formatMetadataValue('Duration', 3.5)).toBe('3.5s');
    expect(formatMetadataValue('Frequency', 44100)).toBe('44100 Hz');
  });

  it('falls through for unknown labels and mismatched types', () => {
    expect(formatMetadataValue('Other', 'hello')).toBe('hello');
    // A recognised label whose value is not a number is not reinterpreted.
    expect(formatMetadataValue('Duration', 'n/a')).toBe('n/a');
    expect(formatMetadataValue('File Size', null)).toBe('null');
  });
});
