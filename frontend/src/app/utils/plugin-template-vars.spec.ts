import { describe, expect, it } from 'vitest';

import { resolveTemplateVars, sanitizeTemplateValue } from './plugin-template-vars';

const CTX = {
  detectorName: 'MyDetector',
  detectorId: 'det-1',
  username: 'alice',
  now: new Date(Date.UTC(2026, 7, 20, 4, 5, 6)),
};

describe('sanitizeTemplateValue', () => {
  it.each([
    ['plain', 'plain'],
    ['a/b', 'a_b'],
    ['a\\b', 'a_b'],
    ['..', '_'],
    ['.', '_'],
    ['...', '_'],
    ['', '_'],
    ['../../etc/passwd', '.._.._etc_passwd'],
  ])('sanitizes %s -> %s', (input, expected) => {
    expect(sanitizeTemplateValue(input)).toBe(expected);
  });
});

describe('resolveTemplateVars', () => {
  it('substitutes a declared detector_name', () => {
    expect(resolveTemplateVars('{detector_name}', ['detector_name'], CTX)).toBe('MyDetector');
  });

  it('leaves a placeholder the field does not declare', () => {
    // The portable_detector exporter deliberately withholds `detector_name`
    // so it can substitute per-detector itself; the preview must respect that.
    expect(resolveTemplateVars('data/{detector_name}.zip', ['YYYYMMDD'], CTX)).toBe(
      'data/{detector_name}.zip',
    );
  });

  it('leaves a declared placeholder that cannot be resolved yet', () => {
    expect(resolveTemplateVars('{detector_name}', ['detector_name'], { ...CTX, detectorName: '' })).toBe(
      '{detector_name}',
    );
  });

  it('formats date variables in UTC', () => {
    expect(
      resolveTemplateVars(
        '{YYYY}.{MM}.{DD}/{YYYYMMDD}/{YYYYMMDD-HHMMSS}',
        ['YYYY', 'MM', 'DD', 'YYYYMMDD', 'YYYYMMDD-HHMMSS'],
        CTX,
      ),
    ).toBe('2026.08.20/20260820/20260820-040506');
  });

  it('substitutes every occurrence of a placeholder', () => {
    expect(resolveTemplateVars('{username}/{username}', ['username'], CTX)).toBe('alice/alice');
  });

  it('sanitizes a resolved value the way the server does', () => {
    expect(
      resolveTemplateVars('data/{detector_name}.json', ['detector_name'], {
        ...CTX,
        detectorName: '../evil',
      }),
    ).toBe('data/.._evil.json');
  });

  it('leaves non-text-like field types alone', () => {
    expect(resolveTemplateVars('{detector_name}', ['detector_name'], CTX, 'number')).toBe(
      '{detector_name}',
    );
    expect(resolveTemplateVars('{detector_name}', ['detector_name'], CTX, 'server_path')).toBe(
      'MyDetector',
    );
  });

  it('is a no-op without declared vars or without a value', () => {
    expect(resolveTemplateVars('{detector_name}', [], CTX)).toBe('{detector_name}');
    expect(resolveTemplateVars('{detector_name}', undefined, CTX)).toBe('{detector_name}');
    expect(resolveTemplateVars('', ['detector_name'], CTX)).toBe('');
  });

  it('ignores a variable name it does not know', () => {
    expect(resolveTemplateVars('{bogus}', ['bogus'], CTX)).toBe('{bogus}');
  });
});
