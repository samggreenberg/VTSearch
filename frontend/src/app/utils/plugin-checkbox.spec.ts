import { describe, expect, it } from 'vitest';

import { checkboxValue, isChecked } from './plugin-checkbox';

describe('isChecked', () => {
  it('passes native booleans straight through', () => {
    expect(isChecked(true)).toBe(true);
    expect(isChecked(false)).toBe(false);
  });

  it('reads the canonical wire strings', () => {
    expect(isChecked('true')).toBe(true);
    expect(isChecked('false')).toBe(false);
  });

  it('is case- and whitespace-insensitive, so a Python-style default reads right', () => {
    // A plugin author writing `default="False"` (or `"True"`) is spelling a
    // Python bool, not our wire format; both halves must agree on what it
    // means or the box disagrees with the value run() receives.
    expect(isChecked('True')).toBe(true);
    expect(isChecked('TRUE')).toBe(true);
    expect(isChecked('  true  ')).toBe(true);
    expect(isChecked('False')).toBe(false);
  });

  it('treats a missing value as unchecked', () => {
    // How an omitted field with no default arrives.
    expect(isChecked(undefined)).toBe(false);
    expect(isChecked(null)).toBe(false);
    expect(isChecked('')).toBe(false);
  });

  it('treats anything unrecognised as unchecked, matching parse_checkbox', () => {
    expect(isChecked('yes')).toBe(false);
    expect(isChecked('1')).toBe(false);
    expect(isChecked(0)).toBe(false);
  });
});

describe('checkboxValue', () => {
  it('emits the strings the backend parses', () => {
    expect(checkboxValue(true)).toBe('true');
    expect(checkboxValue(false)).toBe('false');
  });

  it('round-trips through isChecked', () => {
    expect(isChecked(checkboxValue(true))).toBe(true);
    expect(isChecked(checkboxValue(false))).toBe(false);
  });
});
