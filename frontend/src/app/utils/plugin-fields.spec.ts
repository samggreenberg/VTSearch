import { describe, expect, it } from 'vitest';

import { visibleFields } from './plugin-fields';

/** A plugin field reduced to what this helper reads, plus a key to assert on. */
interface Field {
  key: string;
  hidden?: boolean;
}

const url: Field = { key: 'url_template', hidden: true };
const sep: Field = { key: 'separator' };
const max: Field = { key: 'max_items' };

describe('visibleFields', () => {
  it('drops fields marked hidden', () => {
    expect(visibleFields([url, sep, max]).map((f) => f.key)).toEqual([
      'separator',
      'max_items',
    ]);
  });

  it('keeps declaration order', () => {
    expect(visibleFields([max, url, sep]).map((f) => f.key)).toEqual([
      'max_items',
      'separator',
    ]);
  });

  it('keeps a field that spells hidden out as false', () => {
    const shown: Field = { key: 'a', hidden: false };
    expect(visibleFields([shown])).toHaveLength(1);
  });

  it('keeps a field from a server that predates `hidden`', () => {
    // The key is optional on the wire, so an older backend omits it
    // entirely; that must read as visible rather than as falsy-and-unclear.
    const legacy: Field = { key: 'a' };
    expect(visibleFields([legacy])).toHaveLength(1);
  });

  it('returns an empty list when every field is hidden', () => {
    // The case issue #4078 is about: the form renders nothing and the
    // exporter tab collapses to its action button.
    const other: Field = { key: 'b', hidden: true };
    expect(visibleFields([url, other])).toEqual([]);
  });

  it('treats null and undefined as no fields at all', () => {
    expect(visibleFields(null)).toEqual([]);
    expect(visibleFields(undefined)).toEqual([]);
  });

  it('does not mutate the list it is given', () => {
    const fields = [url, sep];
    visibleFields(fields);
    expect(fields).toHaveLength(2);
  });
});
