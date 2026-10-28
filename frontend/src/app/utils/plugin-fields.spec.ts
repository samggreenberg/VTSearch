import { describe, expect, it } from 'vitest';

import { inactiveFieldKeys, visibleFields } from './plugin-fields';

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

/** A plugin field reduced to what the `visible_when` rules read. */
interface ConditionalField {
  key: string;
  field_type?: string;
  default?: string;
  hidden?: boolean;
  visible_when?: Record<string, string[]>;
}

/** The issue's own example (#5343): a checkbox that switches an e-mail box on. */
const enableEmail: ConditionalField = { key: 'enable_email', field_type: 'checkbox', default: 'false' };
const email: ConditionalField = { key: 'email', field_type: 'email', visible_when: { enable_email: ['true'] } };

function shownKeys(fields: ConditionalField[], values: Record<string, unknown>): string[] {
  return visibleFields(fields, values).map((f) => f.key);
}

// Every case here has a twin in tests_lib/core/test_plugin_field_visible_when.py:
// the form must hide exactly what the server stops requiring.
describe('visible_when', () => {
  it('hides a field while its checkbox is unticked', () => {
    expect(shownKeys([enableEmail, email], { enable_email: 'false' })).toEqual(['enable_email']);
  });

  it('shows it once the box is ticked', () => {
    expect(shownKeys([enableEmail, email], { enable_email: 'true' })).toEqual(['enable_email', 'email']);
  });

  it('reads a checkbox the way isChecked does', () => {
    // A boolean, or `"True"` from a Python-style default, is a ticked box.
    expect(shownKeys([enableEmail, email], { enable_email: true })).toContain('email');
    expect(shownKeys([enableEmail, email], { enable_email: 'True' })).toContain('email');
  });

  it('reads a blank controller as its default', () => {
    const on: ConditionalField = { ...enableEmail, default: 'true' };
    expect(shownKeys([on, email], {})).toContain('email');
    expect(shownKeys([on, email], { enable_email: '  ' })).toContain('email');
    expect(shownKeys([enableEmail, email], {})).not.toContain('email');
  });

  it('accepts any of the values listed under one key', () => {
    const source: ConditionalField = { key: 'source', field_type: 'select' };
    const bucket: ConditionalField = { key: 'bucket', field_type: 'text', visible_when: { source: ['s3', 'gcs'] } };
    expect(shownKeys([source, bucket], { source: 'gcs' })).toContain('bucket');
    expect(shownKeys([source, bucket], { source: ' s3 ' })).toContain('bucket');
    expect(shownKeys([source, bucket], { source: 'local' })).not.toContain('bucket');
  });

  it('requires every key to match', () => {
    const mode: ConditionalField = { key: 'mode', field_type: 'select' };
    const both: ConditionalField = {
      key: 'smtp_port',
      field_type: 'number',
      visible_when: { enable_email: ['true'], mode: ['smtp'] },
    };
    const fields = [enableEmail, mode, both];
    expect(shownKeys(fields, { enable_email: 'true', mode: 'smtp' })).toContain('smtp_port');
    expect(shownKeys(fields, { enable_email: 'false', mode: 'smtp' })).not.toContain('smtp_port');
    expect(shownKeys(fields, { enable_email: 'true', mode: 'api' })).not.toContain('smtp_port');
  });

  it('judges a controller its own condition switches off as its default', () => {
    // `format` is hidden while e-mail is off, so the plugin receives its
    // default — and `subject` follows that default, not the stale "html".
    const format: ConditionalField = {
      key: 'format',
      field_type: 'select',
      default: 'text',
      visible_when: { enable_email: ['true'] },
    };
    const subject: ConditionalField = { key: 'subject', field_type: 'text', visible_when: { format: ['html'] } };
    const fields = [enableEmail, format, subject];
    expect(shownKeys(fields, { enable_email: 'false', format: 'html' })).toEqual(['enable_email']);
    expect(shownKeys(fields, { enable_email: 'true', format: 'html' })).toEqual(['enable_email', 'format', 'subject']);
  });

  it('lets a hidden field control a visible one', () => {
    const fixed: ConditionalField = { key: 'tier', field_type: 'text', hidden: true, default: 'pro' };
    const extra: ConditionalField = { key: 'extra', field_type: 'text', visible_when: { tier: ['pro'] } };
    expect(shownKeys([fixed, extra], {})).toEqual(['extra']);
  });

  it('ignores a key that names no field', () => {
    const typo: ConditionalField = { key: 'email', field_type: 'email', visible_when: { enabel_email: ['true'] } };
    expect(shownKeys([enableEmail, typo], { enable_email: 'false' })).toContain('email');
  });

  it('settles a cycle instead of recursing forever', () => {
    const a: ConditionalField = { key: 'a', field_type: 'text', default: 'x', visible_when: { b: ['y'] } };
    const b: ConditionalField = { key: 'b', field_type: 'text', default: 'y', visible_when: { a: ['x'] } };
    expect(shownKeys([a, b], {})).toEqual(['a', 'b']);
  });

  it('applies only `hidden` when no values are given', () => {
    expect(visibleFields([enableEmail, email]).map((f) => f.key)).toEqual(['enable_email', 'email']);
  });

  it('reports the inactive keys, hidden fields included', () => {
    const hiddenOff: ConditionalField = { ...email, key: 'cc', hidden: true };
    expect([...inactiveFieldKeys([enableEmail, email, hiddenOff], { enable_email: 'false' })].sort()).toEqual([
      'cc',
      'email',
    ]);
    expect(inactiveFieldKeys([enableEmail, email], { enable_email: 'true' }).size).toBe(0);
    expect(inactiveFieldKeys(null, {}).size).toBe(0);
  });
});
