import { describe, expect, it } from 'vitest';
import { Subject } from 'rxjs';

import { DynamicFieldOptions } from './dynamic-field-options';
import type { FieldOptions, ImporterField } from '../models/api.models';

/**
 * Coverage for the cancellation/ordering guard shared by every plugin-field
 * form with ``dynamic_options`` selects (issue #2965). The four copies of this
 * fetch used to write whatever landed last, so a slow response could repopulate
 * a dropdown the user had already navigated away from — and silently
 * auto-select a value invalid for the importer now on screen.
 */
describe('DynamicFieldOptions', () => {
  function field(partial: Partial<ImporterField> & { key: string }): ImporterField {
    return { field_type: 'select', dynamic_options: true, ...partial } as ImporterField;
  }

  /** A controller whose fetches resolve manually, newest response last. */
  function harness() {
    const pending: Subject<{ options: FieldOptions[] }>[] = [];
    const requests: { key: string; values: Record<string, string> }[] = [];
    const controller = new DynamicFieldOptions((key, values) => {
      requests.push({ key, values });
      const subject = new Subject<{ options: FieldOptions[] }>();
      pending.push(subject);
      return subject;
    });
    const resolve = (index: number, options: FieldOptions[]) => {
      pending[index].next({ options });
      pending[index].complete();
    };
    const fail = (index: number, err: unknown) => pending[index].error(err);
    return { controller, requests, resolve, fail };
  }

  it('applies a response and auto-selects the first option for a required strict select', () => {
    const { controller, resolve } = harness();
    const values: Record<string, string> = {};
    const f = field({ key: 'sheet', required: true });

    controller.refresh(f, values);
    expect(controller.loading()['sheet']).toBe(true);
    resolve(0, [{ value: 's1', label: 'Sheet 1' }]);

    expect(controller.optionsFor(f)).toEqual([{ value: 's1', label: 'Sheet 1' }]);
    expect(controller.loading()['sheet']).toBe(false);
    expect(values['sheet']).toBe('s1');
  });

  it('drops an out-of-order response for the same field key', () => {
    const { controller, resolve } = harness();
    const values: Record<string, string> = {};
    const f = field({ key: 'q', required: true });

    // Two keystrokes in flight for the same field.
    controller.refresh(f, values);
    controller.refresh(f, values);

    // The newer request answers first, then the older one straggles in.
    resolve(1, [{ value: 'new', label: 'New' }]);
    resolve(0, [{ value: 'old', label: 'Old' }]);

    expect(controller.optionsFor(f)).toEqual([{ value: 'new', label: 'New' }]);
    expect(values['q']).toBe('new');
  });

  it('drops a late response after reset(), leaving the new form untouched', () => {
    const { controller, resolve } = harness();
    const importerAValues: Record<string, string> = {};
    const f = field({ key: 'path', required: true });

    // Importer A dispatches a slow fetch; the user goes Back and picks
    // importer B, which shares the `path` field key.
    controller.refresh(f, importerAValues);
    controller.reset();
    const importerBValues: Record<string, string> = {};

    resolve(0, [{ value: '/a/only', label: 'A only' }]);

    expect(controller.optionsFor(f)).toEqual([]);
    expect(importerAValues['path']).toBeUndefined();
    expect(importerBValues['path']).toBeUndefined();
  });

  it("drops a late response once another importer's fetch for the same key is in flight", () => {
    const { controller, resolve } = harness();
    const valuesA: Record<string, string> = {};
    const f = field({ key: 'dataset', required: true });

    controller.refresh(f, valuesA); // importer A
    controller.reset();
    const valuesB: Record<string, string> = {};
    controller.refresh(f, valuesB); // importer B, same field key

    resolve(1, [{ value: 'b1', label: 'B one' }]);
    resolve(0, [{ value: 'a1', label: 'A one' }]); // A straggles in

    expect(controller.optionsFor(f)).toEqual([{ value: 'b1', label: 'B one' }]);
    expect(valuesB['dataset']).toBe('b1');
    expect(valuesA['dataset']).toBeUndefined();
  });

  it('drops a superseded error response instead of clobbering fresh options', () => {
    const { controller, resolve, fail } = harness();
    const values: Record<string, string> = {};
    const f = field({ key: 'q' });

    controller.refresh(f, values);
    controller.refresh(f, values);
    resolve(1, [{ value: 'fresh', label: 'Fresh' }]);
    fail(0, { error: { message: 'boom' } });

    expect(controller.error()['q']).toBe('');
    expect(controller.optionsFor(f)).toEqual([{ value: 'fresh', label: 'Fresh' }]);
    expect(controller.loading()['q']).toBe(false);
  });

  it('surfaces a fetch failure inline and empties the option list', () => {
    const { controller, fail } = harness();
    const f = field({ key: 'q' });

    controller.refresh(f, {});
    fail(0, { error: { message: 'boom' } });

    expect(controller.error()['q']).toBe('boom');
    expect(controller.loading()['q']).toBe(false);
    expect(controller.optionsFor(f)).toEqual([]);
  });

  it('keeps a typed free-text value the refreshed options omit', () => {
    const { controller, resolve } = harness();
    const values: Record<string, string> = { q: 'hand-typed' };
    const f = field({ key: 'q', required: true, allow_free_text: true });

    controller.refresh(f, values);
    resolve(0, [{ value: 'a', label: 'A' }]);

    expect(values['q']).toBe('hand-typed');
  });

  it('clears a strict-select value the refreshed options omit', () => {
    const { controller, resolve } = harness();
    const values: Record<string, string> = { q: 'stale' };
    const f = field({ key: 'q' });

    controller.refresh(f, values);
    resolve(0, [{ value: 'a', label: 'A' }]);

    expect(values['q']).toBe('');
  });

  it('coerces static string options into {value,label} pairs', () => {
    const { controller } = harness();
    const staticSelect = { key: 's', field_type: 'select', options: ['x', 'y'] } as ImporterField;

    expect(controller.optionsFor(staticSelect)).toEqual([
      { value: 'x', label: 'x' },
      { value: 'y', label: 'y' },
    ]);
  });

  it('refreshes only the fields that depend on the changed key, blanking them first', () => {
    const { controller, requests } = harness();
    const values: Record<string, string> = { doc: 'doc-1', tab: 'stale', other: 'keep' };
    const fields = [
      field({ key: 'doc' }),
      field({ key: 'tab', depends_on: ['doc'] }),
      field({ key: 'other', depends_on: ['something-else'] }),
    ];

    controller.refreshDependentsOf('doc', fields, values);

    expect(requests.map((r) => r.key)).toEqual(['tab']);
    expect(values['tab']).toBe('');
    expect(values['other']).toBe('keep');
    // The request carries a snapshot, so later edits can't mutate it in flight.
    expect(requests[0].values).toEqual({ doc: 'doc-1', tab: '', other: 'keep' });
  });

  it('fetches every dynamic field of a freshly-selected importer', () => {
    const { controller, requests } = harness();
    const fields = [
      field({ key: 'a' }),
      { key: 'b', field_type: 'text' } as ImporterField,
      field({ key: 'c' }),
    ];

    controller.refreshAll(fields, {});

    expect(requests.map((r) => r.key)).toEqual(['a', 'c']);
  });

  it('runs the onApplied hook only for a response that is still current', () => {
    const pending: Subject<{ options: FieldOptions[] }>[] = [];
    let applied = 0;
    const controller = new DynamicFieldOptions(
      () => {
        const subject = new Subject<{ options: FieldOptions[] }>();
        pending.push(subject);
        return subject;
      },
      () => {
        applied += 1;
      },
    );
    const f = field({ key: 'q' });

    controller.refresh(f, {});
    controller.refresh(f, {});
    pending[1].next({ options: [] });
    pending[0].next({ options: [] }); // superseded

    expect(applied).toBe(1);
  });
});
