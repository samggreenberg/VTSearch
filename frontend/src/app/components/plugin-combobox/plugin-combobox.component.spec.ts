import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it } from 'vitest';

import { FieldOptions } from '../../models/api.models';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';
import {
  PluginComboboxComponent,
  comboboxMatches,
  comboboxValueFor,
} from './plugin-combobox.component';

/** Two datasets that share a name, as in issue #5265. */
const TWINS: FieldOptions[] = [
  { value: 'uuid-1', label: 'my-cool-dataset' },
  { value: 'uuid-2', label: 'my-cool-dataset' },
  { value: 'uuid-3', label: 'birds' },
];

describe('comboboxValueFor', () => {
  it('maps a label exactly one option carries to that option\'s value', () => {
    expect(comboboxValueFor('birds', TWINS)).toBe('uuid-3');
  });

  it('keeps a label two options share as typed', () => {
    expect(comboboxValueFor('my-cool-dataset', TWINS)).toBe('my-cool-dataset');
  });

  it('keeps an option value, and anything unlisted, as typed', () => {
    expect(comboboxValueFor('uuid-2', TWINS)).toBe('uuid-2');
    expect(comboboxValueFor('hand-typed', TWINS)).toBe('hand-typed');
  });
});

describe('comboboxMatches', () => {
  it('offers every option for blank text', () => {
    expect(comboboxMatches('  ', TWINS)).toEqual(TWINS);
  });

  it('matches label or value, ignoring case', () => {
    expect(comboboxMatches('BIR', TWINS).map((o) => o.value)).toEqual(['uuid-3']);
    expect(comboboxMatches('uuid-2', TWINS).map((o) => o.value)).toEqual(['uuid-2']);
  });
});

describe('PluginComboboxComponent', () => {
  let fixture: ComponentFixture<PluginComboboxComponent>;
  let emitted: string[];

  const box = (): HTMLInputElement => fixture.nativeElement.querySelector('input') as HTMLInputElement;
  const rows = (): HTMLElement[] => Array.from(fixture.nativeElement.querySelectorAll('[role="option"]'));
  const detail = (): string | undefined =>
    (fixture.nativeElement.querySelector('.plugin-combobox-detail') as HTMLElement | null)?.textContent?.trim();

  const type = async (text: string): Promise<void> => {
    box().value = text;
    box().dispatchEvent(new Event('input'));
    await settleZoneless(fixture);
  };
  const key = async (name: string): Promise<KeyboardEvent> => {
    const event = new KeyboardEvent('keydown', { key: name, bubbles: true, cancelable: true });
    box().dispatchEvent(event);
    await settleZoneless(fixture);
    return event;
  };

  beforeEach(async () => {
    await configureZoneless({ imports: [PluginComboboxComponent] }).compileComponents();
    fixture = TestBed.createComponent(PluginComboboxComponent);
    fixture.componentRef.setInput('options', TWINS);
    emitted = [];
    // Stand in for the form: store what the widget emits and hand it back.
    fixture.componentInstance.valueChange.subscribe((v: string) => {
      emitted.push(v);
      fixture.componentRef.setInput('value', v);
    });
    await settleZoneless(fixture);
  });

  it('shows the picked option\'s label in the box, and its value beneath', async () => {
    fixture.componentRef.setInput('value', 'uuid-3');
    await settleZoneless(fixture);
    expect(box().value).toBe('birds');
    expect(detail()).toBe('uuid-3');
  });

  it('shows a value no option has as typed, with nothing beneath', async () => {
    fixture.componentRef.setInput('value', 'hand-typed');
    await settleZoneless(fixture);
    expect(box().value).toBe('hand-typed');
    expect(detail()).toBeUndefined();
  });

  it('adds no small print for plain string options', async () => {
    fixture.componentRef.setInput('options', [{ value: 'a', label: 'a' }]);
    fixture.componentRef.setInput('value', 'a');
    await settleZoneless(fixture);
    box().click();
    await settleZoneless(fixture);
    expect(detail()).toBeUndefined();
    expect(rows()[0].querySelector('.plugin-combobox-option-value')).toBeNull();
  });

  it('leads each row with the label, the value beneath it (#5265)', async () => {
    box().click();
    await settleZoneless(fixture);
    expect(rows().map((r) => r.querySelector('.plugin-combobox-option-label')!.textContent!.trim())).toEqual([
      'my-cool-dataset',
      'my-cool-dataset',
      'birds',
    ]);
    expect(rows().map((r) => r.querySelector('.plugin-combobox-option-value')!.textContent!.trim())).toEqual([
      'uuid-1',
      'uuid-2',
      'uuid-3',
    ]);
  });

  it('submits the UUID of whichever same-named dataset is picked (#5265)', async () => {
    box().click();
    await settleZoneless(fixture);
    rows()[1].click();
    await settleZoneless(fixture);
    expect(emitted).toEqual(['uuid-2']);
    expect(box().value).toBe('my-cool-dataset');
    expect(detail()).toBe('uuid-2');
    expect(rows()).toEqual([]);
  });

  it('emits typed text as typed, and filters the list by it', async () => {
    await type('bir');
    expect(emitted).toEqual(['bir']);
    expect(rows()).toHaveLength(1);
  });

  it('emits the value of a label only one option carries', async () => {
    await type('birds');
    expect(emitted).toEqual(['uuid-3']);
    expect(box().value).toBe('birds');
  });

  it('keeps a label two options share as typed text', async () => {
    await type('my-cool-dataset');
    expect(emitted).toEqual(['my-cool-dataset']);
  });

  it('opens from the keyboard, moves the highlight and picks with Enter', async () => {
    await key('ArrowDown');
    expect(rows()).toHaveLength(3);
    expect(box().getAttribute('aria-expanded')).toBe('true');
    expect(box().getAttribute('aria-activedescendant')).toBe(rows()[0].id);

    await key('ArrowDown');
    expect(rows()[1].classList).toContain('active');

    const enter = await key('Enter');
    expect(enter.defaultPrevented).toBe(true);
    expect(emitted).toEqual(['uuid-2']);
    expect(box().getAttribute('aria-expanded')).toBe('false');
  });

  it('lets Enter through when no row is highlighted', async () => {
    box().click();
    await settleZoneless(fixture);
    const enter = await key('Enter');
    expect(enter.defaultPrevented).toBe(false);
    expect(emitted).toEqual([]);
  });

  it('closes on Escape without letting it reach the dialog', async () => {
    let reachedDocument = 0;
    const listener = (): void => {
      reachedDocument++;
    };
    document.addEventListener('keydown', listener);
    try {
      await key('ArrowDown');
      await key('Escape');
      expect(rows()).toEqual([]);
      expect(reachedDocument).toBe(1); // the ArrowDown only

      // With the list shut, Escape is the dialog's again.
      await key('Escape');
      expect(reachedDocument).toBe(2);
    } finally {
      document.removeEventListener('keydown', listener);
    }
  });

  it('shows the picked label again once focus leaves mid-edit', async () => {
    // Typing an option's value picks it, but the box keeps what was typed
    // until the user moves on.
    await type('uuid-3');
    expect(emitted).toEqual(['uuid-3']);
    expect(box().value).toBe('uuid-3');

    box().dispatchEvent(new Event('blur'));
    await settleZoneless(fixture);
    expect(box().value).toBe('birds');
    expect(detail()).toBe('uuid-3');
    expect(rows()).toEqual([]);
  });

  it('wires inputId onto the box so the field\'s label targets it', async () => {
    fixture.componentRef.setInput('inputId', 'field-dataset');
    await settleZoneless(fixture);
    expect(box().id).toBe('field-dataset');
  });

  it('opens nothing while disabled', async () => {
    fixture.componentRef.setInput('disabled', true);
    await settleZoneless(fixture);
    expect(box().disabled).toBe(true);
    box().click();
    await settleZoneless(fixture);
    expect(rows()).toEqual([]);
  });
});
