import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it } from 'vitest';

import { PluginCheckboxComponent } from './plugin-checkbox.component';

describe('PluginCheckboxComponent', () => {
  let fixture: ComponentFixture<PluginCheckboxComponent>;

  const input = (): HTMLInputElement =>
    fixture.nativeElement.querySelector('input') as HTMLInputElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [PluginCheckboxComponent] }).compileComponents();
    fixture = TestBed.createComponent(PluginCheckboxComponent);
  });

  it('renders a real checkbox, not a text input', () => {
    // The whole point of issue #3851: eight surfaces fell through to a text
    // input, inviting the user to type "true" into a box.
    fixture.detectChanges();
    expect(input().type).toBe('checkbox');
  });

  it('reflects the current value', () => {
    fixture.componentRef.setInput('value', 'true');
    fixture.detectChanges();
    expect(input().checked).toBe(true);

    fixture.componentRef.setInput('value', 'false');
    fixture.detectChanges();
    expect(input().checked).toBe(false);
  });

  it('reflects a Python-style default', () => {
    fixture.componentRef.setInput('value', 'True');
    fixture.detectChanges();
    expect(input().checked).toBe(true);
  });

  it('emits the canonical strings when toggled', () => {
    const emitted: string[] = [];
    fixture.componentRef.setInput('value', 'false');
    fixture.detectChanges();
    fixture.componentInstance.valueChange.subscribe((v: string) => emitted.push(v));

    input().checked = true;
    input().dispatchEvent(new Event('change'));
    expect(emitted).toEqual(['true']);

    input().checked = false;
    input().dispatchEvent(new Event('change'));
    expect(emitted).toEqual(['true', 'false']);
  });

  it('wires inputId onto the box so the surrounding label targets it', () => {
    fixture.componentRef.setInput('inputId', 'sef-enable_email');
    fixture.detectChanges();
    expect(input().id).toBe('sef-enable_email');
  });
});
