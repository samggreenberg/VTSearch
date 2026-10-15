import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  fieldsComplete,
  LabelsetSourceFieldsComponent,
  needsValue,
} from './labelset-source-fields.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';
import type { ImporterField } from '../../models/api.models';

const path: ImporterField = { key: 'filepath', label: 'Path', field_type: 'server_path', required: true };
const withDefault: ImporterField = { key: 'fmt', label: 'Format', field_type: 'select', required: true, default: 'json', options: ['json', 'csv'] };
const hidden: ImporterField = { key: 'secret', label: 'Secret', field_type: 'text', required: true, hidden: true, default: 'x' };

describe('labelset source field helpers', () => {
  it('needs a value only for a required field with no default', () => {
    expect(needsValue(path)).toBe(true);
    expect(needsValue(withDefault)).toBe(false);
    expect(needsValue({ ...path, required: false })).toBe(false);
  });

  it('is complete once every field that needs a value has a non-blank one', () => {
    expect(fieldsComplete([path, withDefault], {})).toBe(false);
    expect(fieldsComplete([path, withDefault], { filepath: '   ' })).toBe(false);
    expect(fieldsComplete([path, withDefault], { filepath: '/b.json' })).toBe(true);
  });
});

describe('LabelsetSourceFieldsComponent', () => {
  let fixture: ComponentFixture<LabelsetSourceFieldsComponent>;

  beforeEach(async () => {
    await configureZoneless({ imports: [LabelsetSourceFieldsComponent] }).compileComponents();
    fixture = TestBed.createComponent(LabelsetSourceFieldsComponent);
    fixture.componentRef.setInput('fields', [path, withDefault, hidden]);
    fixture.componentRef.setInput('values', { filepath: '/a.json', fmt: 'json' });
    await settleZoneless(fixture);
  });

  it('renders the visible fields and marks the ones that need a value', () => {
    const el: HTMLElement = fixture.nativeElement;
    expect(el.querySelectorAll('.form-group').length).toBe(2);
    expect(el.querySelectorAll('.required').length).toBe(1);
    expect(el.textContent).not.toContain('Secret');
  });

  it('emits a fresh values object on every edit', () => {
    const emitted: Record<string, string>[] = [];
    fixture.componentInstance.valuesChange.subscribe((v) => emitted.push(v));
    fixture.componentInstance.set('filepath', '/b.json');
    expect(emitted).toEqual([{ filepath: '/b.json', fmt: 'json' }]);
  });
});
